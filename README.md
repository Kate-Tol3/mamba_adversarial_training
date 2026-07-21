# Adversarial Robustness for Mamba

Курсовая работа по повышению устойчивости моделей на основе архитектуры **Mamba** (Selective State Space Model) к состязательным атакам, с учётом её рекуррентной природы.

Классические методы adversarial-защиты (state-of-the-art для CNN и Transformer) не учитывают, что в Mamba небольшое возмущение на входе может лавинообразно накапливаться в скрытом состоянии по мере рекуррентного распространения по последовательности. Работа предлагает метод защиты, который явно контролирует это накопление, и фреймворк для воспроизводимого сравнения Mamba / Transformer / CNN под атаками на общей методологической основе.

## Структура фреймворка

Модульная архитектура из пяти блоков объединённых в единый Jupyter-ноутбук [`Mamba_adversarial_robustness.ipynb`](./Mamba_adversarial_robustness.ipynb):

1. **Config & Data** — конфигурация экспериментов через `@dataclass` (`DatasetConfig`, `TrainConfig`, `AttackConfig`) и загрузка/токенизация данных (изображения, текст, аудио) в единый интерфейс `DataLoader`.
2. **Models** — унифицированные классификаторы `MambaClassifier`, `TransformerClassifier`, `CNNClassifier` с общим интерфейсом `forward`.
3. **Attack Manager** — генерация возмущений: обёртки над `torchattacks`/`AutoAttack` (PGD, FGSM, CW и др.) плюс кастомные атаки (HiSPA, Sliding Patch). Работа через `EmbeddingModelWrapper` (паттерн «Адаптер»), позволяющий дифференцировать градиент даже для дискретных входов (текст).
4. **Trainer** — обучение в чистом и состязательном режимах, early stopping, компонент SA-AT/Attack-Mix.
5. **Evaluator** — расчёт метрик (Accuracy, AdvAcc, Gap, F1, State Drift через forward hooks), профилирование по времени/памяти.

## Метод

**State-Aware Adversarial Training (SA-AT).** К обычной задаче adversarial training добавляется регуляризатор дрейфа скрытого состояния: модель обучается одновременно на чистых и состязательных эмбеддингах, а loss дополняется MSE между hidden state на чистом и на атакованном входе —

```
loss = clean_loss + adv_alpha · adv_loss + sa_at_lambda · MSE(hidden_adv, hidden_clean)
```

Реализация — `train_one_epoch_adversarial()`. Регуляризатор считается только для моделей, у которых `forward_from_embeddings` умеет отдавать скрытое состояние (`return_hidden=True`) — в текущем коде это только `UniversalMambaClassifier`, поэтому SA-AT и метрика State Drift применимы исключительно к Mamba, у Transformer/CNN их нет.

**Attack-Mix.** Гибридное обучение: на первых двух эпохах используется быстрая R-FGSM-атака, дальше на каждой эпохе случайно выбирается одна из `{pgd, r_fgsm, sliding_patch, hispa}` — так модель не переобучается под один конкретный тип возмущения. Дополнительно опция `two_stage` в `AttackConfig` добавляет случайный старт перед градиентными шагами (random-start PGD, снижает переобучение под конкретную начальную точку атаки).

**Входная защита** (независимо включается флагом `use_defense` в `build_model`):
- `pre_filter` — depthwise Conv1d перед основным блоком, сглаживающий резкие локальные выбросы (полезно против Sliding Patch);
- `inference_noise_std` — гауссов шум, добавляемый к эмбеддингам только в `eval`-режиме, сбивает точность атаки, не мешая обучению;
- `stabilize_mamba_delta()` — обрезка весов `dt_proj` (шаг дискретизации Δ) после каждой эпохи, только для Mamba — предотвращает нестабильность, вызванную резким изменением Δ под атакой.

## Атаки (`get_adversarial_examples`)

| Атака | Тип | Комментарий |
|---|---|---|
| PGD | градиентная, итеративная | через `torchattacks` для continuous-входа, кастомная `_pgd_embedding` для входа-эмбеддинга |
| R-FGSM | градиентная, одношаговая | random start + FGS-шаг |
| HiSPA | градиентная, итеративная | кастомная реализация (`apply_hispa_attack`), фиксированный шаг eps/steps |
| CW | оптимизационная | через `torchattacks.CW` |
| AutoAttack / AutoAttack-seq | ансамбль атак | `autoattack` для continuous-входа, self-реализованный ансамбль PGD+Square для эмбеддингов |
| Square (`gradient_free`) | безградиентная | через `torchattacks.Square` |
| Sliding Patch / Info Gap | структурная | случайный контигуальный фрагмент последовательности заменяется шумом / затирается |

Для дискретных входов (текст) атаки идут не по токенам, а по эмбеддингам: `EmbeddingModelWrapper` оборачивает модель так, что она принимает эмбеддинг и работает с ним дальше (`forward_from_embeddings`), это и делает градиентные атаки возможными там, где напрямую дифференцировать нечего.

## Модели

Три классификатора с общим интерфейсом (`get_embeddings` → `forward_from_embeddings` → `classifier`), создаются интерфейсом `build_model(ds_cfg, train_cfg, model_type, use_defense)`:

- `UniversalMambaClassifier` — стек `MambaBlock` (обёртка над `mamba_ssm.Mamba` + LayerNorm + residual);
- `UniversalTransformerClassifier` — `nn.TransformerEncoder` с обучаемыми позиционными эмбеддингами;
- `UniversalCNNClassifier` — TextCNN-подобная свёрточная сеть с несколькими размерами фильтров.

Каждая модель умеет работать и с дискретным входом (`input_type="tokens"`, `nn.Embedding`), и с непрерывным (`input_type="continuous"`, `nn.Linear`) — это то, что позволяет применять один и тот же код на изображениях, аудио и тексте.

## Датасеты

| Датасет | Тип входа | Тип данных |
|---|---|---|
| `mnist` | continuous | изображения |
| `smnist` | continuous | изображения как long-range последовательность пикселей (T=784) |
| `cifar10` | continuous | изображения |
| `imdb` | tokens | текст (бинарная классификация) |
| `ag_news` | tokens | текст (4 класса) |
| `speech_commands` | continuous | аудио (сверхдлинные последовательности) |

Набор атак для тестирования подбирается под тип входа (для текста и аудио добавляются Sliding Patch / Info Gap, для изображений — CW/AutoAttack).

## Режимы защиты

Каждая пара «архитектура × датасет» обучается в четырёх режимах, дающих 2×2-сетку «входная защита × защита обучением»:

| Режим | `use_defense` (gating + inference noise + Δ-clip) | Adversarial training (SA-AT + Attack-Mix) |
|---|---|---|
| R1 — Baseline | — | — |
| R2 — Input Defense | ✔ | — |
| R3 — Training Defense | — | ✔ |
| R4 — Full Defense | ✔ | ✔ |

Чекпоинты сохраняются как `{arch}_{dataset}_{regime}.pt` (например, `mamba_mnist_full.pt`) и содержат веса, историю обучения и `training_summary` с итоговыми метриками.

## Метрики (`Evaluator`-функции)

- **Clean Accuracy / F1** — `evaluate()` на чистых данных;
- **Adversarial Accuracy** — `evaluate_adversarial()` под конкретной атакой;
- **Robustness Gap** = Clean Acc − Adv Acc;
- **State Drift** (только Mamba) — `compute_state_drift()`: средняя L2-норма разницы между hidden state на чистом и атакованном входе; показывает, насколько сильно возмущение «раскачивает» рекуррентное состояние.

## Результаты

На всех рассмотренных датасета Full Defense обеспечивает выполнение критерия сохранения качества и поднимает adversarial accuracy Mamba до 90%+ практически без потери clean accuracy и кратно снижает State Drift на порядок — сильнее всего именно на сверхдлинных последовательностях (аудио, длинные тексты).


| Датасет | Атака | AdvAcc, Baseline | AdvAcc, Full Defense | State Drift, Baseline → Full |
|---|---|---|---|---|
| MNIST | PGD | 3.32% | 97.10% | 5.23 → 0.69 |
| Speech Commands | PGD | 0.00% | 72.32% | 19.68 → 1.12 |
| AG News | PGD | — | — | 9.99 → 0.25 |

Системное ограничение: структурные атаки (Sliding Patch, Info Gap) на сверхдлинных аудиопоследовательностях защита нейтрализует хуже — входное сглаживание конфликтует с сохранением локальной информации. Полные таблицы по всем 6 датасетам, 3 архитектурам, 4 режимам и всем атакам — в тексте работы `docs/Исследовательская работа.pdf`. Веса обученных моделей для каждого из режимов — в папке [`checkpoints`](./checkpoints)

## Технологии 

Python 3.12, PyTorch, `mamba-ssm`, Hugging Face `transformers`/`datasets`, `torchvision`, `torchaudio`, `torchattacks`, `auto-attack`, scikit-learn, pandas, matplotlib/seaborn. Эксперименты проводились на NVIDIA Tesla T4 (Kaggle).

## Запуск

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
pip install -r requirements.txt
jupyter notebook Mamba_adversarial_robustness.ipynb
```

Основные зависимости, требующие отдельной установки (см. первую ячейку ноутбука):

```bash
pip install git+https://github.com/fra31/auto-attack.git
pip install mamba-ssm --no-build-isolation
pip install torchvision tqdm datasets transformers scikit-learn torchattacks
```
> `mamba-ssm` требует CUDA-совместимый GPU для установки selective-scan ядер. Эксперименты проводились на NVIDIA Tesla T4 (Kaggle).

## Структура репозитория

```
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore                         # *.pt, data/, .ipynb_checkpoints/
├── Mamba_adversarial_robustness.ipynb    # весь код: фреймворк + эксперименты
├── docs/
    └── Исследовательская работа.pdf  # полный текст работы
└── checkpoints/
    └── ...     # веса обученных моделей
```

## Документация

- [`docs/Исследовательская работа.pdf`](./docs/Исследовательская работа.pdf) — полный текст работы: обзор архитектур последовательностей и SSM, анализ уязвимостей и атак, вывод и обоснование метода SA-AT, описание архитектуры фреймворка, полные результаты экспериментов.

## Дальнейшие направления

- Перенос метода на масштабные SSM-архитектуры (Mamba-2, Jamba).
- Устойчивость к адаптивным атакам с доступом к механизму защиты.
- Более тонкие стратегии входной предобработки для сверхдлинных последовательностей.

## Лицензия

Учебная работа. Код распространяется под лицензией MIT (см. [LICENSE](./LICENSE)), если не указано иное.
