# Fashion-MNIST MLP in C++

A neural-network training and inference library built from scratch in **C++17**,
with no ML framework dependencies. It implements dense layers, backpropagation,
ReLU, dropout, AdamW, stable cross-entropy, and a reproducible evaluation pipeline.

The default **784 → 128 → 64 → 10** model achieved **88.33% held-out test accuracy**
on Fashion-MNIST, versus **84.57%** for a linear baseline under the same training
protocol. These are measured, single-seed results; see the evidence and limitations
below.

## Results

| Model | Parameters | Best epoch | Validation accuracy | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear softmax classifier | 7,850 | 18 | 85.35% | 84.57% | 0.8460 |
| ReLU MLP, hidden layers 128/64, dropout 0.1 | 109,386 | 12 | 88.93% | **88.33%** | **0.8828** |

The MLP improved test accuracy by **3.76 percentage points** over the linear
baseline. Both used seed 42, the same fixed stratified 54,000/6,000 training and
validation split, batch size 64, AdamW with learning rate 0.001 and weight decay
0.0001, gradient clipping at 5, a maximum of 20 epochs, and patience 5. The MLP
stopped after epoch 17 and restored the epoch-12 checkpoint. Checkpoints were
selected by validation loss; the 10,000 test samples were evaluated afterward.

Training including CSV loading/checkpoint writes took about 16 seconds for the
linear model and 157 seconds for the MLP on an Intel Core i5-1240P with GCC 15.2.0
in Release mode. These are single-run, single-threaded wall times on a shared
machine, not a controlled performance benchmark.

[Recorded configurations, metrics, fingerprints and learning histories](benchmarks/reference/)
provide the evidence. The independent Python evaluator matched the C++ accuracy,
macro F1 and confusion matrix over all 10,000 predictions. This comparison is
against the new linear baseline, not a measured before/after comparison with the
original coursework model. One seed does not establish statistical significance.

## Build and verify

Requires a C++17 compiler and CMake 3.16+. Python 3.9+ is used for dataset preparation,
benchmark orchestration, and integration tests; its standard library is sufficient.

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel 2
ctest --test-dir build/release --output-on-failure
```

Tests need no dataset download. They cover numerical gradients (with and without
dropout), AdamW updates and batch averaging, nonlinear learning, checkpoint round
trips, CSV validation, fixed disjoint splits, partial batches, early stopping,
deterministic training, CLI failures, and dataset conversion. Release and sanitized
Debug builds are also configured in [GitHub Actions](.github/workflows/ci.yml).

```bash
cmake -S . -B build/sanitized -DCMAKE_BUILD_TYPE=Debug -DMLP_ENABLE_SANITIZERS=ON
cmake --build build/sanitized --parallel 2
ctest --test-dir build/sanitized --output-on-failure
```

## Prepare data

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) contains 28×28
clothing images in ten classes. The original `data/` training CSV in this repository
is tracked with Git LFS and may be only a pointer in a fresh checkout. The preparation
script downloads the official archives, verifies their published checksums, checks
IDX structure, and writes CSVs plus a SHA-256 manifest into `data/prepared/`:

```bash
python3 scripts/prepare_data.py
```

Downloads are cached in `data/raw/`. Existing legacy CSV files are preserved.
The loader expects headerless rows of 784 integer pixels in `[0,255]`, and a separate
file with one integer label in `[0,9]` per row. It normalizes pixels by **255**.

Classes: 0 T-shirt/top, 1 trouser, 2 pullover, 3 dress, 4 coat, 5 sandal, 6 shirt,
7 sneaker, 8 bag, 9 ankle boot.

## Train, evaluate, predict

```bash
./build/release/mlp train --data-dir data/prepared \
  --output runs/mlp --hidden 128,64 --dropout 0.1 --epochs 20 --seed 42

./build/release/mlp evaluate --model runs/mlp/best.mlp \
  --vectors data/prepared/fashion_mnist_test_vectors.csv \
  --labels data/prepared/fashion_mnist_test_labels.csv \
  --output runs/mlp-test

./build/release/mlp predict --model runs/mlp/best.mlp \
  --vectors data/prepared/fashion_mnist_test_vectors.csv \
  --output runs/predictions.csv

python3 evaluator/evaluate.py runs/predictions.csv \
  data/prepared/fashion_mnist_test_labels.csv
```

Run `./build/release/mlp --help` for all options. `./run.sh --output runs/my-run`
builds, tests, and trains with any supplied training options. Training and evaluation
refuse nonempty output directories so earlier experiments are not silently replaced.
Prediction writes the requested output file; its parent directory must exist.

Each training run produces configuration, split indices, epoch history, validation
metrics, a summary, and the best checkpoint. Evaluation produces JSON metrics and
one predicted class per CSV line. Checkpoints retain full-precision parameters and
architecture for inference; optimizer/RNG state is not saved for resuming training.

## Reproduce or extend the comparison

```bash
python3 scripts/benchmark.py --epochs 20 --seeds 42 --output runs/comparison

# Estimate variability across independent seeds (takes longer):
python3 scripts/benchmark.py --epochs 20 --seeds 42 43 44 --output runs/multi-seed
```

The runner trains both architectures with the same settings per seed, evaluates
only their selected checkpoints, and saves a small `evidence/` directory with
configurations, results, histories and source/binary/data/checkpoint fingerprints.
Changing `--hidden` in the training CLI supports other dense architectures without
editing source. Use validation results to choose hyperparameters before inspecting
test performance.

## Engineering decisions

- **Reusable library:** separate model, data and training components; instance-owned
  state and contiguous weight arrays replace fixed-size globals.
- **Correct mini-batches:** gradients accumulate before one optimizer update,
  including the final short batch; AdamW moments persist across epochs.
- **Stable learning:** zero-mean He initialization, ReLU, inverted dropout, stable
  log-sum-exp cross-entropy and optional gradient clipping.
- **Reproducible evaluation:** a fixed stratified split, seeded shuffling, early
  stopping, best-checkpoint restoration, per-class metrics and confusion matrices.
- **Defensive interfaces:** strict input validation, versioned checkpoint parsing,
  readable CLI errors, numerical tests and sanitizer checks.

[Architecture and mathematical details](docs/architecture.md) explain the gradient
flow, optimizer, artifact contracts, and limitations. This is a CPU learning-systems
project: it has no GPU/BLAS backend, convolutions, augmentation, or production
serving layer. It loads datasets into memory. Reproducibility is scoped to the same
binary/platform; standard-library random distributions can differ across compilers.

## CV description

> Developed a C++17 neural-network library from scratch with configurable dense
> architectures, backpropagation, dropout and AdamW; achieved 88.33% Fashion-MNIST
> test accuracy, 3.76 percentage points above a linear baseline, with reproducible
> experiments, numerical gradient checks and automated sanitizer tests.

Use the result with its single-seed context, preserve shared authorship, and be
prepared to explain the implementations and tradeoffs. The original coursework
implementation credited **KP & RM**; this repository develops that project into a
modular, tested implementation.
