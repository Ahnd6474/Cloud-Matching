# Stochastic image transition model

An end-to-end PyTorch training scaffold for the proposed closed-loop stochastic
image generator. The primary workflow generates a fixed dataset first and then
reuses it across all training epochs.

## Implemented pipeline

For every clean image `x0`, training data is produced on the fly:

1. Sample an input level `s` and goal-image level `r` with `s > r`.
2. Set the requested answer level to `a = max(r - 10, 0)`.
3. Select one corruption family per item and independently create the current
   image `x_s` and image goal `x_r` at their respective severities.
4. Optionally remove additional detail from the image goal with blur,
   downsample/upsample, and block masking.
5. Directly simulate multiple samples at answer severity `a` to form the target
   cloud. With the corruption mixture disabled, use the exact arbitrary-skip
   VP posterior `q(x_a | x_s, x_0)` instead.

The preparation command stores these records once in compact tensor shards.
Training does not regenerate corruption, so every epoch sees the same finite
dataset (with ordinary shuffled ordering).

## Corruption mixture

`corruption.weights` in `configs/train.yaml` controls relative sampling
probabilities. A weight of zero disables that type; weights are normalized
automatically.

- Gaussian: white, local, edge-weighted, low/high/band-pass frequency,
  channel-correlated, low-rank, anisotropic, and signal-dependent.
- Heavy-tailed/additive: Student-t and Laplace.
- Camera-like: Poisson shot noise and multiplicative speckle.
- Structural: salt-and-pepper impulse, blur, downsample/upsample, and block mask.

All implementations are batched PyTorch operations and remain on the selected
device. The core tensor/distribution primitives come from PyTorch; GeomLoss is
used for Sinkhorn divergence.

## Model

- One shared encoder for current and goal images, selectable as `cnn` or `vit`.
- Every bottleneck token participates in current-to-goal cross-attention.
- A pooled cross-attention head predicts one conditional scalar noise variance
  `lambda` per image. It is trained only through the cloud loss, without a
  variance label, and scales standard Gaussian noise by `sqrt(lambda)`.
- Full-resolution spatial Gaussian noise, rather than only one global vector.
- U-Net decoder with current-image skip connections.
- Learned spatial residual and spatial update gate.
- Condition encoding is computed once and reused for every cloud sample.

## Distribution losses

Change `loss.name` in `configs/train.yaml`:

- `paired`: same posterior noise is used by model and target; available only
  when `corruption.enabled: false` selects the analytic VP Gaussian path.
- `energy`: unpaired energy distance; current default.
- `sinkhorn`: debiased Sinkhorn divergence from GeomLoss.

All losses operate on multi-scale **correction features** rather than raw
internal latent coordinates.

Select the shared encoder in the same config:

```yaml
model:
  encoder_type: vit  # cnn or vit
  vit_depth: 4
  vit_patch_size: 8
```

The ViT path creates tokens directly with an 8x8 patch embedding, applies
self-attention, and learns a feature pyramid for the residual decoder. It is
still one shared encoder instance, not separate current and goal encoders.

## Simulator cloud versus exact VP posterior

The default mixed path is simulator-based: it samples an empirical answer cloud
at severity `a`, so it supports Gaussian, non-Gaussian, signal-dependent, and
structural degradations through one interface. This cloud is unpaired and must
use `energy` or `sinkhorn` loss.

For controlled Gaussian experiments, set `corruption.enabled: false`. The
forward process then uses VP Gaussian noise and composes the interval between
answer level `a` and input level `s` into the analytic teacher distribution
`q(x_a | x_s, x_0)`. The model architecture itself is unchanged and is not
restricted to Gaussian outputs.

## Installation

Create a local virtual environment and install the project in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## 1. Generate the fixed training dataset

Set `data.root` and `data.prepared_root` in `configs/train.yaml`, then run:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py
```

Or override both paths:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py `
  --data "D:\images" `
  --output "D:\bridge-data" `
  --variants 4 `
  --device cuda
```

The destination must be empty. It contains `manifest.json` plus numbered `.pt`
shards. Image tensors are stored as uint8 and decoded to `[-1, 1]` while
loading. `prepare.variants_per_image` controls how many independently cropped
and corrupted records are generated from each source image.

## 2. Inspect generated training data

Synthetic smoke data:

```powershell
.\.venv\Scripts\python.exe preview_data.py --synthetic
```

Prepared fixed dataset:

```powershell
.\.venv\Scripts\python.exe preview_data.py --prepared "D:\bridge-data"
```

The preview rows are `clean | current | goal | target answer`.

## 3. Train repeatedly on the fixed dataset

The default config reads `data/prepared_root` and never touches the source
images during training:

```powershell
.\.venv\Scripts\python.exe train.py --prepared "D:\bridge-data"
```

For development only, set `data.prepared_root: ""` to retain the previous
on-the-fly corruption path.

### Performance settings

The fixed-data loader uses shard-aware shuffled batches, memory mapping,
multi-worker prefetch, pinned CUDA memory, and non-blocking transfers. The
training loop also supports channels-last CNNs, AMP, TF32, fused AdamW, and an
optional compiled forward pass.

```yaml
data:
  workers: 4
  prefetch_factor: 2
  prepared_cache_shards: 2

train:
  amp: true
  amp_dtype: float16       # or bfloat16
  channels_last: true
  tf32: true
  fused_optimizer: true
  compile: false           # enable after an eager run succeeds
  compile_mode: reduce-overhead
```

`torch.compile` has a one-time warm-up cost and may not help small datasets.
For long CUDA runs, set `compile: true` and benchmark both modes on the target
GPU. Increase `data.batch_size` until GPU memory is nearly utilized.

Environment smoke test:

```powershell
.\.venv\Scripts\python.exe train.py --smoke --device cpu
```

Resume:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/train.yaml `
  --resume runs/default/checkpoints/latest.pt
```

Outputs include:

- `previews/`: clean, current, goal, target, and predicted rows
- `checkpoints/latest.pt`: complete model/optimizer/scaler/config state
- `tensorboard/`: loss, gradient norm, and sampled corruption levels
- `config.json`: resolved experiment configuration

TensorBoard:

```powershell
.\.venv\Scripts\tensorboard.exe --logdir runs
```

## Kaggle DIV2K Gaussian-only experiment

Open [`notebooks/div2k_gaussian_only_kaggle.ipynb`](notebooks/div2k_gaussian_only_kaggle.ipynb)
in Kaggle, select the **T4 x2** accelerator, enable Internet, and run all cells.
The notebook downloads DIV2K with `kagglehub`, prepares a fixed Gaussian-only
dataset, and launches `torchrun` with one process per visible GPU. Its output
includes checkpoints, training curves, corruption/severity comparisons, cloud
distribution plots, uncertainty maps, and a 40-step fixed-goal rollout with
per-step image and hidden-feature similarity metrics.

The full experiment prepares 16 transitions per DIV2K training image (12,800
fixed records) and trains for 60 epochs, or roughly 24,000 optimizer updates at
global batch size 32. It uses versioned output and cache directories so the old
20-epoch cosine-scheduler checkpoint is not resumed accidentally.

The standalone launcher and full-size experiment settings are:

- `kaggle/train_div2k_ddp.py`
- `configs/kaggle_div2k_gaussian.yaml`

Because the notebook clones this repository, commit and push these files before
starting a Kaggle run (or attach an updated repository snapshot as a Kaggle
dataset).

## Sample from a checkpoint

```powershell
.\.venv\Scripts\python.exe sample.py `
  --checkpoint runs/default/checkpoints/latest.pt `
  --current current.png `
  --goal goal.png `
  --samples 8 `
  --output samples.png
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Images are normalized to `[-1, 1]` and their height and width must be divisible
by 8. The default loader crops/resizes every image to a square resolution.
