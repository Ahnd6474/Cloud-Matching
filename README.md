# Stochastic image transition model

An end-to-end PyTorch training scaffold for the proposed closed-loop stochastic
image generator. The primary workflow generates a fixed dataset first and then
reuses it across all training epochs.

## Cloud Matching algorithm

In this repository, **Cloud Matching** means learning a conditional transition
distribution by matching finite sample clouds. Given a current state and a
goal, the model does not predict one mandatory answer or a pixelwise Gaussian
parameter for every output. It draws several possible next images and trains
their empirical distribution against several samples from the teacher
transition distribution.

The intended separation is:

- `current` describes where the trajectory is now;
- `goal` supplies the semantic direction and any appearance that is specified;
- the latent field represents appearance left unspecified by the condition;
- the output cloud represents plausible next states, not only their average.

### 1. Construct a transition problem

Start from a clean training image `x_0`. Sample a current corruption level `s`
and a less-corrupted image-goal level `r`, with `s > r`, then choose the answer
level

```math
a = \max(r-J,0),
```

where `J=10` by default. The endpoint curriculum may override this and set
`a=0`, explicitly turning the sample into an arbitrary `x_s -> x_0` recovery
problem. The current and image goal are sampled independently from the same
underlying image:

```math
x_s \sim q(x_s\mid x_0), \qquad g_r \sim q(g_r\mid x_0).
```

For the VP Gaussian path,

```math
x_t=\sqrt{\bar\alpha_t}\,x_0+
    \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad
\epsilon\sim\mathcal N(0,I).
```

`g_r` is an input condition; the clean image remains available only to the
teacher that constructs the training target. The model itself sees `x_s` and
`g_r`, never `x_0`.

### 2. Build the teacher cloud

Instead of storing one target image, sample `N` answers from the exact
arbitrary-skip posterior:

```math
Y=\{y_j\}_{j=1}^{N}, \qquad
y_j\sim q(x_a\mid x_s,x_0).
```

Let

```math
\alpha_{s\mid a}=\frac{\bar\alpha_s}{\bar\alpha_a}.
```

The complete noise interval from `a` to `s` can then be combined without
simulating every intermediate step:

```math
q(x_a\mid x_s,x_0)=\mathcal N(\mu_{a\mid s},\sigma^2_{a\mid s}I),
```

```math
\mu_{a\mid s}=
\frac{\sqrt{\bar\alpha_a}(1-\alpha_{s\mid a})}{1-\bar\alpha_s}x_0+
\frac{\sqrt{\alpha_{s\mid a}}(1-\bar\alpha_a)}{1-\bar\alpha_s}x_s,
```

```math
\sigma^2_{a\mid s}=
\frac{(1-\bar\alpha_a)(1-\alpha_{s\mid a})}{1-\bar\alpha_s}.
```

When `a=0`, $\bar\alpha_a=1$, so the variance becomes zero and the entire
teacher cloud collapses exactly to the clean point mass `{x_0}`. This property
is what makes non-Gaussian structured corruption safe in the current paired
experiment: texture or edges may be removed from `current` and `goal`, but the
answer distribution at level zero is still known exactly. The code does not
invent an unsupported non-Gaussian intermediate posterior.

### 3. Generate the predicted cloud

The transition model encodes the condition once and draws `M` spatial latent
fields:

```math
\hat y_i=F_\theta(x_s,g_r,z_i), \qquad
z_i=\sqrt{\lambda_\theta(x_s,g_r)}\,\epsilon_i,
\quad \epsilon_i\sim\mathcal N(0,I).
```

This produces

```math
\hat Y=\{\hat y_i\}_{i=1}^{M}.
```

The scalar `lambda` is not given a target. It is learned through the cloud
objective: the model must reduce it when the answer is effectively
deterministic and retain it when the teacher cloud has meaningful spread. A
nonlinear spatial decoder can turn a simple Gaussian base field into a
condition-dependent, non-Gaussian image distribution.

### 4. Match corrections rather than an internal latent

Conceptually the compared objects are transition corrections from the same
current image:

```math
\hat C_i=\phi(\hat y_i)-\phi(x_s), \qquad
C_j=\phi(y_j)-\phi(x_s),
```

where `phi` is either the full-band image decomposition or a fixed multi-scale
feature map. This makes the objective describe **how the state should change**,
rather than forcing two unrelated encoder coordinates to coincide.

The current DIV2K experiment uses paired full-band matching. The same base
noise `epsilon_i` is supplied to the analytic teacher sample and the model
sample, giving a low-variance coupling:

```math
\mathcal L_{paired}=\frac{1}{M}\sum_i
d_{full-band}(\hat y_i,y_i).
```

The full-band distance forms an undecimated à-trous pyramid. If `e` is the
paired image error, each level separates

```math
b_k=e_k-B_{2^k}(e_k), \qquad e_{k+1}=B_{2^k}(e_k),
```

and applies a Charbonnier penalty to every high-frequency band plus the final
low-frequency residual. Because there is no spatial subsampling, checkerboard,
edge, and one-pixel texture errors cannot vanish through average pooling.

Two genuinely unordered cloud distances are also implemented:

- **Energy distance** compares predicted-to-target distances and subtracts
  both within-cloud distances;
- **Sinkhorn divergence** performs entropy-regularized optimal transport
  between the two empirical correction clouds.

These are required when samples have no valid one-to-one noise coupling. The
paired loss is computationally cheaper and supplies a stronger learning signal,
but it matches a chosen coupling as well as the marginal distribution. Thus the
current paired experiment is more constrained than pure set-level matching.

### 5. Why this is not ordinary denoising or standard diffusion

A deterministic denoiser trained with pixel MSE learns a conditional mean. If
several textures are plausible, incompatible high-frequency possibilities are
averaged and the result becomes smooth. Cloud Matching instead supervises
multiple samples, so mean, spread, and sample-dependent high-frequency
corrections can all affect the objective.

Unlike a standard diffusion sampler, the model does not predict a score or
noise value for one infinitesimal timestep and then require the complete
reverse schedule. It learns an image-to-image transition distribution that may
skip directly from arbitrary level `s` to answer level `a`, including `a=0`.
The current code also does not pass an explicit timestep to the network; the
transition size is inferred from `current` and `goal`.

This does not remove the need for iteration. One-step training defines a
learned operator

```math
x_{k+1}\sim F_\theta(x_k,g_0,z_k).
```

At inference, it can be reapplied with a fixed goal and either persistent or
resampled latent fields. Rollout quality is therefore evaluated separately:
good one-step cloud matching does not guarantee that many off-training-manifold
updates remain stable.

### 6. Training procedure

```text
for each clean image x0:
    sample s > r and choose answer level a
    sample current xs and image goal gr
    if a == 0 and structured endpoint is selected:
        remove selected texture/edges or apply a spatial corruption
        teacher cloud Y = {x0, ..., x0}
    else:
        sample Y from the exact VP posterior q(xa | xs, x0)

    draw spatial base noise ε1 ... εM
    predicted cloud Ŷ = Fθ(xs, gr, ε1 ... εM)
    update θ by paired full-band, energy, or Sinkhorn cloud loss
```

The preparation command stores these transition records once in compact tensor
shards. Training then reuses the fixed dataset with shuffled ordering, making
experiments directly comparable while limiting diversity to the generated
records.

## Corruption mixture

`corruption.weights` in `configs/train.yaml` controls relative sampling
probabilities. A weight of zero disables that type; weights are normalized
automatically.

- Gaussian: white, local, edge-weighted, low/high/band-pass frequency,
  channel-correlated, low-rank, anisotropic, and signal-dependent.
- Heavy-tailed/additive: Student-t and Laplace.
- Camera-like: Poisson shot noise and multiplicative speckle.
- Structural: salt-and-pepper impulse, blur, downsample/upsample, block mask,
  texture-only suppression, and edge/line erasure.

All implementations are batched PyTorch operations and remain on the selected
device. The core tensor/distribution primitives come from PyTorch; GeomLoss is
used for Sinkhorn divergence.

## Architecture

The model is a conditional stochastic image-transition operator rather than a
direct clean-image regressor:

```text
current x_t ─┐
             ├─ shared encoder ─ current bottleneck ─┐
goal g_t ────┘                                       ├─ cross-attention ─ h_t
                              goal bottleneck ───────┘          │
                                                               ├─ pooled head → scalar λ
ε₁ ... ε_M ~ N(0,I) ───────────────────────────────────────────┤
                                                               ▼
current full-resolution detail ─────────────────────── decoder(h_t, √λ ε_m)
                                                               │
                                             residual Δ_m and update gate u_m
                                                               │
                                      y_m = clamp(x_t + u_m ⊙ Δ_m, -1, 1)
```

One condition produces `M` output images, so the returned tensor is an
empirical conditional cloud `[B, M, C, H, W]`. The model does not evaluate an
explicit likelihood. Its learned distribution is the push-forward of spatial
Gaussian latent fields through the conditional decoder.

### 1. Shared current/goal encoder

The same encoder instance processes both images. In the implementation they
are concatenated on the batch axis for one accelerator-efficient call and are
split afterward; there are not separate current and goal encoder weights.

The default CNN encoder uses residual two-convolution blocks with GroupNorm and
SiLU. Three stride-2 convolutions produce a four-level feature pyramid. With
the Kaggle configuration (`base_channels=32`, 128x128 input), its shapes are:

| level | shape | later use |
|---|---:|---|
| full | `[B, 32, 128, 128]` | current high-resolution detail for the decoder |
| half | `[B, 64, 64, 64]` | convolutional-decoder skip only |
| quarter | `[B, 128, 32, 32]` | convolutional-decoder skip only |
| bottleneck | `[B, 256, 16, 16]` | current/goal cross-attention |

The optional ViT encoder uses an 8x8 patch embedding, 2-D sinusoidal positions,
and Transformer encoder blocks. Its 16x16 token grid is projected back into the
same four pyramid shapes so either decoder can consume it. The configured CNN
implicit model has about 3.24M trainable parameters; the corresponding ViT
variant has about 5.96M.

An important current design boundary is that **only the goal bottleneck enters
the conditioning path**. Full-resolution and intermediate decoder detail comes
from the current image. This prevents an easy pixel-copy shortcut, but it also
means fine goal texture is compressed to the 16x16 bottleneck. Multi-scale goal
fusion is therefore a future architectural extension, not something the
current model already performs.

### 2. Bottleneck cross-attention

Both bottlenecks are flattened to 256 spatial tokens and receive the same 2-D
sinusoidal positions. In every cross-attention block:

- queries come from current tokens;
- keys and values come from goal tokens;
- attention and a four-times-wider GELU feed-forward network both use residual
  connections.

The default two blocks produce the fused condition
`h_t ∈ R[B,256,16,16]`. Every current bottleneck token can attend to every goal
bottleneck token; the condition is not compressed to one vector.

The model currently receives no explicit timestep or corruption-family token.
It infers transition severity from `current` and `goal` themselves. The sampled
levels are used to construct supervision, not passed to `forward()`.

### 3. Self-learned scalar latent variance

Global average pooling over `h_t` feeds a small MLP that predicts one variance
per input image:

```math
\lambda = \lambda_{min} + (\lambda_{max}-\lambda_{min})
           \sigma(\mathrm{MLP}(\mathrm{mean}_{xy}(h_t))).
```

For every cloud member, the model samples a full-resolution three-channel
spatial field and rescales it as

```math
\epsilon_m \sim \mathcal N(0,I), \qquad e_m=\sqrt{\lambda}\,\epsilon_m.
```

`lambda` has no direct label or variance loss. Gradients reach it only through
the reparameterized samples and cloud objective. It is a single global scalar,
not yet a spatial or multi-scale uncertainty map. The deterministic condition
is encoded once, then reused for all `M` samples.

### 4. Decoder choices

`decoder_type: implicit` is used by the current DIV2K experiments. At every
output coordinate, one shared LIIF-style MLP receives:

- bilinearly sampled fused condition `h_t`;
- full-resolution current-encoder features;
- the current RGB pixel;
- the scaled spatial latent RGB value;
- normalized `(x,y)` coordinates and Fourier features.

With six Fourier bands the coordinate representation has 26 values. The
default point query therefore has `256 + 32 + 3 + 3 + 26 = 320` inputs, passes
through a depth-3 MLP of width 128, and returns six values: three residual
logits and three gate logits. Queries are chunked only to control memory; the
same MLP weights are used at every pixel. This decoder has no learned spatial
upsampling or transposed convolution, which removes fixed pixel-phase kernels
as a source of repeated grid artifacts.

The legacy `conv` decoder instead injects projected latent noise at the
bottleneck and all three upsampling scales, using current-image pyramid skips
and transposed-convolution decoder stages. It remains available for ablations.

### Full-resolution CvT cloud transformer

The main full-resolution path is now the CvT-style configuration selected by
`fullres_encoder_type: cvt` and `fullres_attention_type: cvt`. The outer
architecture key remains `fullres_axial` for checkpoint compatibility, but the
current COCO and U-stat experiments use CvT attention rather than the legacy
axial/local mixer. This is a project-specific CvT variant, not a verbatim copy
of the staged architecture from the CvT paper.

The important design choice is to retain one query for every image position
while convolutionally compressing only the key/value context:

1. A shared 3x3 convolutional stem embeds the current image, followed by a
   gated depthwise 3x3 and pointwise local refinement. There is no patch
   embedding or spatial bottleneck in the residual stream.
2. Full-resolution `Q[B,H*W,D]` preserves the position at which a correction
   will be written.
3. Learned depthwise 3x3, 5x5, and 7x7 projections reduce the context to 8x8,
   4x4, and 2x2 grids. Their concatenation supplies only 84 multiscale K/V
   tokens to learned multi-head attention.
4. Every CvT attention block is followed by its own pre-normalized GELU FFN.
   Optional random attention is inserted before the configured refinement
   block, so subsequent CvT+FFN blocks can spatially regularize the sampled
   content.
5. A normalized linear RGB head writes a bounded residual directly on the
   native grid. The decoder contains no transposed convolution or learned
   upsampling phase.

For a 128x128 image, dense spatial attention would form 16,384x16,384 scores
per head. The CvT path forms 16,384x84 scores instead: about 195x fewer
attention-score elements. This reduction does not include the Q projection or
FFN cost, but it is what makes global spatial context practical without
discarding pixel-aligned queries. The convolutional projections also give the
model a useful local/multiscale image bias that a bare per-pixel linear
embedding lacks. Current experiments favor this path for optimization and
reconstruction quality; that observation is not yet a controlled architecture
ablation.

Image goals use the same shared CvT encoder and CvT cross attention. Text goals
use a jointly trained caption Transformer and full image-query/text-KV cross
attention because at most 48 text tokens are cheap enough that K/V pooling is
unnecessary. After conditioning, both routes use the same CvT refinement and
cloud decoder.

The legacy names `pooled_attention` and `pooled_multiscale` are still accepted
as aliases of `cvt`, so existing YAML files and checkpoint state dictionaries
remain loadable. `linear`, `cnn`, and `factorized` remain available only for
ablations and older checkpoints.

The fused grid predicts a positive noise-energy map `w[B,H,W]`, and each cloud
member uses

```math
e_{m,i}=\sqrt{w_i/D}\,\epsilon_{m,i},\qquad
\epsilon_{m,i}\sim\mathcal N(0,I_D).
```

When `fullres_random_attention` is enabled, one Gaussian random memory is
injected through attention before the configured refinement block; there is no
second random-memory layer. Arbitrary spatial sizes are supported. Convolution
is used only to encode/project spatial context; the output path has no
transposed convolution, LIIF query, or patch unprojection.

The unnormalized `w` is retained for sampling. Its normalized form
`p_i=w_i/sum(w)` is optionally trained with `loss.spatial_ce_weight` against the
high-frequency correction-energy distribution derived from the target cloud.
This CE term teaches only *where* to sample: multiplying all `w` values by a
constant leaves it unchanged. Absolute strength and decoded appearance remain
self-supervised by Energy distance. The image-conditioned CvT width-320,
depth-12 configuration has 12.27M parameters.

### 5. Residual update and recurrence

The pyramid implicit decoder ends with a bounded residual and a learned
per-pixel/channel gate:

```math
\Delta_m = \Delta_{max}\tanh(r_m), \qquad
u_m=\sigma(q_m), \qquad
y_m=\mathrm{clip}(x_t+u_m\odot\Delta_m,-1,1).
```

The full-resolution CvT core instead predicts its bounded RGB residual
directly: `y_m = clip(x_t + max_residual * tanh(r_m), -1, 1)`. Its learned
cross-attention and random-attention gates control feature updates upstream;
there is no separate pixel output gate.

The network is trained as a one-step transition operator. A rollout is not an
internal recurrent layer: inference explicitly feeds one output back as the
next `current` while keeping `goal_0` and, when requested, its trajectory latent
fixed. The pyramid path requires height and width divisible by 8. The
full-resolution CvT path accepts arbitrary positive spatial sizes because its
query grid is never patchified or downsampled.

## Distribution losses

Change `loss.name` in `configs/train.yaml`:

- `paired`: same posterior noise is used by model and target; available only
  when `corruption.enabled: false` selects the analytic VP Gaussian path. It
  also supports structured damage on clean-endpoint examples because their
  target cloud is the exact point mass at `x_0`.
- `paired_full_band`: the same analytic noise pairing, evaluated over a
  complete stride-free a-trous frequency pyramid. Unlike pooled features, it
  retains full-resolution checkerboard, edge, and texture errors.
- `energy`: unpaired energy distance; current default.
- `energy_full_band`: unpaired energy distance over a Laplacian pyramid that
  retains pixel-scale edges and texture instead of only pooled low frequencies.
- `sinkhorn`: debiased Sinkhorn divergence from GeomLoss.

All losses operate on multi-scale **correction features** rather than raw
internal latent coordinates.

Select the shared encoder in the same config:

```yaml
model:
  architecture: pyramid  # pyramid or fullres_axial
  encoder_type: vit  # cnn or vit
  decoder_type: implicit  # conv or implicit
  vit_depth: 4
  vit_patch_size: 8
  implicit_hidden_dim: 128
  implicit_depth: 3
```

The bottleneck-free experiment instead uses:

```yaml
model:
  architecture: fullres_axial
  heads: 8
  fullres_dim: 320
  fullres_depth: 12
  fullres_cross_depth: 2
  fullres_ffn_ratio: 2.0
  fullres_encoder_type: cvt
  fullres_attention_type: cvt
  fullres_pooled_kernel_sizes: [3, 5, 7]
  fullres_pooled_output_sizes: [8, 4, 2]
  fullres_gradient_checkpointing: true
  noise_variance_init: 0.1
  noise_energy_parameterization: softplus_amplitude
  noise_amplitude_safety_max: 8.0
loss:
  name: energy_full_band
  samples: 4
  spatial_ce_weight: 0.05
```

With `softplus_amplitude`, the full-resolution model predicts a positive noise
amplitude `a` and injects `a * epsilon / sqrt(D)`. The reported spatial energy
is `w = noise_variance_min + a^2`, so it is learned without the sigmoid ceiling
used by legacy bounded checkpoints. `noise_amplitude_safety_max` is only a
distant mixed-precision guard; the default value corresponds to total energy
64 rather than a normal training target.

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

## Clean-endpoint and structured-detail curriculum

`configs/kaggle_div2k_structured_endpoint.yaml` is the recommended follow-up
to the Gaussian reconstruction experiment. It keeps ordinary intermediate
transitions analytic, while forcing roughly 60% of sampled transitions to
answer at `x_0`. A configurable fraction of those endpoint examples replace
the Gaussian input with one of the following degradations:

- `texture_suppress`: detects local high-frequency energy away from strong
  boundaries and selectively smooths it.
- `edge_erase`: detects and dilates lines/boundaries, then replaces only that
  band with a local low-pass estimate.
- Local, edge-weighted, and high-frequency Gaussian noise, blur, downsample,
  and masks remain in the mixture for coverage.

Because every structured example targets clean `x_0`, it remains compatible
with `paired_full_band`; no fictitious non-Gaussian intermediate posterior is
introduced.

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

Multiple source datasets can be combined without copying their HR images into
one directory. Repeat `--data` once per root; for example, DF2K uses 800 DIV2K
training images plus 2,650 Flickr2K images:

```bash
python prepare_dataset.py \
  --data /workspace/data/div2k/Dataset/DIV2K_train_HR \
  --data /workspace/data/flickr2k/Flickr2K/Flickr2K_HR \
  --output /workspace/data/prepared_df2k \
  --device cuda
```

The same roots can be declared in YAML with `data.roots`. When it is non-empty,
it takes precedence over the backward-compatible scalar `data.root`.

The destination must be empty. It contains `manifest.json` plus numbered `.pt`
shards. Image tensors are stored as uint8 and decoded to `[-1, 1]` while
loading. Format v2 stores one uint32-compatible seed per paired record instead
of its full FP16 Gaussian tensor; the exact cloud noise is reconstructed in a
stateless batch operation on the training GPU. The loader remains compatible
with format-v1 datasets that stored full noise tensors. `prepare.variants_per_image`
controls how many independently cropped and corrupted records are generated
from each source image.

### Frozen lightweight feature encoder

`EfficientNetB0Features` exposes ImageNet-pretrained EfficientNet-B0 stages for
perceptual/content/texture losses. It consumes the bridge's `[-1, 1]` tensors
without resizing and, for a 128x128 input, returns feature maps at 64, 32, 16,
8, and 4 pixels. The classification head is omitted, all parameters are frozen,
and BatchNorm remains in evaluation mode even when a parent loss is trained.

```python
from stochastic_bridge import EfficientNetB0Features

encoder = EfficientNetB0Features(pretrained=True).cuda()
features = encoder(images_128)
```

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

## COCO caption-conditioned experiment

The text path does not depend on CLIP. It builds a deterministic 16K word
vocabulary from COCO Captions and jointly trains token embeddings, learned
positions, a four-layer Transformer text encoder, full image-to-text cross
attention, and the cloud-matching generator. The image generator uses the
full-resolution CvT core described above: convolutional multiscale K/V,
pixel-aligned Q, optional random memory, CvT+FFN refinement, and a direct RGB
residual head.

Download and prepare the official COCO 2017 train/validation images and human
captions. The source JPEG files stay at native resolution; the dataset loader
fits the whole scene into 128x128 with neutral padding at training time so a
captioned object is not silently removed by an aggressive random crop.

```bash
python prepare_coco.py --root data/coco2017
```

Start on one GPU:

```bash
python train_coco_text_ddp.py \
  --config configs/coco_text_fullres.yaml \
  --coco-root data/coco2017
```

Start on two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train_coco_text_ddp.py \
  --config configs/coco_text_fullres.yaml \
  --coco-root data/coco2017
```

TensorBoard records aligned `clean | current | target mean | prediction mean |
prediction sample` images, the captions used for those rows, and
`text/shuffled_caption_output_l1`. The last metric runs the same current/noise
with captions permuted across the batch; a value that stays at zero indicates
that the generator is ignoring its text condition.

## Kaggle DIV2K structured-endpoint experiment

Open [`notebooks/div2k_gaussian_only_kaggle.ipynb`](notebooks/div2k_gaussian_only_kaggle.ipynb)
in Kaggle, select the **T4 x2** accelerator, enable Internet, and run all cells.
The notebook downloads DIV2K with `kagglehub`, prepares a fixed native-128
hybrid dataset, and launches `torchrun` with one process per visible GPU. It
checks clean-endpoint and corruption-family counts before training. Its output
includes checkpoints, training curves, Gaussian-severity and structured-
corruption comparisons, cloud distribution plots, uncertainty maps, and a
40-step fixed-goal rollout with per-step image and hidden-feature metrics.

The full experiment prepares 16 transitions per DIV2K training image (12,800
fixed records) and trains for 60 epochs, or roughly 48,000 optimizer updates at
global batch size 16. It uses versioned output and cache directories so an old
Gaussian checkpoint is not resumed accidentally. Set the notebook's optional
`INIT_CHECKPOINT` path to warm-start its model weights without restoring the
old optimizer or scheduler.

The standalone launcher and full-size experiment settings are:

- `kaggle/train_div2k_ddp.py`
- `configs/kaggle_div2k_structured_endpoint.yaml`

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
