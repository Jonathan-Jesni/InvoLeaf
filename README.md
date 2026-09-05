# InvoLeaf

Involution inside the Vision Transformer encoder block, for plant leaf disease
classification.

**Claim.** Comparable accuracy at lower parameter and compute cost, with better
robustness to realistic field conditions. Not a clean-accuracy win — PlantVillage is
lab-captured and saturated, with every published method sitting between 94% and 98.5%.

---

## The gap this fills

Two papers leave complementary holes.

**Involution** (Li et al., CVPR 2021) introduces a spatial-specific, channel-agnostic,
dynamically-generated operator. Because its kernels are shared across channels, cost
grows linearly with channel count, so a 7×7 receptive field is affordable where 7×7
convolution is not. It is never tested inside a transformer, never on fine-grained data,
and never under corruption.

**AgriTL-ViT** (Expert Systems with Applications, 2025) replaces the ViT MLP with a
module presented as extracting *"hierarchical spatial features"*. That module is six 1×1
convolutions and two 3×1 max-pools. A 1×1 convolution is a per-token linear map — it
cannot mix spatial information. The only spatial interaction comes from the max-pools,
which are non-parametric, lossy, and operate on the *flattened raster sequence*, pooling
token 13 (end of row 0) with token 14 (start of row 1). So roughly 3.5M parameters buy
no learnable spatial mixing. Their Table 17 then shows accuracy collapsing to
74.9% / 61.4% / 80.3% under combined field-style corruption.

There is a slot in the block where spatial mixing was promised but not delivered.
Involution is the operator shaped to fill it.

## Architecture

A timm ViT block is `x = x + attn(norm1(x))` then `x = x + mlp(norm2(x))`. The MLP runs
per token, so **all** spatial mixing in a ViT happens inside attention. We replace the
MLP slot.

Involution is channel-*agnostic*: it mixes space but not channels. Since the FFN is the
block's main channel-mixing stage, a naive swap deletes channel mixing — so the primary
variant keeps a narrowed FFN alongside.

| Variant | Block | Params | Role |
|---|---|---|---|
| `pure` | `x + Inv(norm2(x))` | 36.4M | Faithful swap. Expected to underperform; the ablation showing why channel mixing matters. |
| **`inv_ffn`** | `x + FFN_r2(Inv(norm2(x)))` | **64.7M** | **Primary.** Real spatial mixing plus retained channel mixing. |
| `parallel` | `x + FFN_r4(norm2(x)) + Inv(norm2(x))` | 93.0M | Capacity control: did involution help, or did more parameters help? |

Baseline ViT-B/16 is 85.8M, so the primary variant is **24.6% lighter**. The
AgriTL-ViT reproduction measures 100.0M -- it moves the other way, because a second
attention stage per block outweighs what its narrower MLP replacement saves.

### Two engineering details that make it work

**Memory-efficient involution.** The reference `nn.Unfold` formulation materialises
`B × C·K² × HW`. Measured on the RTX 5050 at C=768, K=7, batch 16, across 12 blocks:

| implementation | 1 block | 12 blocks |
|---|---|---|
| `unfold` (reference) | 0.93 GB | **6.51 GB** |
| `shift` (default) | 0.08 GB | **0.61 GB** |

6.51 GB would OOM an 8 GB card before ViT's own activations are counted. The `shift`
path accumulates over the K² kernel offsets by slicing a padded, pre-reshaped view, so
the K² tensors autograd saves all share **one** storage. Both paths are numerically
identical to machine precision (`tests/test_involution.py`), so this is an optimisation,
not a different operator. The cost is K² small kernels per block instead of one large
one — it trades wall-clock for the ability to run K=7 at all.

**Delta initialisation.** The involution block is randomly initialised inside a
*pretrained* encoder, which normally costs several rough epochs. Zeroing `span.weight`
and setting its bias to a delta kernel makes the operator an exact spatial identity at
step 0, so the pretrained weights see undisturbed activations. Gradients still reach
`span.weight`, so it lifts off the identity after one optimiser step (verified in the
test suite — if it ever regressed, the operator would train only its bias and silently
stay an identity).

## Setup

Install torch for your backend first, from https://pytorch.org — CUDA on the RTX 5050,
ROCm on the W7800. Then:

```bash
pip install -r requirements.txt
```

```bash
python scripts/prepare_data.py --dataset tomato --clone
```

## Running

```bash
python scripts/train.py --config configs/vit_b16_baseline.yaml --epochs 1 --limit-batches 20
```

```bash
python scripts/train.py --config configs/involeaf_b16.yaml --seeds 42 43 44
```

```bash
python scripts/eval_robustness.py --ckpt results/checkpoints/involeaf_b16_seed42.pth
```

```bash
python scripts/benchmark.py --all
```

```bash
python scripts/make_tables.py
```

Any config key can be overridden inline:

```bash
python scripts/train.py --config configs/involeaf_b16.yaml --set kernel_size=3 batch_size=8
```

## Experimental protocol

**Splits.** Stratified 70/15/15, seed 42, committed to `splits/`. AgriTL-ViT uses 70/30
with no validation split, which leaves no honest way to select a checkpoint. We split
their 30% in half; the test partition is untouched until final evaluation. Paths are
sorted before shuffling, because filesystem order is not stable across machines.

**Effective batch is pinned at 64** via gradient accumulation. `batch_size` is a memory
detail (16 on the 8 GB RTX 5050, 64 on the 32 GB W7800); the effective batch is the
hyperparameter. Without pinning, September and October results would not be comparable.

**Robustness** reuses AgriTL-ViT's Table 17 protocol — ColorJitter, GaussianBlur,
RandomErasing, individually and combined — applied at test time to clean-trained
weights, so the whole stage costs no retraining. Corruptions are seeded per sample
index, so the same image gets the same corruption for every model on every run.
Training augmentation deliberately excludes colour jitter: lesion colour is the signal,
and training on colour-shifted data would contaminate that test.

**Resolution.** AgriTL-ViT reports a 200×200 test, but 200 is not divisible by the patch
size 16 — the patch embedding rejects the input, so it cannot have been a native
evaluation. We read "low resolution" as information loss at fixed input size (downscale
to the target, restore to 224). `native_resolution_transform` handles genuine
native-resolution evaluation and accepts only multiples of 16.

**Metrics.** `involeaf/utils/metrics.py` asserts that per-class F1 lies between its
precision and recall, and that support is derived from the labels. This is a direct
guard against the errors in AgriTL-ViT's tables — Tomato Early_blight at P 0.977 /
R 0.958 / F1 0.857 is impossible, and the Rice support column is identical to the Tomato
one across datasets of different sizes. Our tables are structurally incapable of
carrying the same mistake.

**Tables** are generated by `scripts/make_tables.py` from `results/*.json`. No number in
the report is hand-copied.

## Known limitations, stated up front

- **FLOP savings will not fully appear in wall-clock time.** There is no fused
  involution kernel; the original paper says so itself (Table 2: RedNet-50 14.3 ms vs
  ResNet-50 11.4 ms on GPU despite fewer FLOPs). Our shift-based implementation trades
  speed for memory further still. Report the gap rather than hiding it.
- **Involution may overlap functionally with attention.** At a 14×14 token grid a 7×7
  kernel covers half the grid, approaching what attention already does globally. The K
  ablation makes this answerable either way — if K=3 matches K=7, that is a finding
  about locality, and more interesting than a bare accuracy bump.
- **The AgriTL-ViT reproduction is an interpretation.** The paper's prose does not
  determine a unique wiring; ours is documented in `involeaf/models/agritl_vit.py`.
  It probably will not reach 98.5%, and that gap is itself a result.
- **PlantVillage tomato holds ~18,160 images**, against the 14,529 AgriTL-ViT reports.
  We use the full set and report the discrepancy rather than sampling down to match a
  number we cannot reconstruct.

## Layout

```
involeaf/ops/involution.py         the operator: shift + unfold paths, delta init
involeaf/models/vit_involution.py  ViT surgery and the token-grid bridge
involeaf/models/agritl_vit.py      AgriTL-ViT reproduction
involeaf/models/registry.py        build_model, layer-wise param groups
involeaf/data/                     splits, datasets, transforms, corruptions
involeaf/engine/                   training and evaluation loops
involeaf/utils/device.py           the only place that touches the accelerator
involeaf/utils/metrics.py          metrics with the consistency guard
configs/                           base.yaml + one file per model, ablations/
scripts/                           prepare_data, train, eval_robustness, benchmark, make_tables
splits/                            committed, deterministic
tests/                             pytest suite
```
