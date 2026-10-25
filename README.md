# InvoLeaf

Involution inside the Vision Transformer encoder block, for plant leaf disease
classification, compared against ViT-B/16, ResNet-50 and a reproduction of AgriTL-ViT
(Expert Systems with Applications, 2025).

B.Tech Major Project, Semester 7. Mohammad Arsalan Siddiqui, Aditya Harikrishnan,
Jonathan Jesni, Om Arun Thombre. Advisor: Ms Anupama Arun.

**Full results, ablations, claims and limitations: [RESULTS.md](RESULTS.md).**

---

## What we found

On PlantVillage tomato (18,160 images, 3 seeds), with corruptions applied at test time:

| Model | Params | GMACs | Clean | Colour | Blur | Occlusion | Combined | Field photos |
|---|---|---|---|---|---|---|---|---|
| ViT-B/16 | 85.8M | 17.6 | 99.9 | **93.8** | **97.2** | **98.5** | **72.4** | **35.9** |
| **InvoLeaf** | 64.7M | 13.5 | 99.9 | 87.7 | 93.6 | 97.4 | 61.0 | 28.3 |
| ResNet-50 | 23.5M | 4.1 | 99.4 | 89.2 | 74.1 | 98.3 | 59.1 | 34.0 |
| AgriTL-ViT | 100.0M | 21.1 | 99.3 | 71.0 | 89.9 | 87.7 | 46.2 | 18.9 |

- **InvoLeaf matches ViT-B on clean accuracy with 25% fewer parameters and 23% fewer
  MACs**, but it is less robust than plain ViT-B on every corruption and on field photos,
  and slower per image (10.4 vs 4.5 ms) because involution has no fused GPU kernel.
- **InvoLeaf beats AgriTL-ViT on every test** (colour +17, occlusion +10, combined +15,
  field +9 points) while using 35% fewer parameters and 36% fewer MACs. AgriTL-ViT's
  1×1 convolutions cannot mix spatial information; involution can.
- **The robustness cost scales with how many blocks get involution.** Four involution
  blocks instead of twelve come close to ViT-B's robustness but save only 8% of
  parameters (single seed).
  It is not a learning-rate effect, and kernel size does not matter (K=3 ≈ K=7, and K=3
  is 2.4× cheaper to train).
- **A common occlusion test invents robustness differences.** `RandomErasing(value=
  "random")` before normalisation fills the box with values far outside the valid pixel
  range. Under it, InvoLeaf looked 10 points more robust than ViT-B and ResNet-50 sent
  99.8% of images to one class; with a valid (black) patch both effects vanish.
- **The same ordering holds on maize**, a second crop, and **every model collapses on
  real field photographs** (PlantDoc: 19–36% after ~99% in the lab).

## The idea

A ViT block is `x = x + attn(norm1(x))` then `x = x + mlp(norm2(x))`. The MLP works on
each token separately, so all spatial mixing happens inside attention. AgriTL-ViT fills
that MLP slot with six 1×1 convolutions and two 3×1 max-pools, presented as extracting
spatial features; but a 1×1 convolution is a per-token linear map, and the max-pools run
over the flattened token sequence (pooling the end of one row with the start of the
next). **Involution** (Li et al., CVPR 2021) is a spatial operator whose kernel at each
location is generated from that location's own features and shared across channels, so
a 7×7 neighbourhood is cheap. We put it in the slot.

Involution mixes space but not channels, so a naive swap deletes the block's channel
mixing. Three variants test this:

| Variant | Block | Params | Result |
|---|---|---|---|
| `pure` | `x + Inv(norm2(x))` | 36.4M | Breaks robustness (colour 62%): channel mixing is essential |
| **`inv_ffn`** (InvoLeaf) | `x + FFN_r2(Inv(norm2(x)))` | **64.7M** | Primary model |
| `parallel` | `x + FFN_r4(norm2(x)) + Inv(norm2(x))` | 93.0M | Capacity control; still less robust than ViT-B |

### Engineering details

**Two equivalent implementations.** The reference `unfold` formulation materialises a
`B × C·K² × HW` tensor: 6.5 GB of activations for 12 blocks at batch 16, which does not
fit an 8 GB laptop GPU. The `shift` path accumulates over the K² offsets as slices of one
padded tensor, so they share storage: 0.6 GB for the same 12 blocks. Both compute the same
function to machine precision (`tests/test_involution.py`). On the 48 GB cloud GPU,
`unfold` is twice as fast (624 vs 1,230 ms per step), so the configs use it.

**Delta initialisation.** Involution is inserted into a *pretrained* encoder. Zeroing
`span.weight` and setting its bias to a centre-tap delta makes the operator an exact
identity at step 0, so the pretrained network is undisturbed. Gradients still reach
`span.weight`, so it leaves the identity after one optimiser step. For the `parallel`
variant, where involution is *added* beside the pretrained FFN, the branch instead starts
at zero output, so the block starts exactly as the pretrained one.

## Reproducing

Install torch and torchvision for your backend first (https://pytorch.org; CUDA or ROCm),
then:

```bash
pip install -r requirements.txt
```

Download the datasets and build the splits. The committed split files mean everyone
trains on the same partition.

```bash
python scripts/prepare_data.py --dataset tomato --clone
```

```bash
python scripts/prepare_data.py --dataset plantdoc --clone
```

On machines behind an unreliable proxy (our cloud instances), use the batched,
retrying downloader instead:

```bash
bash scripts/fetch_data.sh
```

Train and evaluate one model:

```bash
python scripts/train.py --config configs/involeaf_b16.yaml --seeds 42 43 44
```

```bash
python scripts/eval_robustness.py --ckpt results/checkpoints/involeaf_b16_seed42.pth
```

Any config key can be overridden: `--set kernel_size=3 batch_size=32`.

For time-limited GPU sessions, `scripts/run_queue.sh` runs whole phases of the study,
starts only runs that fit the remaining time, and evaluates each run as soon as it
finishes. `python scripts/status.py` shows what is done, running and left.

```bash
BUDGET_MIN=150 SEED=42 bash scripts/run_queue.sh S
```

Tables, figures and cost numbers, all derived from `results/*.json`:

```bash
python scripts/make_tables.py
```

```bash
python scripts/make_plots.py
```

```bash
python scripts/plot_mechanism.py
```

```bash
python scripts/benchmark.py --flops-only
```

Tests:

```bash
python -m pytest
```

## Protocol

- **Splits.** Stratified 70/15/15, seed 42, committed in `splits/`. AgriTL-ViT uses
  70/30 with no validation set; we keep a validation split so the checkpoint is never
  chosen on test data. We use the full 18,160 tomato images (AgriTL-ViT reports 14,529).
- **Training.** 20 epochs, AdamW, warm-up then cosine decay, label smoothing 0.1, bf16,
  effective batch 64 regardless of GPU memory. Layer-wise learning rates: new modules
  1e-4, pretrained weights 1e-5, classifier 1e-3. Best checkpoint by validation macro-F1.
- **Robustness.** Colour jitter, Gaussian blur, occlusion (5–20% of the image, black
  patch) and all three combined, applied at test time only and seeded per image, so every
  model sees exactly the same corruptions. Training augmentation excludes colour jitter.
  Low resolution is downscale-then-restore (AgriTL-ViT's 200 px cannot be a native ViT
  evaluation: 200 is not divisible by the patch size 16).
- **Field photos.** PlantDoc tomato, 746 images over 9 of the 10 classes, never trained
  on; we report accuracy and macro-F1 over the classes present.
- **Metrics.** `involeaf/utils/metrics.py` asserts that every per-class F1 lies between
  its precision and recall and derives support from the labels. AgriTL-ViT's tables
  contain an impossible F1 (P 0.977, R 0.958, F1 0.857); ours cannot.
- **No hand-copied numbers.** Every table and figure is generated from the result JSONs.

## Limitations

- PlantVillage contains several photos of each leaf, so a random split can put the same
  leaf in training and test; clean accuracy is optimistic for every model.
- The placement, learning-rate and ViT-B K=3 ablations are single-seed.
- AgriTL-ViT is reproduced from the paper's text; the exact wiring is our interpretation
  (documented in `involeaf/models/agritl_vit.py`). It reaches 99.3% on tomato and 97.8% on
  maize, against the paper's 98.5% and 97.2%.
- In the last block, involution receives no gradient (the classification token bypasses
  it and nothing follows), so ~0.6M parameters stay at their initial identity. Removing
  it changes nothing measurable.

## Layout

```
involeaf/ops/involution.py         the operator: shift and unfold paths, delta / zero init
involeaf/models/vit_involution.py  ViT surgery, variants, token-grid bridge
involeaf/models/agritl_vit.py      AgriTL-ViT reproduction
involeaf/models/registry.py        build_model, layer-wise learning-rate groups
involeaf/data/                     splits, datasets, transforms, test-time corruptions
involeaf/engine/                   training (pause/resume) and evaluation
involeaf/utils/                    device, metrics with consistency guard, profiling, paths
configs/                           base.yaml, one file per model, ablations/
scripts/                           data, train, eval, queue, status, benchmark, tables, plots
splits/                            committed, deterministic splits
results/                           result JSONs (tables and figures are regenerated)
tests/                             pytest suite
```

## References

- D. Li et al., "Involution: Inverting the Inherence of Convolution for Visual
  Recognition", CVPR 2021.
- S. Srivastava, V. Sisaudia and S. Meena, "AgriTL-ViT: A Vision Transformer model with
  attention techniques for classification of plant leaf disease", Expert Systems with
  Applications 294, 128793, 2025.
- D. P. Hughes and M. Salathé, "An open access repository of images on plant health to
  enable the development of mobile disease diagnostics", arXiv:1511.08060 (PlantVillage).
- D. Singh et al., "PlantDoc: A Dataset for Visual Plant Disease Detection", CoDS-COMAD 2020.
