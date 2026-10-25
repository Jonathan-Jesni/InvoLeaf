# InvoLeaf: final results

Every number here is derived from `results/*.json` (regenerate with
`python scripts/make_tables.py` → `results/tables.md`). Accuracy is test-set accuracy in %,
reported as mean ± sample standard deviation over seeds 42, 43, 44 unless marked otherwise.
Figures: `python scripts/make_plots.py` and `python scripts/plot_mechanism.py` write
`results/figures/fig1…fig10` (PNG at 300 dpi, plus PDF).

## The result in one paragraph

Replacing the feed-forward slot of every ViT-B/16 block with involution followed by a
half-width FFN (InvoLeaf) keeps clean accuracy at the ViT-B level (99.9%) with **25% fewer
parameters and 23% fewer MACs**, but it is **less robust than plain ViT-B under colour
shift, blur, combined corruption, low resolution and on field photographs, in every one
of the three seeds** (on occlusion alone the two are within seed-to-seed variation), and
2.3× slower per image because involution has no fused GPU kernel. Against the published
AgriTL-ViT design, which fills the same slot with 1×1 convolutions and doubles attention,
InvoLeaf is **better on every test in every seed** (colour +17, occlusion +10, combined
+15, field +9 points) while using 35% fewer parameters and 36% fewer MACs. The robustness cost scales with the number of blocks that
receive involution: with four blocks instead of twelve the model is about as robust as
ViT-B, but saves only 8% of parameters. A second crop (maize) reproduces the same
ordering. Along the way we found that a widely used occlusion test (random erasing with
`value="random"`) produces out-of-range pixels and invents robustness differences; with a
physically valid patch the effect disappears.

## Setup (for the methods section)

- **Data.** PlantVillage tomato, 18,160 images, 10 classes, stratified 70/15/15 split
  (12,712 / 2,725 / 2,723), seed 42, committed in `splits/`. PlantDoc tomato field photos
  as a cross-domain test only: 746 images covering 9 of the 10 classes (no Target Spot;
  spider mites has 2 images). Maize: PlantVillage, 3,852 images, 4 classes, same split
  rule (578 test images).
  AgriTL-ViT reports 14,529 tomato images with a 70/30 split and no validation set; we use
  the full set and add a validation split so the checkpoint is never chosen on test data.
- **Training.** 20 epochs, AdamW, 2 warm-up epochs then cosine decay, label smoothing 0.1,
  bf16, gradient clipping 1.0, effective batch 64. Layer-wise learning rates: new modules
  1e-4, pretrained weights 1e-5, classifier 1e-3 (ResNet-50 backbone 1e-4, because 1e-5
  under-fitted). Best checkpoint chosen on validation macro-F1.
- **Models.** ViT-B/16 (`augreg_in21k_ft_in1k`), ResNet-50 (`a1_in1k`), AgriTL-ViT
  reproduced from the paper's text (wiring documented in `involeaf/models/agritl_vit.py`),
  InvoLeaf (`x + FFN_r2(Inv(norm2(x)))`, K=7, 16 channels per group, GroupNorm, starts as
  an exact identity so the pretrained network is undisturbed at step 0).
- **Corruptions** are applied only at test time, seeded per image so every model sees
  the same corruption: colour jitter (brightness/contrast/saturation 0.4, hue 0.1),
  Gaussian blur (5×5, σ 0.6–2.0), occlusion (a box covering 5–20% of the image, black
  fill), and all three combined. Low resolution: downscale to 128 px, restore to 224.
- **Hardware.** AMD Radeon PRO W7900D 48 GB (ROCm). Seed 42 of InvoLeaf used the
  memory-saving `shift` implementation; all other runs used `unfold`. Both compute the same
  function (tested in `tests/test_involution.py`).

## Main comparison, tomato (3 seeds)

| Model | Params | Clean | Colour | Blur | Occlusion | Combined | 128 px | Field (PlantDoc) |
|---|---|---|---|---|---|---|---|---|
| ViT-B/16 | 85.8M | 99.9 ± 0.0 | **93.8 ± 0.1** | **97.2 ± 0.8** | **98.5 ± 0.7** | **72.4 ± 3.8** | **98.7 ± 0.2** | **35.9 ± 3.6** |
| **InvoLeaf** | 64.7M | 99.9 ± 0.1 | 87.7 ± 0.6 | 93.6 ± 1.1 | 97.4 ± 1.1 | 61.0 ± 3.5 | 96.4 ± 0.6 | 28.3 ± 1.9 |
| ResNet-50 | 23.5M | 99.4 ± 0.1 | 89.2 ± 0.2 | 74.1 ± 2.2 | 98.3 ± 0.1 | 59.1 ± 1.9 | 84.4 ± 2.8 | 34.0 ± 0.5 |
| AgriTL-ViT | 100.0M | 99.3 ± 0.2 | 71.0 ± 1.5 | 89.9 ± 0.4 | 87.7 ± 0.5 | 46.2 ± 2.4 | 91.9 ± 1.0 | 18.9 ± 0.8 |

**Per-seed comparison** (same seed, same corrupted images), which is stronger evidence
than overlapping ± bars:

| InvoLeaf minus … | Colour | Blur | Occlusion | In-range noise | Combined | 128 px | Field |
|---|---|---|---|---|---|---|---|
| ViT-B, seeds 42 / 43 / 44 | −6.6 / −5.7 / −5.8 | −3.2 / −4.0 / −3.7 | −3.0 / −0.7 / +0.4 | −6.1 / −1.2 / −2.1 | −18.8 / −11.0 / −4.4 | −1.9 / −2.9 / −2.1 | −13.0 / −7.9 / −1.9 |
| AgriTL-ViT, seeds 42 / 43 / 44 | +14.5 / +17.4 / +18.3 | +4.9 / +2.5 / +3.7 | +9.0 / +9.9 / +10.2 | +4.5 / +5.3 / +7.2 | +9.8 / +13.6 / +21.1 | – | +7.1 / +8.8 / +12.3 |

InvoLeaf is below ViT-B in all three seeds on every test except occlusion, where the
sign flips between seeds; it is above AgriTL-ViT on every test in every seed.

Field-photo macro-F1 over the 9 classes present: ViT-B 0.241, ResNet 0.215, InvoLeaf
0.189, AgriTL 0.109. Every model loses 60–80 points from lab to field; none is usable on
field photos, which is the strongest argument in the report for field-collected data.
Figure: **fig1** (all conditions), **fig4** (lab vs field), **fig3** (colour vs occlusion).

## Cost (`results/flops.json`, `results/benchmark.json`)

| Model | Params | GMACs | GPU latency, batch 1 | Peak training memory, batch 64 |
|---|---|---|---|---|
| ResNet-50 | 23.5M | 4.11 | 5.7 ms | 3.0 GB |
| ViT-B/16 | 85.8M | 17.58 | 4.5 ms | 4.8 GB |
| **InvoLeaf** | 64.7M | 13.50 | 10.4 ms | 29.5 GB (`unfold`; ~6.9 GB with `shift`) |
| InvoLeaf, K=3 | 60.3M | 12.57 | – | – |
| AgriTL-ViT | 100.0M | 21.10 | 12.7 ms | 8.0 GB |

GMACs (multiply-accumulates, the unit usually reported as "FLOPs" for ViTs) count
fvcore's total plus two terms it cannot see: the attention matrix products and the
involution multiply-accumulate. Latency was measured on the W7900D; the cloud CPU timings
were not credible and are not reported. The lower MAC count does not become lower latency
because involution has no fused kernel; the original involution paper reports the same
(RedNet-50 slower than ResNet-50 despite fewer FLOPs). Report this plainly.

## Methodological finding: the occlusion test

`torchvision.transforms.RandomErasing(value="random")`, applied before normalisation as
in many pipelines, fills the box with N(0,1) noise: pixel values of −4…+4 where real
pixels lie in 0…1. Our first round used it, and it produced two striking but false
results. With the **same boxes** filled by valid pixels:

| Model | Noise patch (out of range) | Black patch | Uniform noise in [0,1] |
|---|---|---|---|
| ViT-B/16 | 84.5 ± 4.7 | 98.5 ± 0.7 | 99.2 ± 0.2 |
| InvoLeaf | **94.6 ± 1.2** | 97.4 ± 1.1 | 96.0 ± 2.5 |
| ResNet-50 | **29.7 ± 0.4** | 98.3 ± 0.1 | 94.6 ± 5.3 |
| AgriTL-ViT | 71.6 ± 5.0 | 87.7 ± 0.5 | 90.4 ± 2.1 |

With the noise patch, InvoLeaf looked 10 points more occlusion-robust than ViT-B, and
ResNet-50 sent 99.8% of occluded images to one class (Yellow Leaf Curl Virus). Both
effects vanish with a black patch. **fig7** shows ResNet's prediction distribution under
both fills. The point for the report: a robustness benchmark must keep the perturbed image
inside the valid input range, or it measures sensitivity to impossible inputs.

One nuance: ResNet-50's weakness to *noise texture* is partly real. With in-range uniform
noise it still drops on maize to 29 ± 8% (on seeds 43 and 44 over 90% of images go to a
single class, a different class per seed) and on tomato seed 43 it sends 39% of images to
Yellow Leaf Curl Virus. The out-of-range values exaggerate it to a total collapse; the
transformers are not affected by in-range noise at all.

## Ablations

**Variants (3 seeds).** What fills the FFN slot matters:

| Variant | Params | Clean | Colour | Occlusion | Combined | Field |
|---|---|---|---|---|---|---|
| `pure` (involution only, no FFN) | 36.4M | 98.5 | 62.3 | 83.0* | 41.6* | 21.4 |
| `inv_ffn` (InvoLeaf) | 64.7M | 99.9 | 87.7 | 97.4 | 61.0 | 28.3 |
| `parallel` (full FFN + involution beside it) | 93.0M | 99.8 | 90.6 | 98.1 | 68.5 | 26.1 |

\* 2 seeds. Removing the FFN entirely breaks the model, so channel mixing is essential.
Adding involution beside the full pretrained FFN costs robustness too (field 26 vs ViT-B
36), so the loss comes from involution in the residual stream, not only from narrowing
the FFN.

**Kernel size (ViT-S/16, 3 seeds).** K=3, 5, 7 are indistinguishable (colour 82.7 / 83.1 /
82.0, field 32.4 / 32.2 / 31.3), all below plain ViT-S (colour 90.3, field 37.3). The same
holds at ViT-B size (seed 42): K=3 gives colour 88.4, combined 60.7, field 27.9, against
87.1 / 57.6 / 26.4 for K=7, with 60.3M parameters and 2.4× faster training (19 vs 45 min).
**K=3 is the better choice: same robustness, lower cost.**

**Where involution goes (ViT-B, seed 42).** Robustness falls as more blocks get involution:

| Placement | Params | Colour | Blur | Combined | Field |
|---|---|---|---|---|---|
| none (ViT-B) | 85.8M | 93.7 | 97.6 | 76.5 | 39.4 |
| every 3rd block (1, 4, 7, 10) | 78.8M | 93.0 | 97.3 | 75.8 | 33.9 |
| last 4 blocks | 78.8M | 96.8 | 98.5 | 70.1 | 39.0 |
| blocks 1–11 | 66.5M | 87.4 | 92.1 | 60.6 | 26.9 |
| all 12 (InvoLeaf) | 64.7M | 87.1 | 94.4 | 57.6 | 26.4 |

Placement is a dial between cost and robustness. Single seed, and validation accuracy is
~99.9% for all of them, so validation cannot pick one: report this as an ablation, not as
a tuned model.

**Is it the learning rate? (seed 42).** InvoLeaf's FFN is sliced from pretrained weights
but trains at the new-module rate (1e-4), ten times the rate of ViT-B's MLP. Keeping it at
the pretrained rate does not recover robustness (colour 85.7, blur 86.2, combined 54.6,
field 29.8). The gap is architectural.

**Dead last block.** In the last block the CLS token skips involution and no later block
carries patch information back to it, so no gradient reaches that block's involution; its
kernels stay at the identity they started from (~0.6M unused parameters). Dropping it
(blocks 1–11) gives the same results as InvoLeaf (table above), confirming the analysis.

## Second crop: maize (3 seeds)

| Model | Clean | Colour | Blur | Occlusion | Combined |
|---|---|---|---|---|---|
| ViT-B/16 | 98.8 ± 0.1 | 97.1 ± 0.5 | 98.7 ± 0.3 | 92.7 ± 1.3 | 88.1 ± 0.6 |
| InvoLeaf | 97.6 ± 0.8 | 94.6 ± 0.8 | 98.2 ± 0.5 | 80.7 ± 3.8 | 78.1 ± 2.8 |
| ResNet-50 | 97.4 ± 0.3 | 96.5 ± 0.9 | 95.9 ± 1.1 | 93.7 ± 1.0 | 87.0 ± 4.0 |
| AgriTL-ViT | 97.8 ± 0.1 | 92.3 ± 0.7 | 96.9 ± 0.7 | 62.2 ± 1.0 | 64.8 ± 0.8 |

Same ordering as tomato. Our AgriTL-ViT reproduction reaches 97.8% on maize against the
97.2% the paper reports, consistent with the published accuracy. ResNet-50 drops to
29.4 ± 8.4% on in-range uniform noise patches (tomato: 94.6%): CNNs react to noise texture
even when the pixels are valid. Almost all clean maize errors are Northern Leaf Blight
predicted as Cercospora leaf spot, two visually similar diseases (e.g. InvoLeaf seed 43:
17 of 147). With 578 test images, one image is 0.17 points; checkpoints were chosen on an
equally small validation set, so maize test F1 sits 2–4 points below validation F1 for
ResNet and InvoLeaf.

## Mechanism and interpretability (seed 42, `results/mechanism_seed42.json`)

Caveat: this analysis was run before the occlusion issue was found, so its "occlusion"
condition is the out-of-range noise patch; the colour-shift results are unaffected. A
black-patch re-run (`python scripts/analyze_mechanism.py`, which now defaults to
`--occlusion erasing_black` and writes `results/mechanism_seed42_black.json`) replaces
the occlusion panels of figures 8–10; until it exists, those panels show the noise patch
and say so in their titles.

- **The kernels are not the weak point.** Under every corruption, InvoLeaf's generated
  kernels change about 5× less than their inputs; colour shift moves them least (0.083
  relative change). Our prediction that content-generated kernels would amplify colour
  shifts was wrong.
- **Under colour shift InvoLeaf's features move as much as ViT-B's (0.33 vs 0.31), but
  its predictions flip 3× as often (15% vs 5%)**: its classes sit closer together, so the
  same disturbance crosses a decision boundary more often. Why is not established.
- **fig9** shows where kernels mix neighbours (mostly background and leaf edges; leaf
  interiors stay close to identity). **fig10** shows feature drift per block for ViT-B and
  InvoLeaf. **fig8** (Grad-CAM) is informative for ResNet; last-block Grad-CAM on ViTs is
  diffuse and speckled over the background, a known weakness of the method for
  transformers, so do not over-interpret those columns.

## What to claim, and what not to

Claim:
- InvoLeaf matches ViT-B clean accuracy at 25% fewer parameters and 23% fewer MACs.
- InvoLeaf beats AgriTL-ViT on accuracy, every robustness test and cost; 1×1 convolutions
  cannot mix spatial information, involution can, and it shows.
- Involution costs robustness relative to plain ViT-B (colour, blur, combined, low
  resolution, field: worse in all three seeds); the cost grows with the number of
  involution blocks and is not explained by the learning rate.
- On occlusion, InvoLeaf and ViT-B are statistically indistinguishable.
- Kernel size does not matter; K=3 is the efficient choice.
- The out-of-range occlusion test creates false robustness results.
- All models, ours included, fail on field photos (18–36%).

Do not claim:
- That InvoLeaf is more robust than ViT-B (it is not, on any realistic test).
- That InvoLeaf is worse than ViT-B under occlusion specifically (within seed noise).
- That ResNet-50 collapses under occlusion: with a black patch it does not; only
  noise-textured patches hurt it (badly on maize, mildly on tomato).
- That InvoLeaf is faster (it is slower in wall-clock time).
- Any number from `results/superseded/` (the first `parallel` run started from a
  perturbed initialisation; fixed and rerun).

## Limitations to state

- PlantVillage contains several photos of the same leaf, so a random split can place one
  leaf in both training and test; clean accuracy is optimistic for all models alike.
- Placement, learning-rate, first11 and ViT-B K=3 ablations are single-seed.
- `vit_b16_baseline` seed 42 was paused after epoch 8 and resumed from its full saved
  state (weights, optimiser, scheduler, random-number streams); every other run trained
  in one go.
- The seed-42 ViT-S kernel-ablation runs predate the in-range occlusion tests, so those
  columns average seeds 43 and 44 only.
- Retrained ResNet-50 and InvoLeaf runs are reproducible only to a few tenths of a point
  (non-deterministic GPU kernels); ViT-B and AgriTL-ViT reproduce exactly.
- Weight decay is also applied to the new modules' biases and normalisation weights,
  equally for InvoLeaf and AgriTL-ViT.
- AgriTL-ViT is reproduced from the paper's description; its exact wiring is our reading.
