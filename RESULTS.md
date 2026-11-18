# InvoLeaf: final results

Every number here is derived from `results/*.json` (regenerate with
`python scripts/make_tables.py` → `results/tables.md`). Accuracy is test-set accuracy in %,
reported as mean ± sample standard deviation over seeds 42, 43, 44 unless marked otherwise.
Figures: `python scripts/make_plots.py` and `python scripts/plot_mechanism.py` write
`results/figures/fig1…fig11` (PNG at 300 dpi, plus PDF).

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
ViT-B, but saves only 8% of parameters. A control that puts a plain, static 7×7
depthwise convolution in the same slot matches InvoLeaf on every realistic test
(3 seeds) at lower cost, so the robustness cost comes from putting a local spatial
operator plus a half-width FFN in place of the MLP. It is not specific to involution, and
involution's content-generated kernels bring no measurable benefit over a fixed kernel in
this setting. A second crop (maize) reproduces the same
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
| ViT-B/16 | 85.8M | 99.9 ± 0.0 | **93.8 ± 0.1** | **97.2 ± 0.8** | **98.5 ± 0.7** | **72.4 ± 3.8** | **98.7 ± 0.1** | **35.9 ± 3.6** |
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

### Per class (3 seeds)

On the lab test set every class is easy for every model: clean F1 is at least 0.98 for
all ten classes and all four models (ViT-B and InvoLeaf ≥ 0.994; the weakest cells are
ResNet and AgriTL-ViT on Early blight, 0.988 / 0.983, and Target spot, 0.984 / 0.987).

On field photos the picture changes completely. Recall per class (share of each class's
field images classified correctly):

| Class (field images) | ViT-B | InvoLeaf | ResNet-50 | AgriTL-ViT |
|---|---|---|---|---|
| Late blight (111) | **0.96** | 0.58 | 0.78 | 0.77 |
| Early blight (88) | 0.44 | **0.76** | 0.60 | 0.27 |
| Septoria leaf spot (151) | 0.45 | 0.31 | **0.56** | 0.07 |
| Yellow leaf curl virus (76) | **0.35** | 0.21 | 0.19 | 0.15 |
| Healthy (63) | **0.15** | 0.11 | 0.11 | 0.04 |
| Bacterial spot (110) | **0.13** | 0.06 | 0.01 | 0.05 |
| Leaf mould (91) | 0.04 | 0.05 | **0.08** | 0.01 |
| Mosaic virus (54) | 0.01 | 0.00 | 0.01 | 0.00 |
| Spider mites (2) | 0 | 0 | 0 | 0 |

The models do not degrade evenly: on field photos they fall back on one or two
"blight-like" answers. Share of all field predictions going to the most-predicted class:
ViT-B 44% Late blight, InvoLeaf 47% Early blight, ResNet 33% Late blight, AgriTL-ViT 58%
Late blight. Bacterial spot, leaf mould, mosaic virus and healthy leaves are almost never
recognised; most healthy field leaves are called diseased. So the field accuracies in the
main table mostly measure how often a photo happens to show a blight. This is the clearest
evidence in the project that lab-trained leaf-disease classifiers are not ready for field
use, whichever architecture is used.

## Cost (`results/flops.json`, `results/benchmark.json`)

| Model | Params | GMACs | GPU latency, batch 1 | Peak training memory, batch 64 |
|---|---|---|---|---|
| ResNet-50 | 23.5M | 4.11 | 5.7 ms | 3.0 GB |
| ViT-B/16 | 85.8M | 17.58 | 4.5 ms | 4.8 GB |
| **InvoLeaf** | 64.7M | 13.50 | 10.4 ms | 29.5 GB (`unfold`; ~6.9 GB with `shift`) |
| InvoLeaf, K=3 | 60.3M | 12.57 | – | – |
| Depthwise-conv control | 57.9M | 12.09 | – | 4.3 GB‡ |
| AgriTL-ViT | 100.0M | 21.10 | 12.7 ms | 8.0 GB |

‡ From the training logs (InvoLeaf's training logs give 30.0 GB on the same
measurement). Training time per run: control 22 min, InvoLeaf 45 min, ViT-B 13 min.

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
| `pure` (involution only, no FFN) | 36.4M | 98.5 | 62.5 | 84.4 | 41.4 | 21.6 |
| `inv_ffn` (InvoLeaf) | 64.7M | 99.9 | 87.7 | 97.4 | 61.0 | 28.3 |
| `parallel` (full FFN + involution beside it) | 93.0M | 99.8 | 90.6 | 98.1 | 68.5 | 26.1 |

Removing the FFN entirely breaks the model, so channel mixing is essential.
Adding involution beside the full pretrained FFN costs robustness too (field 26 vs ViT-B
36), so the loss comes from involution in the residual stream, not only from narrowing
the FFN.

**Kernel size (ViT-S/16, 3 seeds).** K=3, 5, 7 are indistinguishable (colour 82.6 / 82.6 /
81.5, combined 55.2 / 57.5 / 55.2, field 31.9 / 32.1 / 31.6), all below plain ViT-S
(colour 90.3, combined 78.4, field 37.3). The same
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

**Is it involution itself? Depthwise-convolution control (ViT-B, 3 seeds).** The same
model as InvoLeaf (all 12 blocks, FFN_r2 after the spatial operator, identity at step 0,
same learning rates), with involution replaced by a fixed 7×7 depthwise convolution.
Planned as a control after the main results; not used to choose or tune anything.

| Model | Params | Clean | Colour | Blur | Occlusion | In-range noise | Combined | 128 px | Field | Field F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| ViT-B/16 | 85.8M | 99.9 | 93.8 ± 0.1 | 97.2 ± 0.8 | 98.5 ± 0.7 | 99.2 ± 0.2 | 72.4 ± 3.8 | 98.7 ± 0.1 | 35.9 ± 3.6 | 0.241 |
| InvoLeaf | 64.7M | 99.9 | 87.7 ± 0.6 | 93.6 ± 1.1 | 97.4 ± 1.1 | 96.0 ± 2.5 | 61.0 ± 3.5 | 96.4 ± 0.6 | 28.3 ± 1.9 | 0.189 |
| Depthwise-conv control | 57.9M | 99.8 | 87.9 ± 1.3 | 93.9 ± 2.8 | 97.8 ± 1.0 | 86.0 ± 14.1 | 58.0 ± 3.5 | 96.7 ± 1.5 | 27.7 ± 1.0 | 0.185 |

Per seed (42 / 43 / 44), control minus InvoLeaf: colour +0.6 / −1.5 / +1.4, blur
+1.5 / −1.6 / +1.1, occlusion +2.8 / −1.1 / −0.7, combined +3.2 / −6.8 / −5.3, field
+0.5 / −0.9 / −1.5. The sign flips between seeds on every test, so on these the two are
indistinguishable. Control minus ViT-B is negative in all three seeds on colour, blur,
combined, 128 px and field, the same pattern as InvoLeaf. The one test where they differ
is in-range noise: the control drops on two seeds (98.1 / 70.5 / 89.3; on seed 43, 35% of
images go to Early blight), whereas InvoLeaf stays at 93–98. Involution may be steadier
against noise texture than a fixed kernel, but three seeds with this much spread are
not enough to claim it.

The conclusion: **any local spatial operator in the FFN slot, followed by a half-width
FFN, costs the same robustness against ViT-B.** The cost does not come from involution's
content-generated kernels, and those kernels give no measurable gain over a static
kernel here. The control is also cheaper (57.9M parameters, half InvoLeaf's training
time, 4.3 GB training memory against 30 GB), because depthwise convolution is a standard,
fused GPU operation. We have not benchmarked its latency; we predict it is close to
ViT-B's. **fig11** shows the three models side by side.

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

## Mechanism and interpretability (seed 42, `results/mechanism_seed42_black.json`)

Inference only, on 300 test images with the same seeded corruptions as the evaluation;
occlusion and combined use the black patch. "Drift" is the relative change of a feature
against the clean image; "flips" are predictions that change.

| Model | Colour: final-feature drift / flips | Occlusion: drift / flips | Occlusion damage outside the box (mean over blocks) |
|---|---|---|---|
| ResNet-50 | 0.68 / 12% | 0.49 / 1% | 0.29 |
| ViT-B/16 | 0.31 / 5% | 0.33 / 0.3% | 0.27 |
| InvoLeaf | 0.33 / 15% | **0.23 / 4%** | **0.23** |

- **InvoLeaf's features are as stable as ViT-B's, or more, but its predictions flip
  more.** Under colour shift the features move as much as ViT-B's but 3× as many
  predictions change; under occlusion they move *less* (and the damage spreads least
  outside the box, fig10), yet 4% of predictions change against ViT-B's 0.3% (1 of 300 images). The same
  pattern under both corruptions suggests InvoLeaf's classes sit closer together, so a
  smaller disturbance crosses a decision boundary. This is an interpretation, not tested.
- **The kernels are not the weak point.** They change about 5× less than their inputs
  under every corruption, least under colour shift (0.083). Under occlusion they change
  4.7× more inside the box than outside (0.229 vs 0.049): involution keeps the effect
  local. Our prediction that content-generated kernels would amplify colour shifts was
  wrong.
- **fig9** shows where the kernels mix neighbours: mostly background and leaf edges; leaf
  interiors and the uniform black patch stay close to identity. **fig8** (Grad-CAM) is
  informative for ResNet; last-block Grad-CAM on ViTs is diffuse and speckled over the
  background, a known weakness of the method for transformers, so do not over-interpret
  those columns.
- The first version of this analysis (`results/mechanism_seed42.json`) used the
  out-of-range noise patch. It produced two effects that disappear with the black patch:
  a 47× blow-up of ResNet's features (0.49 with the black patch), and kernels inside the
  box switching to heavy neighbour mixing (0.45 of their weight off-centre; with the
  black patch 0.09, below the clean value of 0.16). Neither should be cited.

## What to claim, and what not to

Claim:
- InvoLeaf matches ViT-B clean accuracy at 25% fewer parameters and 23% fewer MACs.
- InvoLeaf beats AgriTL-ViT on accuracy, every robustness test and cost; 1×1 convolutions
  cannot mix spatial information, while involution (and, per the control, a depthwise
  convolution) can.
- Involution costs robustness relative to plain ViT-B (colour, blur, combined, low
  resolution, field: worse in all three seeds); the cost grows with the number of
  involution blocks and is not explained by the learning rate.
- A static depthwise convolution in the same slot gives the same robustness at lower
  cost, so the cost belongs to the design (local operator + half-width FFN in place of
  the MLP), not to involution specifically.
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
- That involution's content-generated kernels help, or that they cause the robustness
  loss: the depthwise-conv control rules out both on our tests.
- Any number from `results/superseded/` (the first `parallel` run started from a
  perturbed initialisation; fixed and rerun).

## Limitations to state

- PlantVillage contains several photos of the same leaf, so a random split can place one
  leaf in both training and test; clean accuracy is optimistic for all models alike.
- Placement, learning-rate, first11 and ViT-B K=3 ablations are single-seed.
- `vit_b16_baseline` seed 42 was paused after epoch 8 and resumed from its full saved
  state (weights, optimiser, scheduler, random-number streams); every other run trained
  in one go.
- There is no "half-width FFN with no spatial operator" run, so the cost of narrowing the
  FFN cannot be separated from the cost of adding a spatial operator. `parallel` (full FFN
  plus involution) also loses field accuracy, which points at the spatial operator, but
  that is one indirect data point.
- The depthwise-conv control's latency was not benchmarked.
- The seed-42 ViT-S kernel-ablation runs and `pure` seed 43 were retrained on Nov 17 to
  get the in-range occlusion tests (their checkpoints had been deleted); their other
  numbers moved by a few tenths, like the other retrains.
- Retrained ResNet-50 and InvoLeaf runs are reproducible only to a few tenths of a point
  (non-deterministic GPU kernels); ViT-B and AgriTL-ViT reproduce exactly.
- Weight decay is also applied to the new modules' biases and normalisation weights,
  equally for InvoLeaf and AgriTL-ViT.
- AgriTL-ViT is reproduced from the paper's description; its exact wiring is our reading.
