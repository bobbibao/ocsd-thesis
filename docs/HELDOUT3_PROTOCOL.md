# Held-out v3: complete OCSD v3 vs the original OCSD (protocol)

Status: locked by `assets/heldout_v3/protocol.json` (written by `tools/lock_heldout_v3.py` right after the dev v3 decision and before any held-out v3 output).

## Question

Does the complete OCSD v3 model beat the original OCSD (thesis/paper_v2) on 18 new scenes? We compare on object preservation, layout, and sketch shape. We also ask how OCSD v3 compares with GLIGEN and Collage, the strongest paper_v2 baselines.

## OCSD v3

OCSD v3 runs in this order:

1. The OCSD instance parser splits the sketch into one sketch per object.
2. The OCSD sketch-silhouette parser turns each object sketch into a mask and a 256-point polygon. It uses only the user's sketch, never oracle masks.
3. The pinned InstanceDiffusion core generates the image from boxes, points, phrases and those silhouettes.
4. The dev v3 decision sets the remaining options: whether the silhouettes are released after the layout steps, and whether the upstream SDXL-Refiner cascade runs.

Code: `ocsd/ocsd_v3.py`. The settings used for held-out v3 are in `assets/heldout_v3/protocol.json`, under `direct_arms` → `ocsd_v3`.

## Design

* Scenes: 18 QuickDraw-Scenes built with seed 20261107. That is 3, 5 and 8+ objects × simple/medium/complex, two scenes per cell. No drawing appears in held-out v1, held-out v2 or the dev scenes. Seeds are 0 and 1.
* Arms:
  * `ocsd_v3`, the proposed model.
  * `direct_box`, the same generator without the OCSD parser (ablation).
  * `ocsd`, the original OCSD. It uses the paper tier (steps 30, obj_steps 20, K 2, S1 = S2 = 100 identity learning) with the frozen paper_v2 tuning (alpha 0.1, LoRA 0.5, energy on phrase tokens with eta 20, region attention).
  * `gligen` (beta 1.0).
  * `collage` (strength 0.2, ControlNet scale 1.0).
  * `controlnet` (scale 1.0).
* The tuning values are copied verbatim from `results/paper_v2/tuning/tuned.json` into `assets/heldout_v3/tuned_paper_v2.json` and are never re-tuned.

## Analysis

* **Unit**: the scene (18). Each scene's value is its mean over the two seeds.
* **Primary family**: `ocsd_v3` vs `ocsd` on three metrics:
  * OPR (OWLv2, the same evaluator as paper_v2)
  * mIoU (OWLv2)
  * sketch chamfer (SAM, lower is better)

  Each uses an exact two-sided sign test over scenes, with Holm correction over the three. Paired bootstrap gives 95% confidence intervals.
* **Key secondary family**: `ocsd_v3` vs `gligen` and vs `collage` on the same three metrics, with Holm correction over the six tests.
* **Other secondary results** (confidence intervals only): count accuracy, OCE, relations, facing accuracy, found rate, CLIP, compute and time.
* **Blind review**: I review 36 six-panel sheets myself, since Bobby delegated the review, and commit my review before unblinding.
* **Claim rule**: we say "OCSD v3 improves over the original OCSD" only if two things hold:
  * at least one primary test has Holm p < 0.05 in OCSD v3's favour;
  * no primary metric is significantly worse.

  Otherwise we report the results as they are.

## Limits

* InstanceDiffusion (Wang et al., CVPR 2024) and the SDXL refiner are external pretrained components, credited as such.
* The original OCSD and the baselines are built on SD 1.4/1.5, so part of any difference comes from the generator itself. `direct_box` isolates what the parser adds on the same generator.
* Chamfer is partly circular for arms conditioned on the silhouette. OPR and mIoU are not.

## Run

`RUN_ALL.ipynb` with `RUN_MODE="heldout3"` on an L4 or A100. The InstanceDiffusion-core arms run first (at most 3600 s), then the four legacy methods (resumable, at most 300 min). Output goes to Drive `heldout_v3`, and the run downloads `ocsd_heldout3_results.zip` and `REVIEW_heldout3.zip`.
