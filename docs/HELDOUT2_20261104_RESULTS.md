# Held-out v2 results: sketch-silhouette conditioning (run `heldout2_4a45adc61c5a9c09abac`, 2026-11-04)

Locked protocol: [HELDOUT2_PROTOCOL.md](HELDOUT2_PROTOCOL.md) (main `c9261cf`). 18 new held-out scenes (no drawing shared
with earlier sets) x seeds 0/1 x 4 arms = 144 images. InstanceDiffusion (Wang et al., CVPR 2024) is the external
generator; OCSD's contribution is the sketch-silhouette parser that turns each instance sketch into a mask for it.

## 1. Audit

144/144 images, 0 failed or skipped. Direct child: 72/72, one attempt, 2046 s including model load, NVIDIA L4, code
`c9261cf1`, peak CUDA 5.96 GiB, conditioning recorded per image (`boxes+centers` vs
`boxes+centers+sketch_silhouette_mask+256pt_polygon`). ControlNet + OCSD generation 8.8 min, evaluation 1.2 min.
`D_report` shows "running" only because the ZIP is written inside that stage. Log warnings are the known GLIGEN
cache messages (not used by these arms).

## 2. Primary endpoint (automatic, locked): sketch fidelity, silhouette vs box

| metric | scenes | wins / losses / ties | silhouette - box [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|
| facing accuracy (higher better) | 16 | 14 / 0 / 2 | +0.568 [0.406, 0.724] | 0.0001 | 0.0001 |
| sketch chamfer (lower better) | 18 | 18 / 0 / 0 | -0.019 [-0.024, -0.015] | < 0.0001 | < 0.0001 |

**Both primary tests are significant after Holm.** On 64 asymmetric drawings, objects faced the sketched direction
in 64/64 with silhouettes vs 28/64 with boxes (box is at chance level: boxes carry no direction).

## 3. All arms

| arm | facing acc | sketch chamfer | OWLv2 count acc | OPR | mIoU | relation acc | CLIP | analyst structure success | analyst naturalness (0-2) | s / image | sampling calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OCSD silhouette + InstanceDiffusion | **1.00** | 0.041 | **0.19** | **0.83** | **0.75** | **0.70** | **30.3** | **16/36** | 1.28 | 26.4 | 313 |
| InstanceDiffusion box | 0.43 | 0.060 | 0.14 | 0.72 | 0.62 | 0.53 | 29.4 | 13/36 | **1.53** | 26.5 | 313 |
| ControlNet scribble | 0.97 | **0.037** | 0.06 | 0.49 | 0.34 | 0.20 | 28.9 | 3/36 | 0.83 | 2.8 | 120 |
| OCSD soft-matte (earlier route) | 0.55 | 0.065 | 0.17 | 0.64 | 0.49 | 0.46 | 29.9 | 1/36 | 1.14 | 5.2 | 231 |

Means over scenes; CIs in [heldout2_20261104/summary.md](heldout2_20261104/summary.md). Paired differences,
silhouette - box (95% CI): OPR +0.110 [0.055, 0.171], mIoU +0.131 [0.077, 0.190], relations +0.171 [0.087, 0.261],
count accuracy +0.056 [-0.056, 0.194], CLIP +0.87 [-0.24, 1.92]. Silhouette - ControlNet: chamfer +0.004
[-0.004, 0.011] (same), facing +0.031 [0.000, 0.094], OPR +0.348 [0.249, 0.463], mIoU +0.405 [0.321, 0.499].

## 4. Secondary: analyst review (Claude, method author)

Done from blind A-D composites ([0](heldout2_20261104/review_composite_0.jpg) ...
[5](heldout2_20261104/review_composite_5.jpg)) before reading the key; record in
[analyst_review_blind.txt](heldout2_20261104/analyst_review_blind.txt). Not fully blind (styles are recognisable),
one reviewer, 8+-object counts are approximate.

* Structure success (count, colour, shape incl. pose): silhouette 16, box 13, ControlNet 3, OCSD soft-matte 1 (of 36).
  Silhouette vs box 4 / 3 / 11 scenes (p = 1.0, diff +0.083 [-0.083, 0.278]); vs ControlNet 11 / 2 / 5 (p = 0.02);
  vs OCSD soft-matte 11 / 0 / 7 (p = 0.001). Exploratory, uncorrected.
* Picks (most natural with right structure): silhouette 9, box 7, ControlNet 2, OCSD 1, none/tie 17.
* By object count (successes / 12): 3 objects 7 / 4 / 2 / 1; 5 objects 5 / 6 / 1 / 0; 8+ objects 4 / 3 / 0 / 0.
* **Cost of silhouettes: naturalness.** 7 silhouette images scored 0 (flat sticker/cut-out look, e.g. golden rabbits as
  cut-outs) vs 2 for box; mean 1.28 vs 1.53.

## 5. Claim (by the locked rule)

* **Supported:** OCSD's sketch-silhouette conditioning makes generated objects follow the sketch's shape and facing
  direction better than box conditioning (both primary tests significant), while count accuracy and analyst structure
  success are not worse (CIs include 0, point estimates higher) and layout metrics (OPR, mIoU, relations) are better.
* It matches ControlNet's sketch fidelity while keeping far better object count and layout, and beats the earlier
  OCSD soft-matte route on every automatic metric.
* **Not supported:** "better image quality / naturalness" (it is lower than box), or "state of the art". The
  generator is external; one analyst; QuickDraw only; 18 scenes.

## 6. Next improvement (not run)

The cut-out look comes from the mask holding the whole diffusion trajectory. A natural follow-up is to drop the mask
input after the layout steps (e.g. silhouette only for the first part of the grounding schedule); it would need its own
dev study and held-out set.
