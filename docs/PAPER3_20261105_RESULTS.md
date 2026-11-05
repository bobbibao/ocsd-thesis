# Paper benchmark v3 results (2026-11-05): OCSD v3 confirmed on the 72 paper_v2 scenes

**Verdict under the locked claim rule: confirmed.** On the full paper benchmark, OCSD v3 improves over the original OCSD on two of the three primary metrics after Holm correction:

* OPR: +0.100, Holm p = 0.013.
* mIoU: +0.152, Holm p < 0.0001.

Sketch chamfer is not significantly different, and no primary metric is worse. Held-out v3 found the same result on 18 new scenes ([results](HELDOUT3_20261105_RESULTS.md)), so the direction now holds on two separate scene sets.

Protocol: [PAPER3_PROTOCOL.md](PAPER3_PROTOCOL.md), locked at main `afaadee` before any output. Evidence is in [paper3_20261105/](paper3_20261105/).

## Run audit

* Run `20261105T085721Z_c08740c2`, at code `afaadee`, on an NVIDIA L4.
* All 864 of 864 images are present: 72 scenes × 2 seeds × 6 arms.
* The InstanceDiffusion child run is `paper3_7e9e8a9921f6cb7a8da1`: 288 of 288 images in 7200 s. The four legacy methods made 576 of 576 images in 152 min with 0 failures. Evaluation took 6 min. D_report shows "running" only because the report stage writes the ZIP from inside itself.
* The configurations used match the lock:
  * `ocsd`: alpha 0.1, LoRA 0.5, energy on phrase tokens with eta 20, region attention, identity learning S1 = S2 = 100, steps 30, obj_steps 20, K 2.
  * GLIGEN: beta 1.0.
  * Collage: strength 0.2.
  * ControlNet: scale 1.0.
  * OCSD v3: silhouettes for the layout steps only, plus the SDXL-Refiner at revision `5d4cfe85…`.
* **The scenes and evaluator are the same as paper_v2.** All 72 imported scenes matched the paper_v2 object counts. On the scene/seed pairs that both runs share, this run reproduces paper_v2 to within one point:

  | method | OPR, paper_v2 → now | mIoU, paper_v2 → now |
  |---|---|---|
  | GLIGEN | 0.816 → 0.813 | 0.664 → 0.660 |
  | Collage | 0.818 → 0.827 | 0.600 → 0.610 |
  | ControlNet | 0.660 → 0.658 | 0.498 → 0.499 |
  | original OCSD | 0.740 → 0.734 | 0.541 → 0.545 |

## Primary family: `ocsd_v3` vs `ocsd` (72 scenes, exact sign test, Holm over three)

| metric | wins / losses / ties | difference [95% CI] | p (Holm) |
|---|---|---|---|
| **OPR** | 35 / 15 / 22 | **+0.100 [0.047, 0.153]** | **0.013** |
| **mIoU** | 59 / 13 / 0 | **+0.152 [0.109, 0.196]** | **< 0.0001** |
| sketch chamfer (lower is better) | 38 / 34 / 0 | −0.005 [−0.009, −0.001] | 0.72 |

## Key secondary family (Holm over six)

| comparison | OPR | mIoU | sketch chamfer |
|---|---|---|---|
| vs GLIGEN | +0.011, Holm p 1.0 | **+0.030, Holm p 0.015** | **−0.016, 65/7 scenes, Holm p < 0.0001** |
| vs Collage | −0.012, Holm p 1.0 | +0.075, Holm p 0.051 (trend) | 0.000, Holm p 1.0 |

## All arms (means over 72 scenes)

| arm | OPR | mIoU | chamfer | facing | count acc. | relations | time per image (s) |
|---|---|---|---|---|---|---|---|
| **ocsd_v3** | 0.812 | 0.682 | 0.050 | 0.82 | 0.38 | 0.59 | 23.6 |
| direct_box | 0.829 | 0.730 | 0.060 | 0.51 | 0.45 | 0.61 | 23.6 |
| ocsd (original) | 0.712 | 0.530 | 0.055 | 0.92 | 0.33 | 0.45 | 6.7 |
| gligen | 0.801 | 0.652 | 0.066 | 0.51 | 0.47 | 0.56 | 2.9 |
| collage | 0.824 | 0.607 | 0.049 | 0.90 | 0.45 | 0.62 | 4.0 |
| controlnet | 0.655 | 0.492 | 0.038 | 0.98 | 0.26 | 0.32 | 2.8 |

By object count, OCSD v3 vs the original OCSD:

| objects | OPR | mIoU |
|---|---|---|
| 1 | 0.97 vs 0.89 | 0.88 vs 0.71 |
| 3 | 0.87 vs 0.80 | 0.75 vs 0.61 |
| 5 | 0.83 vs 0.65 | 0.66 vs 0.45 |
| 8+ | 0.57 vs 0.51 | 0.44 vs 0.35 |

## Blind analyst review (24 pre-registered sheets, the scenes numbered 003)

I wrote the review before opening `review_key.json` or any summary file: [analyst_review_blind.txt](paper3_20261105/analyst_review_blind.txt), sha256 `ee0f8c55…49d0e4`. The unblinded numbers are in [analysis.json](paper3_20261105/analysis.json).

| arm | structure pass /24 | naturalness (0-2) |
|---|---|---|
| direct_box | 13 | 1.67 |
| **ocsd_v3** | 10 | 1.63 |
| gligen | 4 | 1.21 |
| collage | 7 | 0.46 |
| controlnet | 3 | 0.63 |
| ocsd (original) | 5 | 0.25 |

Naturalness sign tests over scenes (secondary):

* vs the original OCSD: OCSD v3 better in 11 scenes, worse in 0 (p = 0.001).
* vs Collage: 9 / 1 (p = 0.02).
* vs ControlNet: 10 / 0 (p = 0.002).
* vs GLIGEN and direct_box: no significant difference.

## Interpretation

1. **The claim holds on both scene sets.** The complete OCSD v3 beats the original OCSD on object preservation (OPR) and layout (mIoU), and its images are much more natural. With 72 scenes, OPR is now significant too; on held-out v3 it was only a trend.
2. **Against the baselines.** OCSD v3 has significantly better layout (mIoU) and sketch-shape fidelity than GLIGEN. Against Collage, the strongest paper_v2 baseline, it is level on OPR and chamfer and better on mIoU (a trend after correction), and it is far more natural in the review. There is no SOTA claim.
3. **Where the gain comes from (unchanged from held-out v3).** The generator (InstanceDiffusion, credited) explains most of the OPR and mIoU gain: `direct_box` is slightly higher than OCSD v3 on OPR, mIoU and count. The OCSD sketch parser adds sketch fidelity on the same generator: chamfer 0.050 vs 0.060, and facing accuracy 0.82 vs 0.51. The trade-off is a few points of count and mIoU.
4. **What remains hard.** Exact counting with 8 or more objects stays near 0 for every method. Colour binding and pasted-looking objects remain the most common review failures.
5. **Cost.** OCSD v3 takes about 24 s per image on an L4, against 6.7 s for the original OCSD.
