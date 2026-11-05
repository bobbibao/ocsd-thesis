# Held-out v3 results (2026-11-05): OCSD v3 vs the original OCSD

**Verdict under the locked claim rule: OCSD v3 improves over the original OCSD.** On layout accuracy (mIoU), the gain is significant after Holm correction (p = 0.023). OPR points the same way (+0.13), but that test does not survive correction (Holm p = 0.070). No primary metric is significantly worse.

Protocol: [HELDOUT3_PROTOCOL.md](HELDOUT3_PROTOCOL.md). It was locked by `assets/heldout_v3/protocol.json` before any output existed. Evidence is in [heldout3_20261105/](heldout3_20261105/).

## Run audit

* Run `20261105T015848Z_8ed9c617`, at code `9ded82b`, on an NVIDIA L4 (22 GB).
* All 216 of 216 images are present: 18 scenes × 2 seeds × 6 arms.
* B_generate took 46 minutes with 0 failures. C_evaluate completed. D_report still shows "running" because the report stage writes the ZIP from inside itself. The summary files are complete.
* The InstanceDiffusion child run is `heldout3_d66d64943a2ee1d01f31`: 72 images in 2223 s.
* The configurations used match the lock:
  * `ocsd`: paper tier (steps 30, obj_steps 20, K 2, S1 = S2 = 100) with the frozen paper_v2 tuning (alpha 0.1, LoRA 0.5, energy on phrase tokens with eta 20, region attention).
  * `gligen`: beta 1.0.
  * `collage`: strength 0.2.
  * `controlnet`: scale 1.0.
  * `ocsd_v3`: silhouettes for the layout steps only, plus the SDXL-Refiner (0.35, 20 steps) at revision `5d4cfe85…`.

## Primary family: `ocsd_v3` vs `ocsd` (18 scenes, exact sign test, Holm over three)

| metric | wins / losses / ties | difference [95% CI] | p | p (Holm) |
|---|---|---|---|---|
| OPR | 12 / 3 / 3 | +0.128 [0.023, 0.235] | 0.035 | 0.070 |
| **mIoU** | **15 / 3 / 0** | **+0.195 [0.095, 0.297]** | 0.0075 | **0.023** |
| sketch chamfer (lower is better) | 11 / 7 / 0 | −0.005 [−0.010, −0.000] | 0.48 | 0.48 |

Secondary differences, `ocsd_v3` minus `ocsd`, with bootstrap CIs:

* Relation accuracy: +0.23 [0.06, 0.40].
* Count accuracy: +0.11 [−0.17, 0.36].
* Joint layout success: +0.14 [−0.11, 0.39].
* Facing accuracy: −0.07 [−0.21, 0.04].
* CLIP: −0.33 [−1.59, 0.95].

## Key secondary family: vs GLIGEN and Collage (Holm over six)

| comparison | OPR | mIoU | sketch chamfer |
|---|---|---|---|
| vs GLIGEN | +0.093, Holm p 0.37 | +0.104, Holm p 0.15 | **−0.016, 17/1, Holm p 0.0009** |
| vs Collage | −0.001, Holm p 1.0 | +0.056, Holm p 1.0 | +0.002, Holm p 1.0 |

OCSD v3 matches sketch shape significantly better than GLIGEN. It is statistically level with Collage on all three metrics. Collage has a higher CLIP score (−2.45 [−3.98, −0.94] for OCSD v3), but in the review below it is far less natural.

## All arms (means over scenes)

| arm | OPR | mIoU | chamfer | facing | count acc. | relations | time per scene (s) |
|---|---|---|---|---|---|---|---|
| **ocsd_v3** | 0.792 | 0.670 | 0.050 | 0.83 | 0.31 | 0.66 | 26.9 |
| direct_box | 0.768 | 0.679 | 0.058 | 0.55 | 0.28 | 0.64 | 27.0 |
| ocsd (original) | 0.664 | 0.475 | 0.055 | 0.90 | 0.19 | 0.43 | 7.7 |
| gligen | 0.700 | 0.565 | 0.066 | 0.50 | 0.36 | 0.55 | 7.9 |
| collage | 0.794 | 0.614 | 0.049 | 0.89 | 0.28 | 0.63 | 4.1 |
| controlnet | 0.599 | 0.457 | 0.036 | 0.98 | 0.14 | 0.37 | 2.8 |

## Blind analyst review (36 six-panel sheets)

Bobby delegated the review, so I (Claude) did it. I wrote it before opening `review_key.json` or any summary file: [analyst_review_blind.txt](heldout3_20261105/analyst_review_blind.txt), sha256 `4ddff261…0884`. A strict rule applied: any extra copy of a listed class counts as a structure failure. The unblinded numbers are in [analysis.json](heldout3_20261105/analysis.json).

| arm | structure pass /36 | naturalness (0-2) | blind "best" picks |
|---|---|---|---|
| **ocsd_v3** | **12** | 1.42 | **7** |
| direct_box | **12** | **1.61** | 6 |
| gligen | 8 | 1.28 | 4 |
| controlnet | 5 | 0.92 | 3 |
| ocsd (original) | 9 | 0.36 | 2 |
| collage | 9 | 0.53 | 1 |
| none acceptable | | | 13 |

Sign tests over scenes (secondary, uncorrected):

* **Naturalness, OCSD v3 vs original OCSD:** 16 / 0 / 2, p < 0.001.
* **Naturalness, OCSD v3 vs Collage:** 15 / 2 / 1, p = 0.002.
* Naturalness, OCSD v3 vs ControlNet: 14 / 4, p = 0.03.
* Naturalness, OCSD v3 vs GLIGEN: 7 / 5, n.s.
* Naturalness, OCSD v3 vs direct_box: 2 / 7, n.s.
* Structure: no pairwise test is significant.

## Interpretation

1. **OCSD v3 is the complete model the thesis needed.** Against the original OCSD it places objects more accurately: mIoU is significant, and OPR and relations point the same way. Its images are much more natural, because the paper_v2 collage and paste artefacts are gone. The original OCSD's naturalness was 0.36, with most images rated 0.
2. **Where the gain comes from.** Most of the OPR and mIoU gain over the original OCSD comes from the InstanceDiffusion generator: `direct_box` reaches similar OPR and mIoU. What the OCSD parser adds on the same generator is sketch fidelity. Compared with `direct_box`, chamfer is −0.008 [−0.012, −0.004] and facing accuracy is +0.28 [0.06, 0.51]. That costs a little naturalness (1.42 vs 1.61, n.s.) and CLIP (−1.07). The paper should credit InstanceDiffusion and present OCSD v3's own contribution as sketch parsing plus the layout-only release with refinement.
3. **What remains hard.** Scenes with eight or more objects fail for every arm. 13 of 36 sheets had no acceptable image, mostly in the 8+ cells: missing or extra animals, and colour binding. Count accuracy stays low for all arms (≤ 0.36), so OCSD v3 does not solve exact counting.
4. **Cost.** OCSD v3 takes about 27 s per scene on an L4, against 7.7 s for the original OCSD and about 4 s for Collage.
5. **No SOTA claim.** OCSD v3 is level with Collage on the automatic metrics and is not significantly better than GLIGEN on OPR or mIoU.
