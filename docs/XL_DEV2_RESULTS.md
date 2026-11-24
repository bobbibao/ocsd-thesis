# OCSD-XL development round 2 (`xl_dev_v2`): objects fixed, anchored images still look pasted

Run 2026-11-13 on Colab A100, code f52537d, 111 minutes. 9 development scenes (held-out v2, `_000` of each cell) ×
seeds 0, 1. 216 of 234 images; the M3 arm (`ocsd_xl_id`) again produced blank images. Files: `results/xl_dev_v2/`.

Development numbers on 9 scenes; they select nothing and are not results to report.

| Arm | OPR | mIoU | S | N (blind, 0-2) | broken (of 18) | s / image |
|---|---|---|---|---|---|---|
| `xl_controlnet` (baseline) | 0.359 | 0.255 | 0.307 | 0.17 | 15 | 6 |
| `xl_collage` (baseline) | 0.783 | 0.613 | 0.698 | 0.17 | 15 | 11 |
| `xl_collage_m2b` | 0.624 | 0.489 | 0.557 | 0.06 | 17 | 11 |
| `ocsd_xl` (centre) | 0.756 | 0.578 | 0.667 | 0.22 | 14 | 24 |
| `ocsd_xl_m2b` | 0.667 | 0.521 | 0.594 | 0.28 | 13 | 24 |
| `ocsd_xl_a02` | 0.714 | 0.561 | 0.637 | 0.22 | 14 | 23 |
| `ocsd_xl_a06` | 0.708 | 0.541 | 0.625 | 0.28 | 13 | 24 |
| `ocsd_xl_w04` | 0.700 | 0.470 | 0.585 | 0.72 | 6 | 25 |
| `ocsd_xl_noenergy` | 0.709 | 0.539 | 0.624 | 0.78 | 5 | 14 |
| `ocsd_xl_sep` (v1 sampler) | 0.625 | 0.495 | 0.560 | 0.22 | 14 | 12 |
| `ocsd_xl_norepair` (ablation) | 0.703 | 0.544 | 0.623 | 0.22 | 14 | 11 |
| `ocsd_xl_region` (ablation) | 0.681 | 0.469 | 0.575 | 0.83 | 5 | 22 |
| `ocsd_xl_id` (+ M3) | no data | | | | | |

## Decision by the pre-registered rule

No candidate reaches N ≥ 1.0 (best: `ocsd_xl_noenergy` 0.78, `ocsd_xl_w04` 0.72). Nothing is frozen.

## What changed against round 1, and what the images show

1. **M2 is fixed.** The objects are photographs now. Structure scores rose (centre S 0.545 → 0.667 on different
   scenes, so only indicative) and the centre arm is far ahead of the SDXL + ControlNet baseline on object presence
   (0.756 vs 0.359).
2. **The anchored arms still look pasted.** Typical failures: a blurred coloured halo around an anchored object, an
   object floating on an unrelated background, picture-frame patches. The Collage baseline shows the same.
3. **A weaker push on the latents is what looks better.** Without the energy step N goes from 0.22 to 0.78 at nearly
   the same S (0.624); with the scene ControlNet at 0.4 it goes to 0.72; without anchoring to 0.83. The energy step at
   unit-RMS η = 0.05 together with ω = 0.8 is too strong for SDXL: it is the source of the halos.
4. **The SDXL + ControlNet baseline is poor on these scenes** (N 0.17, OPR 0.36): on multi-object scribbles it often
   returns one large object or a scene with people. Beating it says little by itself.
5. **M3**: 199 of 200 training steps had a non-finite loss (the guard added in round 2 recorded it). Cause: the float32
   text encoders ran under half-precision autocast during training. Fixed for round 3 (`EngineXL.encode_grad`).
6. M2 variant b (full ControlNet scale on the first half of the steps) is worse than variant a; the v1 sampler is
   the worst OCSD arm in both rounds. Both are dropped.

## Round 3 (`xl_dev_v3`, locked 2026-11-13 before any of its images)

Centre moved to ω = 0.6, η = 0.02. Arms cross the energy step (off / 0.02 / 0.05) with ω (0.4 / 0.6 / 0.8), plus
α 0.2 / 0.6, the two ablations, and M3 with the precision fix. Scenes: `_001` of each cell of held-out v2 (not used
before). Same rule. Honest expectation: the best arms sit near N 0.8 and the rule needs 1.0, so round 3 may also end
without a frozen configuration; in that case the SDXL line is written up as a negative result.
