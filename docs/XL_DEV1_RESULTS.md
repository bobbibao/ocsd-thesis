# OCSD-XL development round 1 (`xl_dev_v1`): no configuration passes

Run 2026-11-12 on Colab A100, code 823a546, 81 minutes. Protocol: `docs/XL_DEV_PROTOCOL.md`. 9 development scenes
(held-out v1, `_000` of each cell) × seeds 0, 1. 180 of 198 images; the M3 arm (`ocsd_xl_id`) produced blank images
(half-precision overflow) on its first three attempts and was switched off, so it has no data.

These are development numbers on 9 scenes. They select nothing and are not results to report.

## Numbers

S = (OPR + mIoU) / 2 (OWLv2, scored at 512 px). N = blind naturalness, 0 broken or pasted, 1 fake but readable,
2 natural, rated by the analyst on shuffled unlabeled sheets before the key was opened
(`results/xl_dev_v1/blind_naturalness_ratings.json`).

| Arm | OPR | mIoU | S | N | broken (of 18) | PickScore | s / image |
|---|---|---|---|---|---|---|---|
| `xl_controlnet` (baseline) | 0.484 | 0.356 | 0.420 | 0.83 | 3 | 21.58 | 6 |
| `xl_collage` (baseline) | 0.636 | 0.494 | 0.565 | 0.17 | 16 | 21.01 | 11 |
| `ocsd_xl` (centre) | 0.625 | 0.464 | 0.545 | 0.06 | 17 | 21.30 | 24 |
| `ocsd_xl_a02` | 0.629 | 0.476 | 0.553 | 0.00 | 18 | 21.11 | 23 |
| `ocsd_xl_a06` | 0.636 | 0.503 | 0.570 | 0.00 | 18 | 21.38 | 25 |
| `ocsd_xl_w04` | 0.618 | 0.388 | 0.503 | 0.67 | 8 | 22.02 | 23 |
| `ocsd_xl_noenergy` | 0.653 | 0.494 | 0.574 | 0.11 | 16 | 21.29 | 14 |
| `ocsd_xl_norepair` (ablation) | 0.612 | 0.456 | 0.534 | 0.00 | 18 | 21.42 | 11 |
| `ocsd_xl_region` (ablation: M5 only, no M2 anchoring) | 0.655 | 0.459 | 0.557 | 0.78 | 6 | 21.78 | 25 |
| `ocsd_xl_sep` (v1 sampler) | 0.441 | 0.326 | 0.384 | 0.06 | 17 | 20.78 | 13 |
| `ocsd_xl_id` (+ M3) | no data | | | | | | |

Full table with confidence intervals: Drive `KLTN_OCSD/xl_dev_v1/results/summary.md`.

## Decision by the pre-registered rule

Admissible needs N ≥ 1.0. No candidate reaches it (best candidate: `ocsd_xl_w04`, 0.67). Nothing is frozen, there is
no confirmation run, and step 5 applies: one more development round on held-out v2 scenes.

## What the images show

1. **M2 fails on SDXL.** The object previews (`m2/` in the review ZIP) are white paper cut-outs and line drawings, not
   photographs. The object prompt ends in "simple white background", most phrases are "white …", and the scribble
   ControlNet at scale 1.0 makes SDXL trace the outline. In one scene the dog is missing from the object set.
2. **Every arm that anchors to those objects inherits them**: glowing white shapes pasted on an unrelated background
   (centre, α variants, no-energy, no-repair, the v1 sampler and the Collage baseline alike). OPR / mIoU look fine
   because the detector accepts a white silhouette in the right box; the blind rating is what exposes it.
3. **M5 carries over.** Without M2 anchoring (`ocsd_xl_region`), object presence rises from 0.484 to 0.655 against the
   SDXL + ControlNet baseline (paired difference 0.170, 95% CI 0.046 to 0.285) at about the same naturalness
   (0.78 vs 0.83). A lower scene ControlNet weight (`w04`) also looks more natural than the centre.
4. **The baseline itself is weak**: N 0.83, with several painting or illustration-style outputs, so the scene negative
   prompt also needs a style term.
5. PickScore does not separate broken from natural images here (range 20.8 to 22.0), so it stays a secondary column.

## Round 2 (`xl_dev_v2`, locked 2026-11-13 before any of its images)

Changes, M2 and prompts only (M4 and M5 untouched): photo-style object prompt on a gray studio backdrop; a negative
prompt against drawings for every arm, baselines included; K = 3 candidates; softer object ControlNet in two variants
(scale 0.7 on all steps, or scale 1.0 on the first half of the steps); a guard that skips non-finite steps in M3
training. Scenes: `_000` of each cell of held-out v2. Same decision rule, with `ocsd_xl_m2b` added as a candidate.
