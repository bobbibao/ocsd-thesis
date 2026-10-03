# Pre-registered tests for the power run (improvement plan item 2.4)

Written on 2026-10-04, before any image of the `E3_quickdraw_power` job was generated. The commit that adds this file
fixes the hypotheses in time; `ocsd/report.py` (`PREREG`, `prereg_tests`) runs exactly these tests and writes
`results/tables/prereg_quickdraw.{csv,md}`. Any other test reported from the power run is exploratory.

## Data

- QuickDraw-Scenes evaluation set of the `paper` tier: 72 scenes, 6 per cell of 4 object counts (1, 3, 5, 8+)
  x 3 sketch complexities, disjoint from the tuning split (`data/benchmarks/tuning_split.json`).
- Seeds 0 and 1 for every scene and every method. The unit of analysis is the scene (mean over the two seeds).
- Methods: OCSD, GLIGEN, ControlNet (base), OCSD-lite, all with the `paper` tier settings. OCSD and OCSD-lite use the
  configuration chosen by tuning phases 1–3 on the tuning split only (`results/tuning/tuned.json`), fixed before
  the power run. The baselines keep the settings of the first paper-tier run.
- Metrics as in E3: OWLv2 detector at threshold 0.30, IoU 0.10 for a preserved object.

## Primary hypothesis

H1. On the 8+ object scenes (18 scenes), OCSD preserves more objects (OPR) than GLIGEN and than ControlNet.

- Test: one-sided Wilcoxon signed-rank on scene OPR (OCSD > other), `zero_method="zsplit"`.
- Family: the two comparisons (vs GLIGEN, vs ControlNet), Holm-corrected. Significance level 0.05.
- H1 holds for a baseline when its Holm-corrected p is below 0.05.

## Secondary tests (reported whatever the outcome)

1. `new_scenes_8plus`: H1 on the 9 scenes with 8+ objects that were not in the first E3 comparison. The crowded-scene
   effect was first seen on the other 9 (E1: OCSD 64.4% vs GLIGEN 47.8% vs ControlNet 40.1% OPR), so this is the
   strictly confirmatory part of H1. One-sided, Holm over the two comparisons.
2. `all_scenes`: OCSD vs GLIGEN and vs ControlNet on all 72 scenes, OPR, OCE-class and RA; two-sided, Holm per
   metric over the two comparisons.
3. `identity_learning`: OCSD vs OCSD-lite on all 72 scenes, OPR, OCE-class and ID-Sim; two-sided (plan item 2.7).

## What is not pre-registered

The ablation (E4), the α sweep, COCO-Sketch and FID/KID are descriptive. The choice of the OCSD configuration is made
by the tuning rule (highest OPR + mIoU + RA with CLIP within 1 point of the reference), not by looking at test scenes.
