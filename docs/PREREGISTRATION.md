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

## Amendment 1 (2026-10-04, added after commit 0889623)

Valid only if made before any image of the power job was generated (check the image timestamps against this commit);
if the power job had already started, H1 must be reported under the original conditions above and everything below is
an addendum. H1 and its three secondary tests are unchanged; `report.PREREG` still runs them as families
`primary:H1`, `secondary:new_scenes_8plus`, `secondary:all_scenes` and `secondary:identity_learning`.

Changes to the conditions:

- The baselines no longer keep the settings of the first paper-tier run: before the power run, stage T tunes one knob
  per baseline on the tuning split with OCSD's rule (phase `baselines`: ControlNet scale, GLIGEN β, …), so the
  comparison is not between a tuned method and default baselines.
- The power job also runs OCSD-v2 (`ocsd_v2`, configured by the `v2` tuning phase on the tuning split) and the
  compute-matched best-of-N baselines (`gligen_bon`, `controlnet_bon`: up to N = 8 samples with the same Grounding DINO
  check as OCSD). The power set stays the 72 evaluation scenes (`power_per_cell = 6`); enlarging it needs another
  amendment before the run.

Added hypothesis (OCSD-v2 had not produced any image when this was written):

H2. On the 8+ object scenes, OCSD-v2 preserves more objects (OPR) than GLIGEN and than ControlNet. Same test as H1
(one-sided Wilcoxon signed-rank on scene OPR, `zero_method="zsplit"`); its own family of two comparisons,
Holm-corrected; significance level 0.05. Family `primary:H2`.

Added secondary tests (each family Holm-corrected per metric, reported whatever the outcome):

1. `new_scenes_8plus_v2`: H2 on the 8+ scenes outside the first E3 comparison; one-sided.
2. `all_scenes_v2`: OCSD-v2 vs GLIGEN and vs ControlNet on all scenes, OPR, OCE-class and RA; two-sided.
3. `compute_matched`: OCSD-v2 vs GLIGEN best-of-N and ControlNet best-of-N on all scenes, OPR; one-sided.
4. `v2_vs_thesis`: OCSD-v2 vs thesis OCSD on all scenes, OPR, OCE-class and RA; two-sided.

Also descriptive (not pre-registered): the OCSD-v2 ablation (E4v2), the control-vs-quality sweeps, the robustness
table (other evaluators) and the human studies.
