# Development screen review

Run: `20261028T225607Z_4480ea4d`. Code: `c5ee889`.
Source: executed `notebooks/RUN_ALL.ipynb` saved to main in commit `212f011`.
This review uses its embedded report and stage logs, not independently recomputed metrics.
The results upload failed authentication; the per-image CSVs and repair traces have not yet been inspected.

All 84 planned images (12 development scenes, seven methods, one seed) were generated with zero reported failures. Evaluation and reporting completed. GPU: A100 40 GB. Generation stage: 8.2 minutes; evaluation: 0.6 minutes; report: 0.2 minutes. These are stage timings, not a billed runtime total.

| Method | OPR ↑ | mIoU ↑ | Count error ↓ | RA ↑ | CLIP ↑ | Mean standalone seconds ↓ |
|---|---:|---:|---:|---:|---:|---:|
| GLIGEN | .8567 | .6710 | 1.9167 | .6516 | 27.9138 | 3.0424 |
| GLIGEN best-of-2 | .8856 | .6901 | 1.6667 | .7120 | 28.1770 | 3.3360 |
| Grounded repair | .9023 | .7034 | 1.6667 | .7638 | 28.1012 | 10.6741 |
| Grounded repair without M2 | .8567 | .6684 | 1.8333 | .6516 | 27.8877 | 2.3497 |
| OCSD-v2 | .8389 | .4454 | 1.5833 | .5643 | 29.8094 | 19.0157 |
| Collage | .6884 | .5132 | 2.3333 | .5047 | 30.0786 | 16.8482 |
| ControlNet | .6350 | .5028 | 2.0833 | .3284 | 28.5804 | 2.3084 |

Grounded repair has the highest mean OPR, mIoU and relation accuracy in this screen. Relative to best-of-2 it gains 1.67 percentage points OPR and .0133 mIoU, ties count error, slightly reduces CLIP, and takes about 3.20 times as long. Relative to GLIGEN it gains 4.56 percentage points OPR. M2's ablation is promising, but paired per-scene evidence is still needed. OCSD-v2 has lower count error; the new method does not dominate all metrics. CLIP is part of the guard and is not independent evidence of visual quality. Timings include caching/loading effects and do not establish equal-compute superiority.

The report's decision is `review_for_heldout_validation`, not approval for a full run. Twelve scenes and one seed do not establish statistical significance, general superiority, novelty, or paper quality. Marginal confidence intervals cannot replace paired tests.

Next steps, before more GPU work:

1. Recover `tables/screen_paired.csv`, `tables/screen_means.csv`, per-image metric CSVs, manifest, plan and logs from this existing run using CPU. Check pairing, corrected tests, scene-level deltas and run completeness.
2. Inspect generated images and repair acceptance traces for preserved objects, style discontinuities, and concentrated gains/losses. Keep visual review blind to method labels where feasible.
3. Only after that review, freeze a held-out protocol with a compute-matched baseline, multiple seeds, independent evaluation and a small staged budget. Keep these 12 scenes in development; do not reuse them as final test scenes. No additional GPU run is currently justified solely by these aggregate means.

## Uploaded ZIP audit (2026-10-29)

The user supplied `ocsd_screen_results.zip`. All 17 present manifest-listed result artifacts match their SHA-256 checksums. There are 84 unique method/scene/seed rows, 12 per method. All 25 paired comparisons were recomputed from the per-image CSVs with the repository's reporting functions; differences, bootstrap intervals, raw p-values and Holm-adjusted p-values match the supplied table. This verifies report arithmetic, not the detector outputs or image quality.

Against GLIGEN, OPR improves on 3 scenes and ties on 9. Against best-of-2 it improves only on `qd_5_complex_000` (+.20), tying on the other 11; the mean gain is .01667. mIoU against best-of-2 improves on 3 scenes, ties on 8 and declines on `qd_8p_complex_000` (-.02407). Count error improves on one scene, worsens on one, and ties on ten.

Against best-of-2, Holm-adjusted p is 1.0 for OPR and .5 for mIoU. The mIoU bootstrap difference interval is [-.00523, .04501]. Against plain GLIGEN, adjusted p is 1.0 for OPR and .375 for mIoU. No superiority over either GLIGEN comparator is established. The mIoU advantage over OCSD-v2 has adjusted p=.00732 within the report's per-metric family of five comparisons; this is exploratory development evidence, not a held-out claim or correction across all 25 tests. Failure to reject does not establish equivalence.

M2 is charged on five scenes: 95.39 of 128.09 total standalone seconds (74.5%). Three of these scenes improve OPR versus plain GLIGEN; `qd_5_medium_000` and `qd_8p_simple_000` do not. This identifies full-scene M2 preparation as a cost target, but missing-object-only M2 would change behavior and has not been validated. No algorithm change is justified as a proven improvement yet.

The ZIP contains no generated images or per-output repair JSON traces. Visual preservation, accepted/rejected repair reasons and compute counters remain unverified. Use `03_export_screen_review.ipynb` on CPU to export existing images and metadata without credentials or inference. Do not scale to paper/full before that review.
