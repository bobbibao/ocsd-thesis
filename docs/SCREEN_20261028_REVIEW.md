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
