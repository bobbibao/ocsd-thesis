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

## Visual and repair-trace audit (2026-10-29)

The supplied `ocsd_visual_review.zip` contains all 48 PNGs and 48 metadata files for GLIGEN, best-of-2, no-M2 repair and full repair. All 96 match the run manifest hashes. Pixel comparisons and metadata for all 12 full-repair outputs are recorded in `SCREEN_20261028_REPAIR_AUDIT.csv`. The six attempted-repair scenes were inspected side by side across all four methods. This was an unblinded qualitative engineering inspection, not a human evaluation study; original sketches and requested object phrases were not included.

Six scenes pass the initial verifier; their final images are pixel-identical to GLIGEN. Six scenes attempt repair: four are accepted and two rejected. Both rejected outputs are also pixel-identical to GLIGEN, confirming fallback for this run. The changed pixel fractions of accepted outputs are .560%, 3.593%, 5.841% and 8.084%. This supports localized changes; without the saved edit masks it does not independently verify every protected object region.

- `qd_3_simple_000`: accepted on a tiny verifier score increase (.000121), with no change to verifier OPR or count error. The train/bicycle scene looks essentially unchanged at contact-sheet scale. It contributes no OPR improvement.
- `qd_5_complex_000`: a small boat appears in the lower-left-middle region. It overlaps the pre-existing blue umbrella-like shape, so the object-presence gain should not be described as demonstrated aesthetic improvement. This is the sole OPR win over best-of-2.
- `qd_5_medium_000`: the attempted repair is rejected (`count_or_layout_regressed`); final horse-scene image equals GLIGEN. M2 cost: 13.60 seconds. The rejected candidate itself is not saved, so its visual failure cannot be inspected.
- `qd_8p_complex_000`: local additions include a dog-like figure near the tree and an orange object. OPR improves over GLIGEN but only ties best-of-2; independent mIoU is lower than best-of-2. The repaired and best-of-2 images have visibly different composition; a metric win cannot settle overall preference.
- `qd_8p_medium_000`: vehicle regions are visibly changed/added while much of the street scene remains intact. The internal verifier reaches OPR=1, but the independent evaluation does not establish all requested objects are correct. Do not substitute internal verifier success for evaluation.
- `qd_8p_simple_000`: repair is rejected (`lost_object`) despite a small aggregate verifier score gain. The final bird/chair scene equals GLIGEN. M2 cost: 21.83 seconds. This supports retaining the explicit matched-object guard; the rejected candidate is unavailable.

### Decision and next implementation target

Do not run paper/full or claim superiority. The protected-repair fallback works on this run, but the benefit over best-of-2 is narrow and visual quality is unresolved. Keep this screen as development evidence.

The next defensible optimization experiment is **M2 generation only for verifier-missing objects**, preserving original object indices/seeds and testing placement and overlaps explicitly. Current M2 prepares 35 objects across the five affected scenes although the initial verifier marks only 12 missing: 70 versus a prospective 24 candidates at K=2. This is a theoretical candidate-count reduction of 65.7%, not a measured runtime reduction or a quality guarantee. Full-scene compositing can affect anchors in overlapping regions, so simply filtering a scene would change both RNG indexing and appearance. It must remain a separately named experimental variant with complete cache signatures and comparisons against the current method.

Before another GPU experiment, add diagnostics for missing object IDs, matched boxes, edit/protected masks, and accepted/rejected candidate images, plus a reason breakdown. Those artifacts will make failures inspectable without paying to recreate them. Avoid tuning acceptance thresholds to these 12 scenes. The rejected attempts account for 35.43 seconds (37.1%) of M2 time, but cannot safely be predicted and skipped from this sample alone. No new model inference was performed for this audit.
