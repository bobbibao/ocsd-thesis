# Missing-only M2 screen: observed results

Run `20261029T000432Z_6ad81320`, code `2de8ca5`, A100 40 GB. Source: user-uploaded `ocsd_review_20261029T000432Z_6ad81320_metrics_01.zip`. All 62 included manifest-listed files match SHA-256. The run manifest reports completion of 48/48 outputs. The CSVs contain 48 distinct method/scene/seed rows, 12 per method. Recomputing all 15 paired comparisons from per-image metrics reproduces the reported differences, bootstrap intervals, raw p-values and Holm-adjusted p-values. Images have not yet been received in accessible parts; image hashes in metadata alone do not verify visual quality.

| Method | OPR | mIoU | Count error | Relation accuracy | Recorded mean standalone s/image |
|---|---:|---:|---:|---:|---:|
| GLIGEN | .8567 | .670963 | 1.9167 | .6516 | 3.2772 |
| GLIGEN best-of-2 | .8856 | .690069 | 1.6667 | .7120 | 3.2324 |
| Full-scene M2 repair | .9023 | .703370 | 1.6667 | .7638 | 12.3968 |
| Missing-only M2 repair | .9023 | .704652 | 1.6667 | .7638 | 5.2407 |

## What improved

Missing-only repair generates 24 M2 candidates instead of 70 (12 versus 35 prepared objects; K=2): a measured 65.7% candidate reduction, matching the earlier prediction. Logged SD/ControlNet calls fall from 790 to 330 each and their batch-weighted sample calls from 2980 to 1140 each. GLIGEN work is unchanged at 360 calls/720 sample calls. These counters establish less sampler work in this run; they are not FLOPs or directly comparable costs between different model architectures.

Recorded M2 time totals 32.67 seconds versus 106.62 seconds. Mean standalone time falls 57.7% (2.37x observed speed ratio). Timing is not a randomized benchmark: full repair precedes sparse repair, first-use setup and diagnostic I/O affect wall time, and an extras-only scene with no M2 (`qd_3_simple_000`) alone shows a 9.09-second timing difference. Do not promise a 2.37x speedup across hardware/runs. The sampler-work reduction provides stronger evidence of efficiency than this single timing ratio.

OPR and count error are identical between sparse/full repair on all 12 scenes; relation accuracy matches on the nine applicable multi-object scenes. Sparse mIoU improves on two scenes (`qd_5_complex_000`: +.010820; `qd_8p_medium_000`: +.005284), decreases slightly on one (`qd_8p_complex_000`: -.000727), and ties on nine. Mean delta +.001281, bootstrap interval [-.000121, .003525], Holm-adjusted p=1.0. These observations are not a statistical equivalence/non-inferiority claim.

## What remains unproven

Compared with best-of-2, sparse repair still wins OPR only on one of 12 scenes (mean +1.67 percentage points), ties mean count error, and has mean mIoU +.014583 with bootstrap interval [-.004968, .047712]. Adjusted p=1.0 for both OPR and mIoU. It still takes about 1.62x the recorded mean standalone time of best-of-2. No superior quality/cost tradeoff over that strong baseline has been established. CLIP is used by the repair guard and cannot independently validate aesthetics.

Metadata shows six initial-pass scenes, four accepted repairs and two rejections for both repair methods. On `qd_8p_simple_000`, the sparse rejection reason changes from `lost_object` to `count_or_layout_regressed`; this does not imply the rejected candidate looks better. The other rejected scene is `qd_5_medium_000`. Candidate images, masks and matching boxes are needed to inspect those failures and the changed accepted outputs.

## Decision

Keep sparse repair as the preferred **development candidate for efficiency**, while retaining full repair as a control. Do not launch paper/full or claim general superiority. Inspect the already-generated image parts next, prioritizing the three changed accepted scenes and both rejected candidates. No additional GPU work is needed for that inspection. Only then freeze a separate held-out, multi-seed, compute-aware validation protocol; these same 12 scenes must stay in development.
