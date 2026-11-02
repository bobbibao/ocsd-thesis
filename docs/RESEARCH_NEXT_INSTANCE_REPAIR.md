# Next optimization: count-aware repair and shape conditioning

This note follows run `20261102T165132Z_e9ea4c56`. It records a concrete research direction, not a claim that a new model has already improved. No inference configuration or default notebook tier changes accompany it. The completed run must not be repeated to generate the review packet.

## Failure signals from saved outputs

The new CPU-only `tools/audit_instance_failures.py` checks manifest hashes and emits [per-scene signals](INSTANCE_20261102_FAILURE_SIGNALS.csv). It separates internal selection checks from OWLv2 outcomes and leaves human verification explicitly unset.

- `qd_3_simple_000`: the router keeps the image because every requested instance is matched, although GDINO reports class-count error 3 and OWLv2 reports 6. Thus missing-only target selection cannot handle extra instances. The counts are detector signals, not confirmed true counts.
- `qd_5_complex_000`, `qd_5_medium_000`: after repair, GDINO reports OPR 1 and class-count error 0; OWLv2 still reports count error 1 in each, with OPR .8 and 1 respectively. More retries or a looser acceptance threshold cannot resolve which detector is correct.
- The three eight-plus-object scenes still fail independent presence/count endpoints. One replacement is intrinsically limited when multiple instances are missing. Spending more per scene could fix more targets but would invalidate the current fixed-budget comparison.
- Shape is absent from the automatic joint endpoint. A detector can match a correct class/box despite a different silhouette or pose. Attribute/background failures also remain outside that endpoint.

## What the review should decide

The [Vietnamese packet](HUMAN_REVIEW_VI.md) provides originals, captions and method-anonymized finals. Prioritize scenes 6–8 for actual per-class counts and shape correspondence; all twelve remain available so any later aggregate is not just a selection of favorable cases. Duplicate images are linked, not counted as independent human opinions. Review outcomes are development annotations, never the final test set.

1. If people confirm extra instances, develop explicit **remove-extra / replace-wrong-class / add-missing** actions. Do not treat “all requested instances found” as “scene correct.” For removal, protect requested matched instances and distinguish a truly extra object from overlapping duplicate detections before editing.
2. If people confirm counts are already correct, work on duplicate/class-alias handling and independent verifier calibration, not image regeneration. Do not fit detector thresholds to these twelve scenes.
3. If class/count improves but shape does not, test target shape conditioning separately from prompt and negative-prompt changes. The existing focused arm changes all three and cannot identify which causes the benefit.

The [received user review](HUMAN_REVIEW_20261102_RESULTS.md) now confirms extra bicycles, a useful dog replacement, and major semantic/shape/appearance failures. It does not confirm the boat presence/count gain reported by the detector. These annotations prioritize count-aware editing and verifier/shape work. Adding an extra-removal operation now on disputed detector counts risks erasing valid content. The current final-only review cannot establish whether the CLIP guard rejected a useful intermediate candidate; that question needs a separately anonymized before/candidate packet if it becomes a priority.

## Stronger reference methods and fair inputs

Official repository READMEs were retrieved again for this review:

- [InstanceDiffusion](https://github.com/frank-xwang/InstanceDiffusion): per-instance phrases with points, scribbles, boxes or masks; learned UniFusion/ScaleU modules and an inference-time multi-instance mechanism. Its scribble interface uses 20 coordinate pairs (duplicates allowed), and CLI boxes are `[xmin, ymin, width, height]`, unlike our saved `[x0,y0,x1,y1]`. An adapter needs explicit coordinate conversion, stroke sampling and tests. Using its trained instance control is a stronger reference than only reranking GLIGEN outputs. This is reported functionality, not a new performance measurement here.
- [MIGC / MIGC++](https://github.com/limuloo/MIGC): multi-instance control and mask/box inputs; published inference scripts also support iterative editing. The README states training code is not open. Keep official weights and inference implementation identifiable; do not represent this as reproducing their training.

Neither baseline has been run in this environment. Their conditioning differs: report box-only, sketch/scribble and mask-assisted tracks separately. A mask-assisted method receives extra information unless every compared method has equivalent access. Supplying benchmark instance labels/locations tests oracle-conditioned generation, not arbitrary raw-sketch parsing. These two references are targeted relevant comparators, not an exhaustive 2026 SOTA survey.

## Budget and decision before another GPU run

First freeze one hypothesis, input contract, candidate budget and acceptance logic. A two-target local policy at the current K=2/object-20/repair-15 settings would cost `60 + 2*(160+60) = 500` sampling units rather than the 300-unit comparison budget: do not silently add retries. A cheaper object proposal or shared feature computation needs its own quality ablation; savings cannot be assumed.

Compare any implemented candidate to the frozen focused arm and a stronger instance-conditioned reference on new scenes, with paired seeds and matched inputs. Separate presence, exact per-class count, every-instance layout, relations, shape and appearance outcomes; also report the combined endpoint. Record total executed work and actual latency including detector overhead. A larger paper experiment is warranted only after a held-out development validation shows a useful effect without material regressions. Human review informs engineering; it cannot certify novelty, statistical superiority or a paper score.

## Delivered in this step

- Standalone offline review HTML with Vietnamese rubric, input/output comparison, zoom, duplicate grouping, uncertainty/NA, partial completion and JSON export/import.
- Reusable manifest-verified builder and offline failure audit; actual generated review packet committed for direct download.
- Integrity tests and browser checks for grouping, blank answers, 48-row export, import, persistence, navigation and responsive layout. Browser checks use localhost because the managed Chromium policy blocks `file://`; direct local-file opening on the user's machine remains subject to their browser policy. No network resources are required by the HTML.

These are research/measurement improvements. No new model-quality gain or pretrained-model test is claimed by this commit.
