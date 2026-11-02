# Instance-focused repair: controlled conditioning ablation

The failed semantic router spent its budget on whole-scene sampling and often returned the original errors. The next hypothesis is narrower: full-scene conditioning during local repair may let dominant classes or unrelated sketch strokes interfere with the target. This is a plausible mechanism, not an established diagnosis or novelty claim.

## New methods

`instance_repair` and `instance_repair_focus` both start from the same GLIGEN seed, choose exactly one verifier-missing object, and prepare that original-index M2 subset. Target priority is largest visible fraction of its requested box after subtracting verified object masks/detection boxes, then lowest original index. No CLIP background routing, extra retries, new seed search, or oracle evaluation data determines the target. If no eligible missing target exists, keep the original.

Both arms use the same M2 seed/cache, foreground placement, bounding-box replacement mask, protected objects, denoising seed, strength (.5), schedule (30 scene/20 object steps, K=2, one repair), and acceptance rule. Other missing objects and unmatched extra detections are not edited in this experiment. The focused arm changes a **conditioning bundle** only:

1. Text: one target phrase plus the requested background.
2. Negative text: additionally suppress other requested classes, excluding the target class and known ancestor aliases (bird for duck; boat for sailboat).
3. ControlNet: only the selected object's sketch rather than the scene sketch.

This ablation tests the bundle, not the separate causal contribution of each component. Region attention, energy guidance and identity training are disabled. Full-scene masks still protect other verified instances; all pixels outside the permitted region are pasted back exactly. Acceptance retains matched objects and rejects count/layout/relation regression, excessive global CLIP regression, or no verifier gain. The internal guard is not a guarantee about OWLv2 or human judgment.

## Comparison and budget

`instance_screen`: 12 development scenes × one seed × four methods = 48 final images:

- `gligen_layout5`: fixed five-candidate GLIGEN control.
- `grounded_repair_sparse`: previous best development repair implementation.
- `instance_repair`: one-object replacement with global conditioning.
- `instance_repair_focus`: identical replacement with focused conditioning.

Shared initial GLIGEN candidates and M2 subsets avoid repeated generation. Each method is charged its standalone work, including reused computations. Each new one-object method costs at most 280 batch-weighted UNet+ControlNet calls at the default settings, versus the 300-unit cap/GLIGEN best-of-five control. The sparse control may cost more. This proxy excludes auxiliary encoders/detectors/segmentation and is not FLOPs or equal wall time. Timing with shared caches is reconstructed, not a randomized runtime benchmark.

## Evaluation aligned with the research goal

In addition to OPR, class count error, box mIoU and relations, `joint_layout_success` is 1 only when all requested classes/instances are matched, count error is zero, **every** matched box has IoU >= .5, and all applicable relations are correct. This avoids a high average hiding one failed object. It is still a detector-based box-layout endpoint, **not shape correctness**.

The review packet now includes each original input sketch and blank ratings for presence, count, layout, shape, relations, attributes, background and realism. Do not call a bounding-box metric a silhouette metric, or AI inspection a human study. The report always requires blind review and never automatically escalates. If focused conditioning does not beat its matched global-conditioning arm on independent outcomes without unacceptable regressions, stop; do not adjust thresholds against these same 12 scenes until it wins.

These synthetic development scenes provide instance labels, boxes and masks derived from the benchmark construction. Results isolate the generation stage; they do **not** establish an end-to-end system that recovers all instances from arbitrary raw sketches and captions. A paper must separate oracle-instance generation from predicted-instance parsing evaluation.

## Run and recover

Use a fresh runtime and current GitHub `RUN_ALL.ipynb`, `TIER="instance_screen"`, blank RUN_ID. Old completed runs remain intact; code-contract changes prevent resuming them under this implementation. Forty-eight final images include multiple internal candidates. The default 20-minute generation limit is checked between scenes and excludes installation/download/evaluation. The first GPU run is complete: see [results and limits](INSTANCE_20261102_RESULTS.md). Do not repeat it or escalate to paper/full based on this screen.

After completion, use CPU-only `03_export_screen_review.ipynb` with the printed RUN_ID. It exports tables, input sketches, final images, rejected candidates and review cards in parts below 32 MiB. Send those parts. No GitHub token or inference is required for export.
