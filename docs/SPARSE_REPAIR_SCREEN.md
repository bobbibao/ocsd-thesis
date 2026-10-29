# Missing-only M2 experiment

The audited screen found 35 M2 objects generated for only 12 verifier-missing objects. `grounded_repair_sparse` prepares only those missing indices, keeping each object's original scene index in its seed formula. It has its own subset-specific cache key and records original IDs and candidate count. The original full-scene `grounded_repair` remains a control. Sparse foreground masks remain aligned with the entire scene; unmatched cache entries never masquerade as other objects in ID similarity evaluation.

This is a new experimental sampler. Removing other foregrounds changes occlusion/compositing and can change repair quality. Fewer M2 candidates does not guarantee better images or proportional speedup. No new pretrained GPU result exists yet.

## Run on Colab

Open a fresh runtime and the current `notebooks/RUN_ALL.ipynb` from GitHub main. Leave RUN_ID blank: the old completed run used a different code contract and must not be resumed under new code. The default is now `repair_screen`: the same 12 development scenes, one seed, four methods (GLIGEN, best-of-2, full repair, sparse repair), 48 outputs. It skips M3, tuning, COCO, FID and extra detectors. The old 84-output `screen` tier is still available explicitly. The 20-minute generation budget is checked between scenes; installation, downloads and evaluation are additional time, not a hard compute-unit cap.

The current notebook has cleared outputs for the new experiment. The executed old screen remains in Git history (`212f011`) and its findings in `SCREEN_20261028_REVIEW.md`; Drive results are not modified by this change.

The report compares sparse repair against all three controls using paired tests. `review_sparse_quality_and_cost` only means the sample mean OPR/mIoU/count error did not regress versus full repair and mean standalone time fell. It is not statistical non-inferiority, superiority, or permission to scale up. Inspect scene deltas and images before deciding on another run; these scenes remain development data.

## Evidence and recovery

Both repair variants save before/candidate images for every attempted edit, including rejected candidates, editable and excluded masks, missing original IDs, matched-object assignments and detector boxes. The candidate image is saved before acceptance. Final outputs still fall back to the best image on rejection. Empty edit masks are saved too. Diagnostic files have checksums in output metadata and missing/changed diagnostic files invalidate that output's resume check. No additional diffusion is used to produce these diagnostics; saving them adds I/O.

After completion, use `03_export_screen_review.ipynb` on CPU. Blank RUN_ID selects the latest run; specify the printed ID when several runs exist. It exports manifest-verified result tables, final images and diagnostic files into separate metrics and image ZIP parts below 32 MiB each, without a GitHub token. It requires a completed run and does not run inference. Attach the metrics ZIP first, then all image parts for review. If multiple downloads are blocked by the browser, download the printed paths from the Colab Files sidebar. The manifest includes diagnostics recursively.

## Offline checks

Tests cover real tiny-model subset/full candidate equality at the original seed, subset cache isolation/reuse/corruption, scene-length sparse masks, lazy missing-ID requests, rejection evidence, real sparse Runner sampling/repair and resume, missing diagnostic detection, and the 48-output report. Existing screening and reproducibility tests also pass. These establish software behavior on CPU, not pretrained quality or Colab GPU speed.
