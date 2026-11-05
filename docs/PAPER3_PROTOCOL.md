# Paper benchmark v3: confirmation of OCSD v3 on the paper_v2 scenes (protocol)

Status: LOCKED 2026-11-05 by `assets/paper_v3/protocol.json` (written by `tools/lock_paper_v3.py`), before any output.

## Why

Held-out v3 ([results](HELDOUT3_20261105_RESULTS.md)) showed that the frozen OCSD v3 beats the original OCSD on 18 new scenes. This run checks the same claim at four times the size, on the 72 scenes behind every paper_v2 number. The paper's main table can then use one benchmark for all methods.

## What is fixed

* **Model.** OCSD v3 exactly as frozen in `ocsd/ocsd_v3.py`. No setting is chosen on these scenes.
* **Arms, analysis and claim rule.** The same as held-out v3:
  * Arms: `ocsd_v3`, `direct_box`, `ocsd`, `gligen`, `collage`, `controlnet`.
  * Legacy methods use the paper tier with the frozen paper_v2 tuning.
* **Primary family.** `ocsd_v3` vs `ocsd` on OPR, mIoU and sketch chamfer. Each uses an exact sign test over scenes, with Holm correction over the three.
* **Key secondary family.** vs GLIGEN and vs Collage, with Holm correction over six.
* **Scenes.** The 72 paper_v2 evaluation scenes: QuickDraw-Scenes numbers 003–008 in each of the 12 cells, 18 per count bin 1/3/5/8+. Seeds are 0 and 1.
  * The scenes are copied at run time from the paper benchmark on Drive (`KLTN_OCSD/data/benchmarks/quickdraw`, built 2026-09-29).
  * Each scene must match `assets/paper_v3/expected_scenes.json` (object count, count bin and complexity, taken from `results/paper_v2/per_image`).
  * The copied files' hashes become the run's manifest.
  * Rebuilding the scenes with today's generator does not reproduce 13 of the 8+ scenes, so the Drive copy is the source of truth.
* **Blind analyst review.** The 24 sheets for the scenes numbered 003 (12 cells × 2 seeds). The review is written before unblinding.

## Limits

* These scenes are not new. They were the paper_v2 evaluation set. paper_v2 tuning used the separate tuning split (scene numbers 000–002). Held-out v3 remains the out-of-sample test, and the two runs are reported separately, never pooled.
* COCO-Sketch is not included.
* InstanceDiffusion and the SDXL refiner are external components, credited as such.

## Run

`RUN_ALL.ipynb` with `RUN_MODE="paper3"` on an L4 or A100:

* InstanceDiffusion-core arms: 288 images, capped at 4 h per attempt.
* Legacy methods: 576 images, up to 6 h.

Both parts resume. Output goes to Drive `paper_v3`, and the run downloads `ocsd_paper3_results.zip` and `REVIEW_paper3.zip`.
