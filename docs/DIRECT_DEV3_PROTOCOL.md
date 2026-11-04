# Dev v3: naturalness of the sketch-silhouette model (locked 2026-11-04)

**Why.** On held-out v2 the OCSD sketch-silhouette parser + InstanceDiffusion beat the box arm on facing direction
(14/0/2 scenes, Holm p = 1e-4) and on sketch chamfer (18/18 scenes), with better OPR, mIoU and relations
([results](HELDOUT2_20261104_RESULTS.md)). Its one weak point was naturalness: 1.28 vs 1.53 (0-2 scale), with 7 images
that looked cut out. This study picks how to fix that before the final comparison against the original OCSD.

**Design.** Same 8 dev scenes and seeds 0/1 as dev v1/v2 (not held-out). Four arms, a 2x2 design:

| arm | silhouette masks used | SDXL-Refiner cascade |
|---|---|---|
| `sil_full` | all 50 steps (= held-out v2 winner) | no |
| `sil_layout` | only the 18 MIS layout steps, released after the merge | no |
| `sil_full_refine` | as `sil_full` | yes, on the same base image |
| `sil_layout_refine` | as `sil_layout` | yes, on the same base image |

* Layout-only release: the n+1 per-instance/global trajectories (first int(50 x 0.36) = 18 PLMS steps) see the
  silhouette masks and polygons; after they are merged the joint trajectory keeps boxes, centre points and phrases
  (UniFusion `test_drop_masks`). Fixed a priori, not tuned.
* Refiner: the cascade that ships with upstream InstanceDiffusion `inference.py` (SDXL-Refiner 1.0, fp16,
  strength 0.35, 20 steps, caption prompt, no negative), with a seeded generator. Revision resolved once and recorded;
  held-out v3 pins it. Credit: Wang et al., InstanceDiffusion, CVPR 2024; Podell et al., SDXL, 2023.
* Automatic evidence: SAM sketch fidelity (`ocsd/sketch_fidelity.py`, unchanged from held-out v2).
* Review: Bobby delegated review. Claude reviews the blind ABCD sheets before opening `review_key.json`
  and commits the blind review first.

**Decision rule** (also in `assets/direct_dev_v3/protocol.json`):

1. Admissible: structure failures <= `sil_full` + 1, mean chamfer <= `sil_full` + 0.010, and pooled facing accuracy >= `sil_full` - 0.10.
2. Keep the admissible arm with the highest mean naturalness. On a tie (< 0.10 apart), keep the one with fewer structure failures, then the earlier (simpler) arm.
3. The kept arm becomes the frozen **OCSD v3** configuration. A new locked held-out v3 then compares it against the original OCSD (paper_v2 tuned), InstanceDiffusion box, ControlNet and GLIGEN.
4. No tuning on these scenes, no reranking, extra seeds or reruns.

**Run.** `RUN_ALL.ipynb`, `RUN_MODE="naturalness_dev"`, L4/A100, at most 3000 s. 32 InstanceDiffusion images (6912
UNet calls), 32 refiner passes, SAM. Output: Drive `results/direct_dev3`.
