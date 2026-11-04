# Held-out v2: silhouette confirmation, LOCKED protocol (stage 4)

**Locked 2026-11-04 before any output for these scenes**, in the commit that adds this file. Machine-readable:
[`assets/heldout_v2/protocol.json`](../assets/heldout_v2/protocol.json); scenes and checksums:
[`assets/heldout_v2/manifest.json`](../assets/heldout_v2/manifest.json). Built by
[`tools/build_heldout_v2.py`](../tools/build_heldout_v2.py): same generator and cells as held-out v1 (3, 5, 8+ objects x
simple/medium/complex, 2 per cell), seed 20261105, **no QuickDraw drawing shared with held-out v1 or the dev scenes**.
Run: `RUN_ALL.ipynb`, `RUN_MODE="heldout2"` (Drive root `heldout_v2`, resumable).

## Question

Does OCSD's sketch-silhouette parser (sketch -> instance masks for the InstanceDiffusion generator) make generated
objects follow the sketch's **shape and facing direction** better than boxes alone, without losing count, colour or
layout, on scenes never used for development?

## Arms (seeds 0 and 1, paired noise for the two direct arms)

| arm | method | role |
|---|---|---|
| `direct_silhouette` | OCSD sketch parser + InstanceDiffusion masks (boxes + silhouette + polygon) | candidate (stage-3 rule) |
| `direct_box` | InstanceDiffusion boxes + phrases | primary comparator (best arm of held-out v1) |
| `controlnet` | SD 1.5 ControlNet scribble, cn_scale 1.0 | baseline |
| `instance_repair_count_matte_soft` | earlier OCSD route | baseline |

Direct arms: 50 PLMS steps, MIS 0.36, alpha 0.8, CFG 7.5, official mean merge; 22 544 UNet sample calls; cap 3600 s.
ControlNet/OCSD: held-out v1 settings. No tuning, reranking, extra seeds or reruns.

## Primary endpoint and tests (automatic, fixed now)

Sketch fidelity ([`ocsd/sketch_fidelity.py`](../ocsd/sketch_fidelity.py)): SAM ViT-H segments each instance inside its
target box; the raw sketch ink (not the parsed silhouette) is compared with the mask outline.

* `facing_acc`: share of asymmetric drawings (ink-vs-mirror distance >= 0.04 of the box diagonal) whose ink fits the
  outline better than its mirror image. Missing object (mask < 2 % of the box) = wrong.
* `sketch_chamfer`: mean ink-to-outline distance / box diagonal (lower is better; missing object = 1).

Two primary tests, `direct_silhouette` vs `direct_box`, unit = scene (seed mean), exact two-sided sign test, Holm over
the two; paired bootstrap CIs over scenes. Thresholds were fixed on dev data only.

**Claim rule.** "Follows sketch shape/pose better" only if both tests are significant after Holm, or one is and the
other points the same way. "Better overall" additionally needs OWLv2 count accuracy and the analyst structure success
not to be worse (CI of the difference reaches 0 or above).

## Secondary

Fidelity vs ControlNet and OCSD soft-matte; OWLv2 count accuracy, OPR, count error, mIoU, relations, joint layout,
CLIP for all pairs (CIs only); analyst review of 36 A-D sheets done before unblinding (the user checks hard cases only);
by-count-bin tables; cost per arm.

## Limits

InstanceDiffusion is external (Wang et al., CVPR 2024); OCSD's contribution is the parser. Chamfer partly rewards the
silhouette arm for obeying a mask parsed from the same sketch; the mirror-based facing test is the less circular
measure. SAM errors add noise to all arms. The analyst is the method author. 18 QuickDraw scenes.
