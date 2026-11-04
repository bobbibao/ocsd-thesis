# Held-out comparison results (run `20261104T135658Z_41d72d97`, 2026-11-04)

Protocol locked before generation in main `f64bcea`: [HELDOUT_PROTOCOL.md](HELDOUT_PROTOCOL.md).
18 held-out QuickDraw scenes (3, 5 or 8+ objects) x seeds 0/1 x 4 arms = 144 images. One blind reviewer (Bobby).
InstanceDiffusion (Wang et al., CVPR 2024) is an **external baseline**, not OCSD's method.

## 1. Technical audit

| Check | Result |
|---|---|
| Images | 144/144 (36 per arm), 0 failed, 0 skipped |
| Stages | A data, B generate (8.6 min for ControlNet + OCSD), C evaluate (1.3 min) OK; D report was still marked running when the ZIP was written (the ZIP is written inside D) |
| Direct images | imported with SHA256 checks into the legacy outputs; the direct child's own run.json/checksums were not in the results ZIP (they stay on Drive under `heldout_v1/direct`) |
| Wall time per image (recorded) | direct_box 27.2 s, direct_scribble 27.1 s, ControlNet 2.8 s, OCSD soft-matte 5.1 s |
| Sampling work per image (mean) | direct 315 UNet calls; ControlNet 120; OCSD soft-matte 213 (different architectures, not equal FLOPs) |

## 2. Primary endpoint: blind human review

Structure success = count, colour and shape all marked pass. The reviewer judged shape strictly, including
pose and facing direction from the sketch. "unclear" counts as not success (11 of 576 fields).

| arm | success (images) | per-scene rate [95% CI] | shape fails | count fails | colour fails | naturalness mean (0-2) | picked most natural with right structure |
|---|---|---|---|---|---|---|---|
| direct_box | 8/36 | 0.22 [0.08, 0.36] | 28 | 9 | 6 | 1.31 | 5 |
| direct_scribble (post-hoc) | 7/36 | 0.19 [0.06, 0.36] | 29 | 4 | 5 | 0.89 | 4 |
| ControlNet | 1/36 | 0.03 [0.00, 0.08] | 35 | 21 | 21 | 0.44 | 1 |
| OCSD soft-matte | 3/36 | 0.08 [0.00, 0.17] | 33 | 14 | 24 | 1.00 | 2 |

No image was picked in 24 of 36 sheets.

**Primary tests** (exact two-sided sign test over scenes, Holm over two tests):

| comparison | scene wins / losses / ties | rate diff [95% CI] | p | p (Holm) |
|---|---|---|---|---|
| direct_box vs ControlNet | 7 / 1 / 10 | +0.19 [0.06, 0.36] | 0.070 | 0.14 |
| direct_box vs OCSD soft-matte | 6 / 1 / 11 | +0.14 [0.00, 0.28] | 0.125 | 0.14 |

**Neither primary test is significant.** direct_box is ahead on every descriptive human measure, but 18 scenes with
many ties do not give enough power; this is direction, not evidence.

Exploratory (not corrected): direct_scribble vs direct_box 2/3/13 wins/losses/ties (no difference); OCSD soft-matte vs
ControlNet 3/1/14. By object count (successes out of 12): 3 objects box 3, scribble 4, ControlNet 1, OCSD 2;
5 objects 4, 2, 0, 1; 8+ objects 1, 1, 0, 0.

Descriptive, not pre-registered: count and colour both pass in 26/36 (box), 29/36 (scribble), 6/36 (ControlNet),
8/36 (OCSD). **Shape and pose are the bottleneck for every method**: 28-35 shape fails per arm.

## 3. Secondary: automatic metrics (OWLv2/CLIP)

Full tables: [summary.md](heldout_20261104/summary.md). Both direct arms beat ControlNet and OCSD soft-matte on count
accuracy, OPR, mIoU, relation accuracy and joint layout success, with paired CIs that exclude 0 (except count
accuracy vs OCSD). Example: OPR 0.81 (box) vs 0.55 (ControlNet) vs 0.64 (OCSD). The detector cannot see pose, so it
rates the direct arms much higher than the human shape judgement does.

## 4. What this supports

* Supported: on these 18 held-out scenes, external direct instance conditioning (boxes + phrases) gets object
  count, colour binding and layout right far more often than ControlNet scribble or OCSD soft-matte, by both the
  reviewer's count/colour marks and the detector metrics.
* Not supported: any claim that one method is significantly better on the primary endpoint, or "best". Holm p = 0.14.
* Main open problem: following the sketch's **shape/pose** (facing direction, upright vs lying). Boxes and 20 ink
  points do not carry pose; no arm solves it.
* OCSD's own routes (soft-matte) are not better than the external baseline here; consistent with earlier negative results.

## Limits

One reviewer, who refined the shape criterion (pose) while reviewing; 18 scenes; QuickDraw only; different steps and
architectures per arm; InstanceDiffusion is external and uses instance boxes/phrases from the scene file, not raw
sketch parsing.

## Files

[heldout_20261104/](heldout_20261104/): filled review, review key, human_review_analysis.json, automatic summary and
tables, status. Analysis script: [tools/analyze_heldout_review.py](../tools/analyze_heldout_review.py).
