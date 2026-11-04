# Sketch-silhouette conditioning: LOCKED development protocol (stage 3)

**Locked 2026-11-04, before any image generated with silhouette conditioning**, in the commit that adds this file.
Machine-readable: [`assets/direct_dev_v2/protocol.json`](../assets/direct_dev_v2/protocol.json). Run with
`RUN_ALL.ipynb`, `RUN_MODE="silhouette_dev"` (results on Drive under `results/direct_dev2`).

## Why

The [held-out comparison](HELDOUT_20261104_RESULTS.md) showed that box conditioning (external InstanceDiffusion) gets
count and colour right far more often than ControlNet or OCSD soft-matte, but **every method fails shape/pose**
(28-35 of 36 images): animals face the wrong way, umbrellas lie down, side views become rear views. A box and 20 ink
points do not tell the generator which way an object faces. The sketch does.

## What is new (OCSD's part)

`ocsd.direct_instances.sketch_silhouette` parses each instance sketch into a filled silhouette mask:

1. **Bridge open outlines.** QuickDraw objects are open line drawings (a car body is an arc, a tree a U). Each pair of
   mutually nearest stroke ends (skeleton end pixels) is joined by a straight line, up to one ink-box diagonal apart.
2. **Close and fill.** Morphological closing with a disk of radius 0.03 x diagonal, then hole filling. Ink is kept.

The mask and a 256-point polygon (sampled as in upstream `sample_sparse_points_from_mask`, made deterministic) go
into InstanceDiffusion's own instance-mask inputs (`segs`, `polygons`), which the official checkpoint was trained with
(upstream `configs/test_mask.yaml`). Masks come **only from the input sketch**; no dataset or oracle masks.
Parameters were fixed while looking only at input sketches ([preview](direct_dev2/silhouette_parser_preview.png):
top two rows are the dev scenes, bottom rows held-out v1 inputs), never at generated images. Known parser misses:
open single-line drawings with branches (e.g. the mountain) stay thin.

InstanceDiffusion (Wang et al., CVPR 2024) remains an **external** generator. The contribution under test is the
sketch parser that turns a sketch into instance masks for it.

## Design

* Scenes: the 8 stage-1 development scenes ([DIRECT_DEV_PROTOCOL.md](DIRECT_DEV_PROTOCOL.md)), seeds 0 and 1,
  paired noise across arms. Not held-out.
* Arms: `direct_box` (reference, regenerated in this run), `direct_silhouette` (boxes + silhouette mask, = upstream
  test_mask setting), `direct_silhouette_scribble` (+ 20 ink points). All use the official mean MIS merge.
* Fixed sampler as stage 1: 50 PLMS steps, MIS 0.36, alpha 0.8, CFG 7.5, FP16 UNet autocast, FP32 VAE.
* Budget: 48 images x 216 UNet sample calls = 10 368 calls, about 17 min on L4 plus model load; cap 2400 s; resumable.
* Review: 16 blind PNG sheets (A/B/C), `HOW_TO_REVIEW.txt`, `REVIEW_TEMPLATE.txt`, same criteria as the held-out review
  (shape includes pose and facing direction).

## Decision rules (fixed now; unit = 16 scene/seed pairs)

1. Structure failure = count, colour or shape marked fail.
2. Kept silhouette arm = fewer structure failures; tie -> more picks; tie -> `direct_silhouette`.
3. It goes to a **new locked held-out comparison** (new scenes; vs `direct_box`, ControlNet, OCSD soft-matte) only if it
   has at least 3 fewer structure failures than `direct_box`, or at least 3 more picks with no more structure failures.
4. Otherwise the silhouette line stops and is reported as a negative result.
5. No parser tuning, reranking, extra seeds or reruns on these scenes to change the outcome.

## Limits

8 development scenes, one reviewer: direction only. Silhouettes from QuickDraw drawings are rough; a filled mask
also constrains size and may over-constrain thin parts (legs, handles).
