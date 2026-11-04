# Direct-conditioning development study: protocol DRAFT (not locked, nothing run)

Status 2026-11-04: draft for the user's decision. No scene, image or result for this protocol exists.
It becomes binding only after scenes, captions and decision rules are committed **before** any GPU
output, with the commit SHA and date recorded here.

## Question

After the [canary](DIRECT_CANARY_20261103_RESULTS.md), is direct instance conditioning (external
InstanceDiffusion, official core) plus our region-aware MIS merge worth a held-out comparison
against ControlNet and OCSD soft-matte, measured on count, shape, layout, colour binding and
naturalness? This stage cannot answer "which method is best"; stage 2 can, within its limits.

## Stage 1: development (InstanceDiffusion arms only)

* **Scenes:** 8 new scenes, built like the canary from real QuickDraw strokes (first recognized
  records at line index >= 4000, so none overlap the canary or the usual downloader), with layout and
  captions written before output. Fixed mix: 4 scenes with two same-class instances of different
  colours, 2 with overlapping boxes, 2 with a box aspect ratio beyond 2:1; 2 to 4 objects each.
  None of the canary objects/colours are reused as-is.
* **Seeds:** 0 and 1, the same initial noise in every arm of a scene/seed pair.
* **Arms:**
  * D1 `direct_box`, official mean merge
  * D2 `direct_scribble`, official mean merge
  * D3 `direct_scribble`, region merge (OCSD modification; feather 2 latent px, fixed)
* **Fixed sampler:** 50 PLMS steps, MIS 0.36, alpha 0.8, CFG 7.5, same negative prompt, no refiner,
  no reranking, no retries with other settings.
* **Size and cost:** 48 images. UNet sample calls per image = 2 × (50 + 19n + 1) for n objects
  (178, 216, 254 for n = 2, 3, 4), so 48 images need 8 544 to 12 192 calls. On the canary's L4,
  one 216-call image took about 20 s, so the generation estimate is about 13 to 19 min plus
  2 to 3 min of model loading. Hard child cap: 40 min; completed images are kept and resumed.
* **Review:** one PNG per scene/seed with the sketch and the three images in a seeded random A/B/C
  order, a text template and instructions; the key is kept separately. Per image: count, shape,
  position, relation, colour pass/fail/unclear; naturalness 0/1/2/unclear; per sheet: most natural
  image that still has the right structure. OWLv2 count/OPR/mIoU are logged as secondary signals only.
* **Decision rules (fixed in advance, 16 scene/seed pairs):**
  * D3 replaces D2 for stage 2 only if it is preferred in at least 3 more pairs than D2 is preferred,
    and it has no more than 1 extra structure failure (count, colour or shape) across the 16 pairs.
  * Box vs scribble: keep scribble only if D2 or D3 is not worse than D1 on structure failures and
    the preference rule above holds against D1; otherwise stage 2 uses D1.
  * If all three arms fail structure in more than 4/16 pairs, stop the direct line and report it.

## Stage 2: held-out comparison (separate run, separate lock)

* 16 to 24 new scenes, locked before generation, never used in stage 1 or for any tuning.
* Arms: at most two direct arms selected in stage 1, **ControlNet scribble** and **OCSD
  instance-repair with learned soft matte**, all on the same scenes and seeds.
* Primary: blind human review as in stage 1. Secondary: OWLv2 count/OPR/mIoU/relations, CLIP.
* Analysis: paired by scene; bootstrap confidence intervals over scenes; all arms and negative
  results reported; full compute including model loads, preparation and failed attempts.
* Costs are reported per architecture (UNet calls, wall time, peak memory) and are not called equal
  FLOPs.

## Limits that hold regardless of outcome

* InstanceDiffusion is an external method (Wang et al., CVPR 2024); its results are a baseline, not
  OCSD's architecture. Only the region merge is our change and needs its own ablation (D2 vs D3).
* Input is still instance-separated sketches with phrases and boxes; 20 ink points lose detail. This
  is not end-to-end raw-sketch parsing. No oracle masks.
* QuickDraw drawings may be in pretraining data; no training-independence claim.
