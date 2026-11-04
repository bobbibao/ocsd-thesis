# Direct-conditioning development study: LOCKED protocol (stage 1)

**Locked 2026-11-04, before any generated image for these scenes**, in the commit that adds this file
(approved by the user on 2026-11-04). Machine-readable version: [`assets/direct_dev_v1/protocol.json`](../assets/direct_dev_v1/protocol.json)
(SHA256 `5c8395e6…a741`); scenes: [`assets/direct_dev_v1/scenes.json`](../assets/direct_dev_v1/scenes.json)
(SHA256 `a4b1cfeb…469c`), [sketch preview](../assets/direct_dev_v1/sketches_with_boxes.png).
The run's contract hashes both files, so any later edit produces a different run ID.

## Question

After the [canary](DIRECT_CANARY_20261103_RESULTS.md), is direct instance conditioning (external
InstanceDiffusion, official core) plus our region-aware MIS merge worth a held-out comparison
against ControlNet and OCSD soft-matte, judged on count, shape, layout, colour binding and
naturalness? Stage 1 cannot answer "which method is best"; stage 2 can, within its limits.

## Stage 1: development (InstanceDiffusion arms only)

* **Scenes (8, three objects each):** real QuickDraw strokes, first recognized records at line index
  >= 4000 (none overlap the canary or the usual downloader; wide objects: first record with
  width/height >= 2.2). Built by [`tools/build_direct_dev_scenes.py`](../tools/build_direct_dev_scenes.py).
  Layout, captions and phrases were written before output. The bird placement box was moved once
  after viewing the rendered sketches (no generated images existed) so the overlap scene overlaps.

  | Scene | Group | Caption |
  |---|---|---|
  | dev1_cars_house | same class, two colours | a red car on the left and a white car on the right, parked in front of a house |
  | dev2_cats_tree | same class, two colours | a black cat on the left and a white cat in the middle, sitting on a lawn next to a tree |
  | dev3_umbrellas_palm | same class, two colours | a red umbrella and a blue umbrella on a sandy beach, with a palm tree on the right |
  | dev4_apples_cup | same class, two colours | a green apple and a red apple next to a white cup on a wooden table |
  | dev5_dog_couch | overlapping boxes | a brown dog sitting on a gray couch next to a floor lamp in a living room |
  | dev6_bird_fence | overlapping boxes | a small bird perched on a wooden fence in a garden, with a flower in front of the fence |
  | dev7_train_tree_sun | wide box (2.3:1) | a long train passing by a tree under the sun |
  | dev8_canoe_mountain | wide box (3.6:1) | a red canoe on a calm lake, with a mountain behind it and a cloud in the sky |

* **Seeds:** 0 and 1; all three arms of a scene/seed share the same initial noise.
* **Arms:**
  * `direct_box`: boxes + phrases, official mean merge
  * `direct_scribble`: + 20 ink points per instance, official mean merge
  * `direct_scribble_region`: as `direct_scribble`, region merge (OCSD modification, feather 2 latent px)
* **Fixed sampler:** 50 PLMS steps, MIS 0.36, alpha 0.8, CFG 7.5, canary negative prompt, UNet FP16
  autocast, VAE FP32, no refiner, no reranking, no retries with other settings.
* **Budget:** 48 images × 216 UNet sample calls = 10 368 calls. The canary's L4 took about 20 s per
  216-call image, so roughly 16 min of generation plus model loading. Child-process cap 2400 s,
  excluding library install; completed images are kept and resumed on the next Run all.
* **Review:** 16 PNG sheets (sketch + images A/B/C in a fixed seeded random order per sheet),
  `HOW_TO_REVIEW.txt` and `REVIEW_TEMPLATE.txt` in a small review ZIP without the key. Per image:
  count, colour, shape, position/relation pass/fail/unclear; naturalness 0/1/2/unclear; per sheet the
  most natural image with correct structure. Automatic OWLv2/CLIP scores, if computed later, are
  secondary only. Merge diagnostics (trajectory/fused latent std, background gap) are logged per image.

### Decision rules (fixed now; unit = 16 scene/seed pairs)

1. Structure failure = count, colour or shape marked fail for an image.
2. `direct_scribble_region` replaces `direct_scribble` for stage 2 only if it is picked as most natural
   with correct structure in at least 3 more pairs than `direct_scribble`, and has at most 1 more
   structure failure.
3. The scribble arm kept by rule 2 goes to stage 2 instead of `direct_box` only if it has no more
   structure failures than `direct_box` and is picked in at least 3 more pairs; otherwise stage 2 uses
   `direct_box`.
4. If every arm has a structure failure in more than 4 of 16 pairs, stop the direct line and report it
   as a negative result.
5. No tuning, reranking, extra seeds or reruns on these scenes to change the outcome; failed images
   stay in the record.

## Stage 2: held-out comparison (separate run, separate lock)

* 16 to 24 new scenes, locked before generation, never used in stage 1 or for tuning.
* Arms: the direct arm(s) selected by the rules above, **ControlNet scribble** and **OCSD
  instance-repair with learned soft matte**, on the same scenes and seeds.
* Primary: blind human review as in stage 1. Secondary: OWLv2 count/OPR/mIoU/relations, CLIP.
* Analysis: paired by scene; bootstrap confidence intervals over scenes; all arms and negative
  results reported; full compute including model loads, preparation and failed attempts, reported
  per architecture and not called equal FLOPs.

## Limits that hold regardless of outcome

* InstanceDiffusion is an external method (Wang et al., CVPR 2024); its results are a baseline, not
  OCSD's architecture. Only the region merge is our change, and stage 1 is its ablation
  (`direct_scribble` vs `direct_scribble_region`).
* Input is instance-separated sketches with phrases and boxes; 20 ink points lose detail. This is not
  end-to-end raw-sketch parsing. No oracle masks.
* Eight development scenes and one reviewer give direction, not statistical evidence.
* QuickDraw drawings may be in pretraining data; no training-independence claim.
