# Direct development study results (run `dev_c80aed018746b51a0cee`, 2026-11-04)

Stage 1 of the [locked protocol](DIRECT_DEV_PROTOCOL.md). Development scenes, one reviewer (the user),
blind A/B/C sheets. Direction only, not statistical evidence and not a comparison with ControlNet or OCSD.

## Technical audit

* 137/137 ZIP checksums match; status `complete`, 48/48 images, one attempt, no failures.
* Contract: OCSD commit `d89f9c9` (the lock commit), protocol and scene SHA256 equal the locked files;
  model provenance unchanged from the CPU PASS. NVIDIA L4.
* Work: 216 UNet sample calls per image, 10 368 total = plan. Mean wall time per image 19.1 s (box),
  18.8 s (scribble), 18.9 s (scribble + region); peak CUDA 5.95 GiB for every image; attempt wall
  time 1042 s including model loading (excludes pip install and idle time).
* Per-image compute and merge diagnostics: [compute_and_merge_diagnostics.json](direct_dev_20261104/compute_and_merge_diagnostics.json).
  [All 48 images by arm](direct_dev_20261104/all_48_by_arm.jpg) (columns: sketch, box, scribble, scribble + region).

## User review (unblinded after receipt)

The user's table is kept as sent: [user_review_table.png](direct_dev_20261104/user_review_table.png).
Picks were mapped to arms with `review_key.json` only after the review arrived.

| | direct_box | direct_scribble | direct_scribble_region |
|---|---:|---:|---:|
| Picked most natural with correct structure (of 16) | **8** | 5 | 3 |
| Structure failures noted (count/colour/shape) | 4 | 1 | 0 |
| Naturalness 0 noted | 0 | 0 | **4** |

The user wrote notes rather than filling every field, so failures were coded from the notes by the
analyst: structure = wrong colour on the right car (dev1 s0: box, scribble), black/brown cup instead of
white (dev4 s1: box), table lamp plus ceiling light instead of a floor lamp (dev5 s0: box), an extra
person in the background (dev5 s1: box). Naturalness 0 = three separate panels (dev1 s1), a flat 2D
sticker cat (dev2 s1), a tree trunk clipping through the train (dev7 s1), a cartoon cloud (dev8 s0), all
region. Other complaints (pasted-looking trees, odd wooden blocks, a 1-2-car "long train") are in the
table but not coded. Sheet dev3 s1 was "C (or A)"; A was region, which does not change any rule.

## Decision rules applied as locked

1. Region vs scribble: region picked 3 vs 5, not 3 more. **Region merge rejected.**
2. Scribble vs box: scribble has fewer structure failures (1 vs 4) but is picked 5 vs 8, not 3 more.
   **By the locked rule, stage 2 uses `direct_box`.**
3. No arm fails structure in more than 4/16 pairs, so the direct line continues.

## What this supports and refutes

* **Refuted (in this sample): the region merge improves naturalness.** It never improved on official
  scribble and produced all four naturalness-0 images. Visually the region backgrounds are flatter and
  more studio- or clip-art-like, consistent with dropping the instance trajectories' contribution to
  the background. Panels also appear with the official merge (dev1 s1 scribble), so the canary
  hypothesis that the plain mean merge causes panels is not supported either. Kept as a negative result;
  the option stays in the code, off by default, and will not be retuned on these scenes.
* **Tension to report, not resolve here:** box-only conditioning gave the most natural, photographic
  scenes but 4 structure failures; adding 20 ink points cut structure failures to 1 while looking more
  like product photos. With 16 pairs and one reviewer, neither difference is established.
* InstanceDiffusion handled two same-class objects with different colours correctly in most pairs, and
  all 48 images ran in about 19 s each on an L4 at under 6 GiB.

## Limits

Eight development scenes, two seeds, one reviewer who also chose the study, failures coded by the
analyst from free-text notes. No automatic metrics were computed. InstanceDiffusion is an external
baseline (Wang et al., CVPR 2024); only the rejected region merge was our change.
