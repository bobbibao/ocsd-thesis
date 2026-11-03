# Clean-canvas repair: isolate old-content carryover

The completed count diagnostic fixes detector counts but the user confirms both repaired bicycles remain severely distorted and unnatural. This experiment tests two concrete paths by which old bicycle content can survive: starting local denoising from the old image and blending the decoded result back over the old image at mask boundaries. It does not assume these are the only causes. A malformed M2 proposal, segmentation/placement, sketch conditioning or latent boundary effects may dominate instead.

## Fixed 2 × 2 ablation

All arms use focused count-aware replacement, the same target, mask, seed, M2 selection/placement, prompt/control, strength .5, 15 repair steps and acceptance guard. Missing-object routes retain their prior behavior; clean processing is enabled only for count replacement.

| Method | Denoising initialization | Final blend reference |
|---|---|---|
| `instance_repair_count` | original image | original image |
| `instance_repair_count_init` | clean canvas | original image |
| `instance_repair_count_blend` | original image | clean canvas |
| `instance_repair_count_clean` | clean canvas | clean canvas |

The **clean canvas** is deterministic: fill only the permitted edit mask using OpenCV Telea inpainting (radius 3), then paste the existing selected M2 foreground through its placed mask, clipped to the editable region. This is a classical background fill, not a new learned model or an extra diffusion proposal. All pixels outside the edit mask remain identical to the original, including protected trains. No mask expansion or retry budget is introduced. A fully masked image or an empty visible foreground causes the clean repair to fall back rather than invent a target.

Outside-region latent anchors still come from the original image and the existing M2 foreground anchor remains unchanged. Consequently this tests two particular carryover paths; even the combined variant does not guarantee removal of all original-content influence through VAE receptive fields or mask boundaries. Telea may create blur or poor background texture, and a clean canvas cannot fix a malformed object proposal. Final blending against that canvas may reveal proposal artifacts. Those are outcomes to inspect, not reasons to lower acceptance thresholds.

## Diagnostic visibility

The previous export intentionally omitted the `_objects` cache, preventing examination of the selected object before repair. Instance repair now explicitly saves the following in method diagnostics, so the existing CPU exporter includes them:

- selected `proposal.png` and its `proposal_mask.png` (not all unselected M2 candidates);
- `placed_proposal.png` and `placed_proposal_mask.png`;
- existing initial image, conditioning, editable/excluded masks;
- `local_clean_background.png` and `local_clean_seed.png` for clean arms;
- `local_decoded_candidate.png` before the final soft blend, plus candidate and accepted final.

These permit separate questions: was the proposal already distorted; did placement/segmentation damage it; did denoising deform it; did final blending bring old remnants back? Diagnostics are saved for rejected candidates too. No human labels are filled automatically.

## Cost and population

`RUN_ALL.ipynb` defaults to `clean_screen`: **one previously reviewed bicycle scene × one seed × four arms = four final images**. GLIGEN initialization and M2 preparation are shared. Each arm is charged the same 280 batch-weighted UNet/ControlNet sampling units. At the default steps, with all arms taking the count route, actual shared work is 60 initial + 160 M2 + 4 × 60 repair = 460 units; this excludes evaluation and auxiliary models. No GLIGEN best-of-five rerun is needed for this mechanism comparison.

Clean-init arms use **one additional VAE encode**; clean arms also incur CPU inpainting and diagnostic I/O. `repair_canvas` logs the flags and extra encode count. These are not free wall-time improvements even though sampling counts match. Timing includes those operations. Detector count reduction and layout preservation remain necessary guards, not evidence of naturalness or shape correctness.

This is a selected development failure, not a generalization benchmark. No confidence intervals or significance tests are emitted for this one-scene diagnostic. Do not rank variants based on tiny mIoU differences or claim a paper/SOTA result. A visually useful effect must later be validated on new scenes and paired seeds with stronger instance-conditioned comparators.

## Tests and decision

CPU tests check that changing old pixels inside the permitted mask cannot change the deterministic clean seed, and that context/foreground pixels are retained correctly. Real tiny diffusion tests cover all four arms, common M2 key/mask/work, unchanged outside pixels, resume, diagnostic exports and the factorization: blend-only produces the same decoded trajectory as baseline; init-only and both produce the same decoded trajectory. Plan/report tests enforce four outputs, no inferential tables and unproven research status. Tiny networks test mechanics, not pretrained quality.

All 47 CPU tests passed (3 new clean tests and 44 existing regressions). All five notebooks validate and regenerate deterministically. No pretrained clean-screen result exists yet. If the selected proposal is already badly formed, prioritize proposal quality and instance-shape conditioning instead of continuing local cleanup sweeps. If neither initialization nor blending materially improves the visible structure, stop this hypothesis; do not increase strength, steps or retries until this same scene looks good. If it does improve, freeze the successful component and evaluate new cases before scaling.

## Colab steps

1. Open the current GitHub `RUN_ALL.ipynb` in a fresh GPU runtime.
2. Keep `TIER="clean_screen"` and `RUN_ID=""`; Run all. This is a new run, not a resume of count_screen.
3. Export the printed new run ID with `03_export_screen_review.ipynb` on CPU, as before. Send the metrics and all image ZIP parts.

The next review will include side-by-side originals/candidates and selected proposal diagnostics with explicit questions. No manual metric computation is required from the user.
