# Semantic routing: prospective development experiment

**Completed outcome:** the first run did not improve the sparse-M2 comparison. Do not scale up; see [results](SEMANTIC_20261029_RESULTS.md). The protocol below is preserved for reproduction. RUN_ALL now defaults to repair_screen.

The prior sparse-M2 screen reduced sampler work but retained wrong backgrounds, colors and malformed replacements. This experiment tests a limited routing policy against stronger selection controls. It does not claim an improvement before pretrained results and independent review.

## Fixed policy (version 1)

Internal observations use Grounding DINO class/layout matching plus CLIP contrasts. Whole-image background is contrasted against the fixed QuickDraw background vocabulary. Crops at requested object boxes contrast the requested color/material against fixed alternatives. Supported colors: black, white, red, green, blue, yellow, orange, purple, pink, brown, gray, golden. Supported materials: wooden, metal, plastic, glass, stone. A contrast margin <=0 flags an unresolved issue. This is an uncalibrated heuristic, not a reliable semantic classifier. Size, fine-grained shapes, arbitrary materials and phrases are not automatically verified. Background scoring can be biased by foreground content. All constants and the routing decision are saved in metadata; none are tuned on this run's scores.

The initial image is GLIGEN at the original seed. A background flag, multiple local targets, or unresolved count/relation issues without a single local target chooses global resampling. One missing or attribute-flagged target with an acceptable background chooses local replacement if it fits the sampling-work budget. Otherwise keep the initial image. Missing class detection can indicate absence or a wrong-class object; the policy does not pretend to distinguish those cases reliably.

Global mode draws four additional GLIGEN candidates, at seed + 100003*r for r=1..4. Local mode prepares M2 only for the selected original index, preserving its seed. Its requested bounding box is included in the editable region, allowing removal of a wrong-shaped object instead of just tracing the old sketch. That target is released from mask protection; every other matched object remains protected. Local mode makes one repair attempt and does not then escalate to global generation.

Acceptance must retain previously matched object IDs, avoid worsening count error/mIoU/relations, introduce no newly flagged color/material/background issue, and keep caption CLIP within .5 of the current image. It also requires strictly fewer semantic flags or a strictly better class/layout verifier score. These guards are internal proxies and can reject useful changes or accept visually flawed ones. Rejection returns the previous image. Do not relax guards using the inspected test outcomes merely to force a win.

## Four-method comparison, 48 final outputs

`semantic_screen` uses the same 12 **development** scenes and one seed:

- `gligen_layout5`: all five GLIGEN candidates, select the highest original class/layout verifier score; no early stop.
- `gligen_semantic5`: the same five candidates, use the semantic acceptance rule above. This controls for improvements due only to semantic selection.
- `grounded_repair_sparse`: the prior missing-only M2 method, unchanged sampling policy, as a regression control.
- `semantic_repair`: route between keep/local/global as described above.

Identical GLIGEN candidates are shared in memory across the three new methods within a scene. Five images are generated once rather than three times; semantic scoring and choice remain separate. Resume does not depend on this ephemeral pool. Outputs, diagnostics, metadata and source/config signatures retain the regular versioned run behavior.

The new methods have a declared cap of 300 **batch-weighted denoiser calls**, summing GLIGEN UNet, SD UNet and ControlNet. At 30 steps, five GLIGEN samples cost 300 units. One local target with K=2, 20 M2 steps and 15 repair steps costs 280 units including the initial GLIGEN image. Local routing falls back to global if its planned work exceeds the cap. Runner checks the actual charged work before marking an image complete. The legacy sparse control is not subject to this cap and may cost more.

This cap is a compute **proxy**, not equal FLOPs, equal latency, equal memory, or equal compute units: models differ and auxiliary CLIP/GDINO/SAM, VAE and I/O are excluded. `sampling_work.csv` reports the charged work. Cached M2 work and shared GLIGEN work are charged to each method; actual executed counters and reused work are recorded separately. Standalone times reconstruct shared generation durations and cached M2 preparation, so they are not independent wall-clock benchmarks. Do not present a quality win as equal-cost superiority without a separate calibrated timing experiment.

## Independent assessment and stop rule

OWLv2 remains the independent class/layout evaluator. Neither internal CLIP contrasts nor the reported CLIP score establish attribute/background success. The report always says `requires_blind_review`, never auto-escalates to paper/full. It creates `review_blind/cards/*.png` with shuffled A-D labels, original prompts and a blank ratings CSV for object presence, attributes, background and realism (1–5). Complete ratings before opening `review_key.json`; ideally obtain independent reviewers. No scores are invented or filled automatically. The packet enables independent assessment; it is not itself a completed human study.

Before another GPU run, inspect errors and per-scene deltas against **both** best-of-five controls. If routed repair does not improve independently judged content at an acceptable cost, do not expand or claim it beats the controls. Freeze any later held-out multi-seed protocol separately; never count these 12 development scenes as final paper evidence.

## Colab and export

Open GitHub main's `RUN_ALL.ipynb` in a fresh GPU runtime; keep `TIER="semantic_screen"`, `RUN_ID=""`. Code changed, so do not resume a previous run ID. Default generation budget is 20 minutes, checked between scenes; downloads/setup/evaluation are outside that soft limit. Forty-eight final outputs involve multiple candidates, not 48 diffusion samples. No M3, COCO, FID or tuning grid runs in this tier.

After completion, run `03_export_screen_review.ipynb` on CPU, using the printed RUN_ID (blank selects latest). It includes the new methods, diagnostics, review cards and ratings in ZIP parts below 32 MiB, without a GitHub token. Send metrics and image parts. Reviewers should not inspect the key/metadata before scoring the anonymized cards.

## Offline verification

Tests cover supported/unsupported contrasts, routing/cost bounds, semantic guard failures, identical global seed pools, rejected-image evidence, real tiny-model replacement with protected pixels, cache/resume, charging shared generation and cached M2, all 48 planned rows, missing-population rejection, and anonymized review mapping/blank ratings. Older screen, sparse repair, reproducibility and ZIP-export regressions are also run. These tests verify software behavior on CPU, not GPU quality, attribute correctness, or novelty.
