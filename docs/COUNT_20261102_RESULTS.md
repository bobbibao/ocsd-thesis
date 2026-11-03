# Count replacement: detector success, unresolved visual quality

Run `20261102T181111Z_be735f1b`, code `32e973f`, NVIDIA L4 22 GB. This is the predeclared one-scene/four-method diagnostic, not held-out validation. Both uploaded ZIPs were merged safely. All 45 exported manifest artifacts pass SHA-256; four omitted entries belong to the intentionally unexported M2 `_objects` cache. All four final metric rows are present. The run completed normally.

[Per-image exported metrics](COUNT_20261102_METRICS.csv) are preserved without score changes.

## Measured result on the known bicycle failure

| Method | OPR | Class-count error | Box mIoU | Relations | Joint box-layout success | Sampling units |
|---|---:|---:|---:|---:|---:|---:|
| GLIGEN layout best-of-five | 1 | 6 | .76632 | 1 | 0 | 300 |
| Prior focused repair | 1 | 6 | .76632 | 1 | 0 | 60 |
| Count replacement, global conditioning | 1 | 0 | .81512 | 1 | 1 | 280 |
| Count replacement, focused conditioning | 1 | 0 | .81991 | 1 | 1 | 280 |

The independent OWLv2 count error falls from six to zero in **both** new arms. Internal GDINO count error falls from three to zero; both propose and accept the bicycle replacement. Their agreement supports a detector-level correction, not a guarantee of a clean, single, correctly shaped bicycle. OPR/relations are unchanged. Focused versus global differs only .00479 in mIoU, with identical count/joint outcomes; this one case does not establish an added benefit from focused conditioning. The useful new capability is routing an over-counted region into reconstruction.

The old focused arm keeps the initial image; it is pixel-identical to GLIGEN best-of-five's selected final. The repair therefore costs more than the no-op control (280 versus 60 units), while remaining below best-of-five's 300-unit proxy. Reconstructed standalone times are 61.77/25.14/47.10/46.95 seconds in table order. One-scene timing includes warmup/cache effects and auxiliary overhead; it is not a representative speed benchmark. Dependencies also differ from prior runs. No cross-run speed claim is made.

The exported summary contains collapsed bootstrap limits equal to each value because n=1. Those are not meaningful population confidence intervals. Reporting code now omits these limits for `count_screen`; paired significance tests remain empty. This presentation fix requires no model rerun.

## Mechanism checks

- Both new arms choose original object 0 and extra detection indices 2/3/4, with the same original-index M2 key/subset, seed, configuration, work and edit mask.
- The mask covers 5.0446% of the image. Both candidates preserve **every pixel outside it**, including both requested train rectangles. Their finals exactly match the accepted candidates. Counts/work reconstructed from executed and cached counters agree with the logs.
- The candidates are not pixel-identical: 13,016 pixels differ between them, but the visible structure and defects are very similar.
- Minor changes in detector boxes for the trains do not indicate edited train pixels: context-sensitive detection can change even when those pixels are preserved.

## Visual assessment and decision

Engineering inspection compared the original sketch, initial/final controls and both repaired finals. This is AI inspection with method identities known, not a human study. Subsequent qualitative user feedback is recorded below; numerical ratings have not been supplied.

The initial image contains a row of small blue bicycles. Both replacements introduce a much larger yellow bicycle-like structure in the requested region. The blue trains remain intact. However, the yellow frame/wheels are distorted, and dark fragments resembling old bicycle parts remain around/below the new structure. The generated outline is not convincingly the clean two-wheel sketch. A detector can count the dominant structure once while ignoring fragments.

**Do not call this a complete scene-quality success or escalate to paper/full.** The routing and count correction have a positive mechanism signal, but naturalness/shape remain unresolved. Do not select the .00479 mIoU difference as evidence of focused superiority. No additional sampling or threshold relaxation is justified from this result alone.

A small [review packet](reviews/ocsd-count-review-20261102.zip) contains the original sketch/caption and three distinct final images (four anonymized method labels). Ask the user specifically about count, shape, requested color and realism, with optional comments on fragments; leave unknowns/missing answers intact. This is a development review, not independent blind evidence given prior discussion.

The next engineering hypothesis is **residual content in the reconstruction region**: the current repair re-noises the old image at strength .5 and anchors M2 foreground, which may preserve old structures elsewhere in the editable region. Incomplete segmentation, the generated proposal itself, or mask/conditioning interactions are also possible; this audit does not isolate their contributions. Investigate clean region initialization and proposal/mask diagnostics before spending on broader evaluation. Do not erase or enlarge regions blindly: other valid objects must remain protected. No inference algorithm or default tier was changed by this results audit.

## Original review instructions

Download the linked ZIP, extract, open `REVIEW.html`, and examine its one scene. For each of the three distinct images, prioritize **count, shape, attributes and realism**. Use 0/1/2 or ? if unclear. In comments, say how many complete bicycles you can identify and whether fragments/extra wheels remain; do not force an exact count when the geometry is unclear. Export JSON and return it in chat. No Colab execution is needed.

## User feedback received — 2026-11-03 (Asia/Saigon)

The user reports: “ảnh B và C đã xuất hiện xe đạp màu vàng nhưng còn rất méo mó, và ảnh cũng chưa được tự nhiên lắm”. The manifest-verified review key maps B to `instance_repair_count_global` and C to `instance_repair_count`. [Structured qualitative record](reviews/count_20261102_user_feedback.json).

This confirms the appearance of a yellow bicycle in both new variants, but also substantial distortion and insufficient naturalness. It does **not** confirm exact count, absence of residual parts, a preference between B/C, or any numerical rating. No scores are inferred from this statement. This feedback is sufficient for the current development decision; the user does not need to re-enter it in JSON.

**Visual-quality gate not met for either variant.** Retain count-aware routing as a potentially useful component, but do not promote this reconstruction version or schedule a broader GPU run based on the zero detector count error. The next work should isolate proposal quality, segmentation/placement and old-content persistence during local denoising. Clean region initialization is a hypothesis to test, not an established fix; stronger instance/shape conditioning may also be necessary. Protect both trains and keep a matched-compute control in any next ablation. Do not tune detector thresholds to accept the visibly malformed bicycle.
