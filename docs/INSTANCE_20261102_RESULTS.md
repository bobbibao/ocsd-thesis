# Instance conditioning: limited development gains, no overall success gain

Run `20261102T165132Z_e9ea4c56`, code `69bdfea`, NVIDIA L4 (22 GB), 12 development scenes × one seed × four methods. Source: user-supplied `1.zip` metrics and all four image parts. This is the completed GPU run of the [conditioning protocol](INSTANCE_SCREEN.md), not a new proposed experiment.

## Integrity and controlled comparison

All 287 exported manifest-listed files pass SHA-256 verification, including 221 PNGs. The 53 omitted manifest artifacts are M2 `_objects` cache entries outside the export scope. All 48 final metric rows are present and unique. Recalculation reproduces all 18 paired comparisons, including bootstrap intervals and raw/Holm-adjusted p-values. Human review ratings remain blank.

Across all 12 matched global/focused pairs, initial images, target IDs, configuration, seed, M2 cache key/subset, edit masks where applicable, and charged sampling work agree. All ten repair attempts preserve every pixel outside their saved edit masks. All 24 final images match the accepted candidate or fallback indicated in the logs. Recomputed standalone work agrees with the logs and respects the cap. See [per-scene routes](INSTANCE_20261102_ROUTES.csv). These checks support the conditioning-bundle comparison; they do not isolate text, negative text and sketch conditioning separately.

## Results

| Method | OPR | Box mIoU | Class count error | Relation accuracy | Joint layout success | Sampling work/image | Reconstructed s/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLIGEN layout best-of-5 | .8856 | .7287 | 1.7500 | .7237 | 6/12 | 300.00 | 18.0562 |
| Sparse M2 repair | .8690 | .7024 | 2.0000 | .6819 | 6/12 | 250.00 | 10.9018 |
| One-object repair, global conditioning | .8493 | .6910 | 2.0833 | .6460 | 6/12 | 151.67 | 9.1373 |
| One-object repair, focused conditioning | .8826 | .7207 | 1.7500 | .7314 | 6/12 | 151.67 | 9.1270 |

Relations apply to nine scenes; other endpoints use twelve. Joint success requires all instances matched, zero class-count error, every box IoU >= .5, and all applicable relations correct. It does not measure silhouette, attribute or aesthetic correctness.

Focused versus matched global conditioning improves OPR by 3.33 percentage points (95% bootstrap interval [0, 8.33] points; raw p=.5, Holm p=1), mIoU by .02978 (interval [-.000064, .07475]; raw p=.25, Holm p=.75), and count error by -.3333. These are development signals, not statistically established superiority. The same six scenes pass the joint endpoint for all four methods; there is no improvement on that endpoint.

Versus sparse repair, focused OPR is +1.37 points (interval [-3.43, 7.29]); versus GLIGEN best-of-five it is -.30 points (interval [-4.05, 4.07]). Neither supports superiority. Focus uses 49.44% fewer sampling units than best-of-five, but this proxy excludes auxiliary models, VAE and I/O and is not FLOPs. Times charge reused generation and cached M2; they are not independently randomized latency measurements or proof of equivalent quality at half the cost.

The preceding screens used A100 hardware. Even though the dataset identifier is unchanged, their scores/times must not be pooled with these L4 results as a controlled algorithm comparison; code and runtime differ too.

## What the images show

Engineering inspection covered the five attempted-repair scenes with original sketches and all four method finals. This was an AI inspection with method identities and metrics known, not blind human evaluation. No ratings were invented.

- `qd_5_medium_000`: focused repair visibly replaces the central extra horse with a brown dog while retaining the other horses. The global-conditioning candidate is rejected. Independent OWLv2 OPR rises .2, mIoU .16230, and class-count error falls by three. Yet OWLv2 still reports count error one, so joint success remains zero despite the internal GDINO guard reporting count error zero. This verifier disagreement warrants annotation, not a relaxed success threshold.
- `qd_5_complex_000`: focused repair replaces an umbrella-like blue region with a small sailboat. OPR rises .2, mIoU .19079 and count error falls by one versus the global arm. Other errors remain: independent OPR is .8 and joint success is zero. Requested sketch geometry is not established by this class correction.
- `qd_8p_complex_000`: both local variants are accepted; focus changes mIoU only +.00498 with no presence/count gain. The crowded scene remains unresolved.
- `qd_8p_medium_000`: both arms reject the bus-region candidate for CLIP regression and keep the original. The current policy supplies no improvement here.
- `qd_8p_simple_000`: both edits are accepted, but focus changes mIoU by -.00077 and adds no presence/count benefit. Bird-like forms do not establish correct requested instance identity or shape.

Each local variant keeps seven scenes without a target and attempts five edits. Focus accepts four; global conditioning accepts two. Thus the aggregate benefit is concentrated in two five-object scenes, not broad success across scene complexity.

## Decision and next research gate

Retain focused conditioning as an experimental candidate and preserve this exact implementation. Do not run paper/full, repeat this completed screen, increase candidate budgets, or tune thresholds against these twelve scenes. No inference code or notebook parameters were changed for this audit. The current results show useful local corrections but do not meet the user's full scene-generation objective or establish SOTA.

The next useful step is a CPU-only annotation pass over the existing packet: resolve detector disagreements and separately rate presence, count, layout, shape, relations, attributes, background and realism using the supplied input sketches. Keep human ratings separate from internal selection scores. Wrong-class replacement and untouched missing instances are concrete remaining failure categories; a new repair policy must be justified against both before requesting more GPU compute.

Before another validation run, freeze its rules and budget, use new held-out scenes and multiple paired seeds, and include a stronger instance-conditioned baseline such as InstanceDiffusion or MIGC++ with comparable inputs. Their integration is not completed here. The twelve repeatedly inspected scenes are development data, not a publishable held-out test. Also separate oracle instance annotations used by this generator from end-to-end parsing of arbitrary raw sketches and captions. Shape needs its own independently validated endpoint. None of those missing evaluations can be replaced by the current box metrics.
