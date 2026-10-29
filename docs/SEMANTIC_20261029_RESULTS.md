# Semantic routing: negative development result and stop decision

Run `20261029T061148Z_c663e968`, code `393ebd3`. Source: metrics ZIP and all six image ZIP parts supplied by the user. All 318 exported manifest-listed files pass SHA-256 verification, including 252 PNGs. Forty-five unexported manifest files belong to the M2 `_objects` cache; they are outside this export's scope. All 48 final method/scene/seed rows are present and unique, 12 per method. Recomputing all 15 paired comparisons reproduces differences, bootstrap intervals and raw/Holm-adjusted p-values.

| Method | OPR | mIoU | Count error | Relation accuracy | Reconstructed standalone s/image | Sampling work/image |
|---|---:|---:|---:|---:|---:|---:|
| GLIGEN layout best-of-5 | .8690 | .6978 | 1.7500 | .6887 | 12.2840 | 300.00 |
| GLIGEN semantic best-of-5 | .8400 | .6777 | 2.0000 | .6282 | 10.7894 | 300.00 |
| Sparse M2 repair | .9023 | .7047 | 1.6667 | .7638 | 5.2369 | 250.00 |
| Semantic routed repair | .8400 | .6722 | 1.6667 | .6282 | 9.7415 | 238.33 |

Semantic repair loses 6.23 percentage points OPR and .03245 mIoU versus sparse M2, with equal mean count error and lower relation accuracy. Recorded reconstructed time is about 1.86 times sparse M2. Sampling work is slightly lower, illustrating why that proxy cannot substitute for latency including verifier/scoring overhead. Shared-generation/M2 timing is reconstructed, not an independent timing trial. No statistically significant superiority is established; adjusted p-values for OPR/mIoU versus sparse are .75/.625. This small screen does not establish universal inferiority either, but provides no reason to scale this version.

## What actually ran

Routing: eight global, three keep, one local. Of 33 candidates examined after initialization, three are accepted; 15 rejected for count/layout regression, 14 for lost objects, one for a new semantic issue. The accepted local edit is the yellow bicycle scene. The accepted global changes occur in the three-chair beach and five-object beach scenes. Routed finals are pixel-identical to the semantic best-of-five control on 10/12 scenes, and to sparse repair on 7/12.

All 12 routed outputs match the selected candidate/fallback implied by their logs. The single new local edit and all six sparse-control repair attempts preserve every pixel outside the saved edit masks. Sparse accepted/rejected finals match their logged selection. Charged work recalculated from executed/reused counters matches logged standalone work and stays within the routed cap on all 12 scenes. No software failure is evident from these checks; an internally accepted edit can still regress under independent OWLv2 or human inspection.

## Image findings and their limits

All 12 anonymized A-D cards were inspected before opening the key/metric table. This is an AI engineering inspection, **not independent human ratings**: prior runs make several images recognizable. The supplied human ratings CSV remains blank. No human scores were fabricated.

- On the yellow-bicycle scene, local repair adds a prominent yellow bicycle, but fragments/extra bicycle content remain. Independent count error improves by four versus sparse, while mIoU declines .04785. This is a narrow useful effect, not an overall method win.
- On the five-object beach scene, semantic/global selection prefers red sail/tent-like forms and a more evident sea background; OWLv2 OPR falls .40 versus sparse. The internal score/attribute gain did not preserve independent object presence.
- On the crowded living-room request, the routed final remains outdoors, with incorrect chair colors and unresolved animal counts. It gives up the object-presence gains of sparse repair (.22222 OPR difference).
- On the bus/train scene, routing returns the original red-bus/small-green-road-vehicle image instead of sparse's green-vehicle repair; OPR loses .125.
- The horse/dog and bird/rabbit/tree scenes remain unresolved. Global alternatives fail the guard; falling back prevents additional damage but does not fix the initial errors.
- On the white-umbrella/gray-cat request, all methods keep a blue umbrella and dark cat. The tree-in-meadow request remains on a gray/white surface. Detecting a semantic issue has not supplied an effective correction.

## Diagnosis and decision

The policy often diverts missing-object repair into global resampling because of background/attribute flags. Its conservative guard then rejects most alternate layouts; the fallback retains errors which sparse repair would address. Conversely, a few accepted proxy gains do not transfer to the independent detector. The yellow-bicycle result is worth documenting, but not selecting alone as evidence for the whole method.

**Stop this semantic policy version.** Keep its code and negative result for reproducibility; restore `RUN_ALL`'s default to `repair_screen`. Do not rerun completed screens, increase seeds, or launch paper/full for this candidate. Sparse M2 remains the better development implementation in these tests, not a proven superior method over all baselines.

The next useful work is offline failure diagnosis using this saved candidate pool: measure what the pool contains versus what the guard selects, and calibrate a verifier on separately annotated development data. A new method proposal must address correction quality (especially wrong-class replacement) and verifier disagreement, rather than relaxing thresholds on these 12 scenes until metrics improve. Freeze any new rules before a new held-out run. No additional GPU execution is needed to complete this audit.
