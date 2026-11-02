# Count-aware replacement: bounded mechanism test

The user review confirms surplus bicycles inside the requested bicycle region. The prior missing-only router sees all requested objects and skips this scene. Simply removing unmatched boxes outside all requested regions would not address this case.

## Implementation

`instance_repair_count` retains the focused missing-object branch. When no requested object is missing but class-count error remains, it can instead reconstruct **one overloaded instance region** with the existing M2 object proposal and local diffusion repair. It does not globally remove unmatched detections.

An eligible extra detection must:

- be unmatched, belong to an over-counted requested class, and have a finite positive-area box;
- overlap every matched detection by less than .5 intersection/minimum-area (across classes too), conservatively excluding duplicate labels/contained detections;
- have more than half its area inside exactly one requested box of its class.

These are prospective geometry heuristics, not calibrated duplicate-detection probabilities. Choose the requested instance with most eligible extras; original index breaks ties. If nothing qualifies, keep the prior result. Outside-region extras, ambiguous/occluded instances, wrong shapes, and detector false positives may remain unhandled.

For reconstruction, release only this target from the **repair-mask** matching. The original matching remains the basis of acceptance. Edit the requested region plus its selected extra boxes, then subtract all other verified object masks/detection boxes. Reuse the original-index M2 subset, noise seed and existing focused conditioning. Pixels outside the resulting mask are pasted back exactly.

Acceptance retains every originally matched requested instance, does not regress count/mIoU/relations or exceed the existing CLIP tolerance, and **requires strict class-count-error reduction**. A better IoU alone is insufficient. Rejected candidates are saved and the original image is returned. No extra retries, new threshold fitting, identity training or energy guidance were added.

## Four-image Colab diagnostic

`RUN_ALL.ipynb` now defaults to `TIER="count_screen"`: **one known development scene (`qd_3_simple_000`) × one seed × four methods**:

1. `gligen_layout5`: five-candidate layout selection.
2. `instance_repair_focus`: frozen prior focused policy, which skips count-only cases.
3. `instance_repair_count_global`: new count routing with full-scene conditioning on the count-replacement path.
4. `instance_repair_count`: identical count routing/mask/M2/seed with focused conditioning on that path.

Both new arms use the prior focused policy on missing-object cases; their conditioning difference is limited to the count path. Initial GLIGEN generation and M2 proposals are shared and charged in standalone counters. At default settings a repair costs at most `60 + 160 + 60 = 280` batch-weighted sampling units, versus 300 for GLIGEN best-of-five. This excludes detector/encoder/SAM/VAE work and is not a latency guarantee. Installation/model loading may dominate a four-image run.

This is a deliberately tiny test of the proposed correction, **not held-out validation**. It generates four final images plus internal candidates. The old 12-scene screens remain available unchanged by their tier names. The count report writes no paired significance tests. If the routing conditions do not occur in a new runtime, it may keep the input; the logs will say so rather than forcing an edit.

## Verification before GPU use

CPU tests exercise real tiny diffusion execution, original-index M2 caching, shared initialization, identical masks/work across the new pair, protection of other objects, exact outside-mask preservation, resume, count-only acceptance, duplicate/outside/missing rejection, four-image planning and report generation. All 44 tests passed: 4 new count tests plus 40 instance, screening, sparse-repair, semantic, reproducibility and export regressions. All five notebook schemas/code cells and deterministic regeneration also pass. Tiny random networks verify mechanics, not pretrained image quality.

[Replay of saved detections](COUNT_ROUTE_AUDIT.json) selects the bicycle instance and extras 2/3/4 on the reviewed run. Only this count-only scene triggers the new router among those twelve saved initial states. The replay produces no new images and does not establish an improvement. It also cannot validate the .5 geometry heuristic on unseen examples.

## Run and interpret

Open the current GitHub `RUN_ALL.ipynb` in a fresh GPU runtime. Leave `TIER="count_screen"` and `RUN_ID=""`; use **Run all**. Keep the printed new run ID. Do not resume the prior instance run under new code. Export that run with `03_export_screen_review.ipynb` on CPU, and send the metrics/image ZIPs as before. No GitHub token is needed for export.

Inspect whether the bicycle count improves while both trains, requested position and image quality are retained. Review the before/candidate/final, including rejected edits; a detector-only gain is not sufficient. If successful, freeze the policy and test new scenes/paired seeds plus stronger instance-conditioned baselines; do not call this single selected failure a model-wide win. If unsuccessful, stop this variant rather than increase retry budgets or relax guards on this scene.

This change does not solve the airplane false acceptance, elephant pose, per-instance colors, background mismatch or crowded bird/rabbit scene. Those require stronger conditioning and independently validated semantic/shape evaluation. No SOTA, paper score or pretrained quality improvement is claimed before the GPU test.
