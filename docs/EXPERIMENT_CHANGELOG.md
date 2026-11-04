# Experiment changes

## Real-weight CPU PASS and bounded GPU canary — 2026-11-03

- Audited `20261103T142406Z_013a5f0e`: matching code hashes, complete EMA/text weights, exact conditioner cache parity, nonzero scribble response and finite pretrained UNet output on an 8×8 latent. Preserved the result JSON; this is not image-quality evidence.
- Locked those model revisions/checksums for a four-image development canary: one new attributed QuickDraw composition, seeds 0/1, box-only versus box+scribble. No ControlNet/OCSD comparison or superiority claim yet.
- Added per-forward deadline/nonfinite checks, a 600-second child-process timeout, fixed FP32 VAE decode, hashed image completion records, resume, attempt-cost history and recoverable ZIP export. Added original-pixel PNG comparisons and a plain-text review template.
- RUN_ALL defaults to `direct_canary` on L4/A100; legacy experiment stages remain guarded. See [protocol, scope and instructions](DIRECT_CANARY.md). No pretrained GPU quality run has occurred in the development environment.

## Fix CPU preflight logging during CLIP loading — 2026-11-03

- Audited uploaded run `20261103T140807Z_e0a5503f`: source hashes match, downloads finish, but Transformers' loading report fails because the preflight Tee lacks `isatty()`. The numerical probe has not run; this is not a quality result.
- Forwarded stream capabilities/metadata to the wrapped terminal and kept `writelines` mirrored in the log. Added regression tests that fail before the fix, including real local CLIP loading inside the redirected preflight. Strict learned-weight checks remain enabled.
- Corrected download guidance using the observed 13.6 GB checkpoint and 1.71 GB CLIP files. Retry Run all in the existing CPU runtime to retain cache and fetch the fix; no GPU rerun or parameter change. [Audit](DIRECT_PREFLIGHT_20261103_RESULTS.md).

## Direct instance-conditioning candidate and CPU preflight — 2026-11-03

- Added an adapter to the pinned official InstanceDiffusion core: separate phrases/boxes and 20 real-ink sketch points per instance, without benchmark masks. This is an external baseline integration, not a new OCSD architecture or a measured quality gain. The inspected diffusers port nulls scribble embeddings, so it was not used for the sketch path.
- Cached immutable UniFusion features within one image and restored sampler/convolution state on success and failure. On the small official-core CPU test, cached/uncached latents match exactly; conditioner evaluations fall from 34 to 4 while UNet calls remain 34. Added CLIP pre-projection equivalence and strict Transformers-4-to-5 weight-key mapping tests.
- Added a strict real-checkpoint CPU compatibility probe with code/model hashes and ZIP logs, including dependency/download failures. Local real-weight verification is blocked by a Hugging Face proxy 403; no pretrained quality/FP16 claim is made.
- RUN_ALL now defaults to `RUN_MODE="direct_preflight"` on CPU. Its legacy experiment stages are guarded; `RUN_MODE="experiment"` retains the old GPU workflow. No new GPU quality sweep or paper/full run is recommended until compatibility is reviewed. See [protocol and instructions](DIRECT_INSTANCE_PREFLIGHT.md).

## Review fallback and qualitative feedback — 2026-11-03

- Fixed JavaScript-dependent image display: the review generator now embeds original images in static HTML and enables the form only after successful initialization. Added a normal PNG comparison with the same anonymous A–D mapping. A restricted HTML viewer no longer needs to run scripts to display the evidence; PNG is available when the viewer cannot render HTML at all.
- Recorded the user's verbatim count/shape acceptance for A–D, joint preference for A/B and remaining naturalness concern, bound to the original manifest labels/hashes. Perceived equality does not merge distinct image files; no numeric ratings are inferred.
- Updated current docs/notebook status: feedback received; no repeated rating or GPU run requested. No inference code, parameters or model weights changed.

## Foreground result audit — 2026-11-03

- Audited run `20261103T035056Z_dc40e4b5`: 81 artifact hashes, four exact CPU seed reconstructions, protected pixels, shared inputs and sampling counts. Learned extraction reduces visible wheel contamination, but geometry/naturalness remains inadequate. See [results](FOREGROUND_20261103_RESULTS.md).
- Added a reproducible CPU audit and original-image review packet with three visible criteria; unshown criteria remain unanswered. No weights, inference settings, acceptance thresholds or additional GPU work changed.
- Updated the notebook's opening status and current runbook: completed four-arm sweep; no rerun or paper/full recommendation. The existing tier remains reproducible.

## Original optimization — 2026-10-28

Base source: `8edd71a084a6a49ff3acc2e31f9a11594c16ecb5`. See Git history for the implementation commit.
No new GPU quality results are claimed. Historical result reference: `colab-results@9851d9991c6d4a518b072865e047a451e1dc37ef`.

## Changes that can change scientific results

- QuickDraw construction now has a deterministic random stream per scene. Previously extending a
  benchmark skipped random draws for existing scenes; incremental and fresh construction differed.
  Failed layout attempts advance that scene's stream; repeated placement failures raise an error.
  New notebook runs use `data/benchmarks_v2`, preserving the historical dataset. **Do not merge new
  tables with the old benchmark or claim the historical preregistration covers this new dataset.**
- M3 initialises LoRA in a seeded RNG scope, rebuilds a changed rank, and uses the scheduler's
  epsilon/v-prediction training target. These are correctness changes, not measured quality gains.
- Foreground placement respects each variant's placement setting instead of the first member of
  its shared M2 group. No new tuning defaults, verifier thresholds or method selection rules were chosen.
- E1/E2 now use common scene/seed rows, as E3 already does. Paired tests include a bootstrap CI
  for the paired mean difference and explicitly label insufficient finite pairs; they do not emit p=1
  as a substitute for an unrun test. Wilcoxon and Holm families are preserved.
- KID subsampling uses a fixed RNG stream. Its published standard deviation remains the subsample
  standard deviation, **not** a confidence interval across scenes.

## Experiment integrity and performance

- `provenance.py`: run namespaces, code/notebook/dependency hashes, Git metadata, frozen model
  revisions, input fingerprint, stage state, progress and artifact checksums. Explicit resume rejects drift.
  Notebook compatibility hashes executable code, excluding saved outputs and operational form literals;
  scientific settings remain in the experiment contract. The raw notebook hash is also recorded separately.
- `runner.py`: image signatures and pixel checksums; atomic PNG/JSON completion; stale artifacts are
  archived. M2 dependencies include sampling settings and input content. M3 grouping includes learning
  rates and attention-loss weight. Error cleanup clears identity and attention state.
- `metrics.py`: input/settings identities for per-image scores and robustness evaluators; content-based
  FID/KID identities include reference images and LPIPS mode. Reference hashes/features are reused within
  an evaluation call. Requested FID pairs must all exist; evaluator failures stop report preparation.
- `stages.py`: tuning refuses incomplete/duplicate evaluation populations; generation reports incomplete
  work and retains completed artifacts rather than continuing into an apparently final report.
- Actual U-Net/ControlNet forward calls and batch sample counts are recorded in per-image `compute`
  and shared preparation `prep_compute`; GLIGEN has its own counter. These are workload counters,
  **not FLOPs**. M3 backward passes and detectors also cost time. Do not sum shared preparation cost
  once for every image; group by scene/M2/M3 key when comparing amortised cost.
- Model revisions are locked through public Hugging Face metadata before a new Colab run. A locked
  run refuses a PiDiNet loading failure rather than silently changing its sketch extractor. Standalone
  synthetic CPU tests retain the documented Canny fallback.
- Notebook checkout reuse preserves local modifications. At the user's request, the GitHub/Colab
  workflow and `PUSH_RESULTS=True` default are restored; GH_TOKEN is read from Colab secrets.
  New runs fast-forward a clean checkout; explicit RUN_ID retains its code for resume.
  GPU presence is checked, stages propagate failures, and GLIGEN/engine memory is released before metrics.
- The manifest records the planned jobs after data preparation and is finalised after report-stage
  logging, so its checksums cover final stage status. The separate upload notebook selects the explicit
  or latest RUN_ID and includes its manifest. Versioned runs keep their own current config when freezing
  the pilot split, instead of moving it into a legacy pilot archive.
- `RUN_ALL.ipynb` has stale cell outputs cleared. The original notebook including outputs is preserved
  in Git at the base SHA and in the handover's `RUN_ALL_before_optimization.ipynb` backup.

## Quality experiment added — 2026-10-29

`screening.py` adds an experimental GLIGEN-initialised, protected local repair pipeline and its no-M2
control. It uses existing SD 1.5 repair after GLIGEN SD 1.4 generation, not a port of GLIGEN's grounded
denoiser. `repair_scene(protect_matched=True)` preserves detected object boxes/masks and prevents the
final feathering from changing pixels outside the allowed region. The default remains false for old methods.
The verifier/CLIP acceptance guard can reject a repair; it is not a guarantee of independent image quality.

`screen` is a separate 12-scene, 84-output plan, with no M3/COCO/FID/tuning. New variants are not
automatically added to paper/full. Full baselines, negative ablations and historical preregistrations
remain intact. See QUALITY_SCREEN.md for prospective rules, checks, cost accounting and limitations.

## Direct canary audited, review font fixed, region MIS merge added — 2026-11-04

Canary `direct_9b63ee5bdadc694320c9` (L4) completed 4/4 images within plan: 864 UNet sample calls,
peak 5.95 GiB, 199 s attempt wall time, checksums and provenance match the CPU PASS. Analyst
observations and limits: [DIRECT_CANARY_20261103_RESULTS.md](DIRECT_CANARY_20261103_RESULTS.md).
The review sheets showed boxes instead of Vietnamese text (Pillow fallback font); fixed with a
glyph-checked font choice and an accent-free fallback. `sample_latents` now runs the upstream MIS
loop with a pluggable merge: `official_mean` (default, bit-identical to upstream, tested) or
`region` (OCSD variant, untested on GPU). Merge diagnostics are logged per image. The next run is
only drafted: [DIRECT_DEV_PROTOCOL.md](DIRECT_DEV_PROTOCOL.md).

## Direct development study locked — 2026-11-04

User approved stage 1. Scenes (`assets/direct_dev_v1/scenes.json`) and the protocol with decision
rules (`assets/direct_dev_v1/protocol.json`, [DIRECT_DEV_PROTOCOL.md](DIRECT_DEV_PROTOCOL.md)) are
committed before any output. `ocsd/direct_dev.py` + `tools/run_direct_dev.py` run 8 scenes × 2 seeds
× 3 arms with checksums, resume and attempt history; `RUN_ALL.ipynb` default is now
`RUN_MODE="direct_dev"` and downloads a full ZIP plus a small blind review ZIP.

## Direct development study reviewed — 2026-11-04

Run `dev_c80aed018746b51a0cee` (L4, 48/48, 10 368 UNet sample calls, 1042 s). Blind review picks:
box 8, scribble 5, scribble + region 3; region produced all 4 naturalness-0 images and is rejected by
the locked rule; stage 2 uses `direct_box` by rule, although scribble had fewer structure failures
(1 vs 4). See [DIRECT_DEV_20261104_RESULTS.md](DIRECT_DEV_20261104_RESULTS.md).

## Held-out comparison locked — 2026-11-04

User chose stage 2 with `direct_box` (rule) plus `direct_scribble` (post-hoc). 18 new scenes
(`assets/heldout_v1`, generator seed 20261104, 3/5/8+ objects) and the analysis plan are committed before output
([HELDOUT_PROTOCOL.md](HELDOUT_PROTOCOL.md)). New tier `heldout_screen` and `ocsd/heldout.py`: direct arms in a
child run, imported with checksums; ControlNet and OCSD soft-matte via the existing pipeline; joint OWLv2/CLIP
evaluation, paired bootstrap report and 36 blind review sheets. `RUN_ALL` default is `RUN_MODE="heldout"`.

## Held-out results — 2026-11-04

Run 20261104T135658Z_41d72d97, 144/144 images. Blind review: structure success direct_box 8/36, direct_scribble 7/36, ControlNet 1/36, OCSD soft-matte 3/36; primary sign tests Holm p = 0.14 and 0.14 (not significant). Automatic metrics favour the direct arms with CIs excluding 0. Shape/pose is the bottleneck for all arms. See docs/HELDOUT_20261104_RESULTS.md.
