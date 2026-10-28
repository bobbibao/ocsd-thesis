# Experiment changes — 2026-10-28

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
