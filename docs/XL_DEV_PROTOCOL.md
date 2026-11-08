# OCSD-XL development study (`xl_dev_v1`), written before any SDXL image existed

Date: 2026-11-12. Run mode `xl_dev` in `notebooks/RUN_ALL.ipynb`. Machine-readable copy: `assets/xl_dev_v1/protocol.json`
(the runner refuses to start if the arm settings, scenes or models differ from that file).

## Why this study

Two facts from the earlier runs set the direction.

1. On Stable Diffusion 1.5 the original OCSD (M1-M5) and OCSD-v2 lose to tuned GLIGEN and to the Collage baseline on
   object presence, and their images look less natural (`results/paper_v2`, `docs/PAPER3_20261105_RESULTS.md`).
2. OCSD v3 fixes this by replacing M2-M5 with an external layout generator (InstanceDiffusion) and a refiner. It is a
   different method from the one in the thesis and slides, so it stays in the repository only as an external comparison.

OCSD-XL keeps the method and changes the backbone. The code path is the same `ocsd/method.py` (M2 object branch, M3
identity learning, M4 latent blending, M5 region attention + attention-energy guidance + scene ControlNet + verification
and repair). What changes:

| | SD 1.5 runs | OCSD-XL |
|---|---|---|
| Backbone | SD 1.5, 512 px | SDXL base 1.0, 1024 px (`ocsd/engine_xl.py`) |
| Sketch control | `lllyasviel/control_v11p_sd15_scribble` | `xinsir/controlnet-scribble-sdxl-1.0` |
| VAE | SD 1.5 | `madebyollin/sdxl-vae-fp16-fix` |
| Sampler | DDIM 30 steps, CFG 7.5 | DDIM 30 steps, CFG 6.0 |
| Energy step (M5b) | raw gradient, `eta` tuned to SD 1.5 | unit-RMS gradient, `eta = 0.05` (scale-free) |
| Trajectory (M4) | v1 "separate" or v2 "joint" | v2 "joint" as centre; v1 "separate" kept as an arm |

Images are generated at 1024 px and scored at 512 px with the unchanged evaluator, so numbers are comparable in kind with
every earlier run.

## Design

- Scenes: the 9 scenes `*_000` of `assets/heldout_v1` (one per cell: 3, 5, 8+ objects × simple, medium, complex). These
  scenes were already used for development of the InstanceDiffusion line; they are development data here too and are
  never used for a claim.
- Seeds: 0 and 1. M2 objects are generated once per (scene, seed) and shared by every arm.
- 11 arms, 198 images:

| Arm | What it is |
|---|---|
| `xl_controlnet` | SDXL + scribble ControlNet on the whole sketch (reference baseline on the same backbone) |
| `xl_collage` | M2 objects pasted at sketch positions, then ControlNet + SDEdit (strength 0.3) |
| `ocsd_xl` | centre configuration: joint trajectory, α = 0.4, ω = 0.8, energy on, repair on, M3 off |
| `ocsd_xl_a02`, `ocsd_xl_a06` | α = 0.2 / 0.6 (how long the M2 anchors are blended in) |
| `ocsd_xl_w04` | scene ControlNet weight ω = 0.4 |
| `ocsd_xl_noenergy` | M5b off |
| `ocsd_xl_norepair` | M5d off |
| `ocsd_xl_region` | α = 1.0: no M2 anchoring, single-branch generation with M5 only |
| `ocsd_xl_sep` | the thesis (v1) sampler: separate trajectories, α = 0.1, no repair |
| `ocsd_xl_id` | centre + M3 identity tokens and LoRA (100 + 100 steps) |

## Measures

Same evaluator as before: OPR, mIoU, count accuracy, relation accuracy (OWLv2), CLIP score, sketch chamfer and facing
(SAM silhouettes). Added: PickScore v1 as a secondary image-quality proxy, and a blind naturalness review (0 = broken,
1 = acceptable, 2 = natural) on shuffled, unlabeled contact sheets (`REVIEW_xl_dev_v1.zip`).

## Decision rule (fixed now)

- Candidates for "the" OCSD-XL: `ocsd_xl`, `ocsd_xl_a02`, `ocsd_xl_a06`, `ocsd_xl_w04`, `ocsd_xl_noenergy`,
  `ocsd_xl_sep`, `ocsd_xl_id`. (`_norepair` and `_region` are ablations only.)
- Structure score S = mean of (OPR + mIoU) / 2 over the 18 images. Naturalness N = mean blind score.
- Admissible: N ≥ 1.0 and S ≥ (best candidate S) − 0.03.
- Pick the admissible candidate with the highest N; ties within 0.1 go to lower sketch chamfer, then lower run time.
- Go to confirmation only if the picked candidate has S ≥ S of `xl_controlnet`. Otherwise run one more development
  round (on held-out v2 scenes) before any confirmation.
- Only then is one configuration frozen and tested on untouched scenes (the 18 held-out v3 scenes and the 72 paper
  scenes) against OCSD v1, OCSD v2, GLIGEN, Collage, ControlNet, InstanceDiffusion and the SDXL baselines, with a
  separately locked protocol.

No number from this development study will be reported as a result. It only selects the configuration.

## Cost

Roughly 2 to 3 A100 hours plus a one-time download of about 20 GB of public models. The run is resumable: finished
images are kept on Drive (`KLTN_OCSD/xl_dev_v1/`).
