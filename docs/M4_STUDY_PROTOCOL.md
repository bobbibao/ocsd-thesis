# OCSD with harmonised M4 (`m4_study_v1`), written before any image of this study

Date: 2026-11-24. Notebook `notebooks/06_ocsd_m4_one_run.ipynb` (one Run all). Code `ocsd/m4_study.py`, M4 in
`ocsd/method.py` (`blend_mode="harmonized"`), arms in `ocsd/methods.py` (`M4H`, `M4_ARMS`). Locked copy:
`assets/m4_study_v1/protocol.json` (written by `tools/lock_m4_study.py`; the run refuses to start if the code's settings
differ). CPU test of the whole path: `tests/test_m4_study.py`.

## Why

There is one OCSD: Stable Diffusion 1.5 with all five modules M1-M5 and M3 on. The trace of the thesis configuration
(`notebooks/04_trace_ocsd`) shows that M4 pastes the M2 objects into the latent for 27 of 30 steps and leaves 3 steps
to the scene prompt, the scene ControlNet and M5. The background is denoised alone, with the object classes in its
negative prompt. Results: visible object borders (a white halo from the M2 crops), light and shadow that do not
match, and OPR 74.0 against GLIGEN 82.2 and Collage 84.2 on the paper benchmark (`results/paper_v2`).

## What changes (M4 only; M1, M2, M3, M5 are the thesis modules)

`blend_mode="harmonized"`:
1. **One scene trajectory.** The U-Net sees the whole composite from the first step, with the scene prompt, the scene
   ControlNet M5(c), region attention M5(a), energy guidance M5(b) and the M3 identity tokens and LoRA. The background
   therefore gets the scene's light and its contact with the objects.
2. **Halo-free anchor.** Each placed M2 mask is eroded by 2 px, and outside the objects the canvas is filled by
   push-pull extrapolation of their colours instead of white, so the VAE latent at an object border holds no white.
3. **Feathered, shrinking mask.** Gaussian feather of 4 px. The border band of each object is released step by step
   (30% of its inner radius by the last anchored step).
4. **Same noise.** The anchor `q(z0, t)` uses the trajectory's own initial noise, so there is no noise seam.
5. **Smooth release.** `z <- z + w(t) m (q(z0, t) - z)`, with w = 1 until 45% of the steps, then a cosine decay to
   0 at (1 - alpha) T = 70%. During the decay only the low frequencies of the difference are re-imposed (Gaussian with
   sigma 2 latent px), so colour and layout stay while edges, texture, shadows and lighting are redrawn. The last 30%
   of the steps are free.
6. **M5 fixes of OCSD-v2** (small-object masks, caption class words limited to their masks, mean energy) and
   object-level M5(d) repair: a failed check re-denoises only the missing objects' regions.

The tuned values of the thesis run (LoRA strength 0.5, energy on phrase tokens with eta 20, region attention, scene
ControlNet 0.4, 30 steps, K = 2, M3 100 + 100 steps) are unchanged (`assets/heldout_v3/tuned_paper_v2.json`).

## Stage 1: development (tuning split only)

All 36 QuickDraw scenes of the frozen paper_v2 tuning split (`data/benchmarks/tuning_split.json`, scenes 000-002 of
each of the 12 object-count x complexity cells), seeds 0 and 1. No development scene is an evaluation scene. Arms:

| arm | change against the centre |
|---|---|
| `ocsd` | the thesis OCSD (reference) |
| `ocsd_m4` | centre (above) |
| `ocsd_m4_a02`, `ocsd_m4_a04` | anchoring ends at 80% / 60% of the steps |
| `ocsd_m4_rel40` | release over 40% of the steps |
| `ocsd_m4_w06` | scene ControlNet 0.6 |
| `ocsd_m4_lora_late` | M3 LoRA only from 45% of the steps |
| `ocsd_m4_ctx` | M2 objects drawn in the scene's context ("a photo of a {phrase} {bg}, natural light") |
| `ocsd_m4_hard` | the same trajectory with the hard anchor of OCSD-v2 (none of points 2-5) |

**Second development step (combined arm).** Each one-factor arm that is eligible (see below) and scores above the
centre contributes its change; of the two alpha arms, only the better one counts. If at least two changes qualify,
the centre with all of them applied is generated as `ocsd_m4_combo` on the same scenes and seeds and joins the
candidates. The plan is written once to `dev/combo.json`.

## Stage 2: freeze (automatic, before any evaluation image)

Per arm: means over the 36 scenes (seeds averaged) of OPR, mIoU and RA (OWLv2, as in every earlier run) and PickScore.
`score = mean(OPR, mIoU, RA) + 0.05 x (PickScore - PickScore of ocsd)`. Eligible: at least 95% of the images exist
and PickScore is not below that of `ocsd`. The eligible arm with the highest score is frozen into
`<Drive>/KLTN_OCSD/m4_study_v1/locked.json` and becomes `ocsd_final`. A later Run all never chooses again. No human
looks at development images before the freeze. Development numbers are never reported as results.

## Stage 3: paper benchmark

The 72 evaluation scenes of `results/paper_v2` (copied from the benchmark on Drive and checked against
`assets/paper_v3/expected_scenes.json`), seeds 0 and 1. `ocsd_final` is generated here. The images of `ocsd`,
`gligen`, `collage`, `controlnet` and `ocsd_v2` on the same scenes and seeds come from the locked run of 2026-11-17
(`xl_paper_v1`; checksums verified). Their scores from that run are reused only for byte-identical images (same
evaluator code and models). Anything missing is generated or scored again with the same settings.

Primary tests: `ocsd_final` vs `ocsd` on P1 OPR, P2 mIoU, P3 RA and P4 PickScore. Unit: scene, mean over the two seeds.
Exact two-sided sign tests with Holm over P1-P4, paired bootstrap 95% CIs, and Wilcoxon p alongside.

- **C-structure** holds if P1, P2 or P3 is significant in favour of `ocsd_final` and none of P1-P4 is significantly worse.
- **C-natural** holds if P4 is significant in favour and neither P1 nor P2 is significantly worse.
- Comparisons with OCSD-v2, GLIGEN, Collage, ControlNet, T2I-Adapter and Zhang et al. (2025) on OPR, mIoU, RA, count
  accuracy, sketch chamfer, PickScore and CLIP are secondary, with Holm over that family. No claim of beating a
  baseline is made unless its adjusted p is below 0.05.
- Blind naturalness ratings (0 / 1 / 2) of 36 sheets (scenes 003-005, seed 0, all arms shuffled) are made after the
  run, before `review_key.json` is opened. They are secondary.

## Stage 4: extra baselines

T2I-Adapter (scale 0.8) and Zhang et al. (2025) (alpha 0.1), the paper_v2 tuned values, are generated on the same
scenes and seeds after the primary report. The report is then written again.

## Cost and resume

A100: development about 5 to 6 hours (9 arms, plus the combined arm, x 72 images, M2 and M3 for every scene), paper
benchmark about 2 to 2.5 hours, extras about 1.5 hours. In total about 9 to 10 hours, or roughly 110 to 120 Colab
compute units on an A100. Every image, M2 object
set, score and the freeze are kept on Drive. A new Run all continues where the last one stopped. The run also stops
cleanly at `MAX_HOURS`.

## Limits

- Selection uses automatic metrics. PickScore is a learned proxy of human preference, not a human rating.
- Selection is one factor at a time around the centre, plus one combined arm. It is not a full grid.
- The 72 paper scenes have been used to report the SD 1.5 methods before. No harmonised-M4 image of them exists
  before this run, and no setting is chosen on them.
- QuickDraw-Scenes only. COCO-Sketch is not part of this study.
