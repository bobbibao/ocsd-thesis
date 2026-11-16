# OCSD-XL confirmation on held-out v3 (`xl_confirm_v1`), written before any image of this study

Date: 2026-11-17. Run mode `xl_confirm` in `notebooks/RUN_ALL.ipynb`. Locked copy: `assets/xl_confirm_v1/protocol.json`
(written by `tools/lock_xl_confirm.py`; the runner refuses to start if the settings differ). Analysis code:
`tools/analyze_xl_confirm.py`, committed with the lock.

## Why, and what is different from the development rule

Three development rounds ([round 3](XL_DEV3_RESULTS.md)) looked for one OCSD-XL setting that is both natural and within
0.03 of the best structure score. None exists: the scene ControlNet weight trades one against the other. That rule
stays failed. This study asks a different, narrower question, stated here before the data: do the two ends of that
trade-off, frozen as they were in round 3, each beat the original OCSD on the thing they are good at, on scenes the
SDXL line has never touched? Bobby approved running it on 2026-11-17.

## Frozen settings

Same code path as the thesis method (`ocsd/method.py`, M2, M4, M5; M3 off as in OCSD-v2) on SDXL base 1.0 at 1024 px.

| arm | round 3 name | scene ControlNet | energy step | everything else |
|---|---|---|---|---|
| `ocsd_xl_natural` | `ocsd_xl_w04` | 0.4 | on, 0.02 | joint trajectory, alpha 0.4, repair on, K = 3 |
| `ocsd_xl_faithful` | `ocsd_xl_w08_noenergy` | 0.8 | off | same |

## Arms (11)

| arm | source |
|---|---|
| `ocsd_xl_natural`, `ocsd_xl_faithful` | generated in this run |
| `xl_controlnet`, `xl_collage` | generated in this run (same backbone and ControlNet) |
| `ocsd_v2` | generated in this run on SD 1.5, paper settings with the frozen paper_v2 tuning (alpha 0.4, no shrink) |
| `ocsd` (original, with M3), `gligen`, `collage`, `controlnet`, `ocsd_v3`, `direct_box` | images of the locked held-out v3 run of 2026-11-05 (same scenes, seeds 0 and 1), read from Drive (`heldout_v3/outputs/runs/20261105T015848Z_8ed9c617`, checksums verified) and scored again here |

If an image of the earlier run is missing on Drive, `ocsd`, `gligen`, `collage` and `controlnet` are generated again
with the same settings; `ocsd_v3` and `direct_box` cannot be and would be reported as incomplete.

Scenes: the 18 scenes of `assets/heldout_v3` (3 / 5 / 8+ objects x simple / medium / complex, two per cell), seeds 0
and 1. No OCSD-XL setting was chosen or run on them.

## Measures

OPR, mIoU, count accuracy, relations (OWLv2), sketch chamfer and facing (SAM), CLIP, PickScore, all at 512 px. Blind
naturalness (0 broken or pasted, 1 fake but readable, 2 natural) on all 36 sheets, every arm on each sheet in shuffled
order, written before the key or any table is opened.

## Analysis and claims

Unit: scene (18), seeds averaged. Exact two-sided sign tests, Holm over the four primary tests, paired bootstrap CIs.

| id | comparison | measure |
|---|---|---|
| P1 | `ocsd_xl_faithful` vs `ocsd` | OPR |
| P2 | `ocsd_xl_faithful` vs `ocsd` | mIoU |
| P3 | `ocsd_xl_natural` vs `ocsd` | blind naturalness |
| P4 | `ocsd_xl_natural` vs `ocsd` | OPR |

- **C-faithful** holds if P1 or P2 is significant (Holm p < 0.05) in favour of OCSD-XL and neither is significantly worse.
- **C-natural** holds if P3 is significant in favour of OCSD-XL and P4 is not significantly worse.
- An improved OCSD goes into the thesis, paper and slides only for a setting whose claim holds, with its trade-off
  stated. If neither holds, the SDXL line stays a negative result.
- Everything else (both settings against OCSD-v2, the SDXL baselines, GLIGEN, Collage, ControlNet, OCSD v3,
  InstanceDiffusion) is secondary: differences with CIs and unadjusted tests, labelled exploratory.

## Limits

- The naturalness rater is the analyst who developed the settings. Sheets are blind, but SDXL images can be
  recognisable by their look.
- OCSD-XL runs without M3; the original OCSD uses it.
- Part of any difference to the SD 1.5 methods is the backbone. `xl_controlnet` and `xl_collage` show what the
  backbone gives without M2-M5.
- 18 scenes give limited power; a true difference can fail the test.
- The 72 paper scenes are a second phase with its own lock, run only if a claim holds here.

## Cost

About 2 to 2.5 A100 hours: 144 SDXL images plus M2 objects, 36 SD 1.5 images, scoring of 396 images. Resumable.
