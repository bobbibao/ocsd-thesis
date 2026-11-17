# OCSD-XL on the paper benchmark (`xl_paper_v1`), written before any image of this study

Date: 2026-11-17. Run mode `xl_paper` in `notebooks/RUN_ALL.ipynb`. Locked copy: `assets/xl_paper_v1/protocol.json`
(written by `tools/lock_xl_paper.py`; the runner refuses to start if the settings differ). Analysis code:
`tools/analyze_xl_confirm.py`, unchanged since the first confirmation.

## Why

The first confirmation ([results](XL_CONFIRM_RESULTS.md)) tested two frozen OCSD-XL settings on 18 held-out scenes.
The claim for the natural setting held; the claim for the faithful setting did not reach significance. Its protocol
named a second phase: the same two settings on the 72 scenes of the paper benchmark. Bobby chose on 2026-11-17 to run
that phase before anything is added to the thesis, paper or slides. Nothing is changed between the two phases: same
code path, same settings, same arms, same four primary tests, same claim rules. The lock tool checks that the arm
digest and the SD 1.5 digest equal those of `assets/xl_confirm_v1`.

## Frozen settings

Same code path as the thesis method (`ocsd/method.py`, M2, M4, M5; M3 off as in OCSD-v2) on SDXL base 1.0 at 1024 px.

| arm | scene ControlNet | energy step | everything else |
|---|---|---|---|
| `ocsd_xl_natural` | 0.4 | on, 0.02 | joint trajectory, alpha 0.4, repair on, K = 3 |
| `ocsd_xl_faithful` | 0.8 | off | same |

## Scenes

The 72 evaluation scenes of `results/paper_v2` (QuickDraw-Scenes; 1 / 3 / 5 / 8+ objects x simple / medium / complex,
scene numbers 003 to 008), seeds 0 and 1. They are not committed: the run copies them from the paper benchmark on
Drive and refuses to start unless every scene matches `assets/paper_v3/expected_scenes.json`. The run also records
whether the copied files have the dataset identity that the 2026-11-05 paper benchmark run recorded.

No OCSD-XL image of these scenes exists before this run and no OCSD-XL setting was chosen on them. The SD 1.5 methods
and OCSD v3 have been reported on them before ([2026-11-05](PAPER3_20261105_RESULTS.md)).

## Arms (11)

| arm | source |
|---|---|
| `ocsd_xl_natural`, `ocsd_xl_faithful` | generated in this run |
| `xl_controlnet`, `xl_collage` | generated in this run (same backbone and ControlNet) |
| `ocsd_v2` | generated in this run on SD 1.5, paper settings with the frozen paper_v2 tuning (alpha 0.4, no shrink) |
| `ocsd` (original, with M3), `gligen`, `collage`, `controlnet`, `ocsd_v3`, `direct_box` | images of the locked paper benchmark run of 2026-11-05 (same scenes, seeds 0 and 1), read from Drive (`paper_v3/outputs/runs/20261105T085721Z_c08740c2`, checksums verified) and scored again here |

If an image of the earlier run is missing on Drive, `ocsd`, `gligen`, `collage` and `controlnet` are generated again
with the same settings; `ocsd_v3` and `direct_box` cannot be and would be reported as incomplete.

## Measures

OPR, mIoU, count accuracy, relations (OWLv2), sketch chamfer and facing (SAM), CLIP, PickScore, all at 512 px, on all
72 scenes and both seeds.

Blind naturalness (0 broken or pasted, 1 fake but readable, 2 natural) on 36 sheets: the scenes numbered 003 to 005
(3 of the 6 scenes of every cell, 9 per object-count bin), seed 0, every arm on each sheet in shuffled order, written
before the key or any table is opened. The subset is fixed in the lock by scene number, before any image exists. It
keeps the number of sheets equal to the first confirmation (36 sheets, 396 panels).

## Analysis and claims

Unit: scene. Automatic metrics: 72 scenes, seeds averaged. Naturalness: the 36 reviewed scenes, seed 0. Exact
two-sided sign tests, Holm over the four primary tests, paired bootstrap CIs.

| id | comparison | measure |
|---|---|---|
| P1 | `ocsd_xl_faithful` vs `ocsd` | OPR |
| P2 | `ocsd_xl_faithful` vs `ocsd` | mIoU |
| P3 | `ocsd_xl_natural` vs `ocsd` | blind naturalness |
| P4 | `ocsd_xl_natural` vs `ocsd` | OPR |

- **C-faithful** holds if P1 or P2 is significant (Holm p < 0.05) in favour of OCSD-XL and neither is significantly worse.
- **C-natural** holds if P3 is significant in favour of OCSD-XL and P4 is not significantly worse.
- A claim is reported as confirmed on the paper benchmark only if it holds here. The two test sets are reported side
  by side and never pooled. A claim that holds on one set and not on the other is reported as such.
- Everything else (both settings against OCSD-v2, the SDXL baselines, GLIGEN, Collage, ControlNet, OCSD v3,
  InstanceDiffusion; means by object count) is secondary: differences with CIs and unadjusted tests, labelled
  exploratory. In particular, no claim that OCSD-XL beats GLIGEN, Collage or InstanceDiffusion is tested here.

## Limits

- The naturalness rater is the analyst who developed the settings and who has seen OCSD-XL images of other scenes.
  Sheets are blind, but SDXL images can be recognisable by their look.
- Naturalness is rated on half of the scenes and one seed.
- OCSD-XL runs without M3; the original OCSD uses it.
- Part of any difference to the SD 1.5 methods is the backbone. `xl_controlnet` and `xl_collage` show what the
  backbone gives without M2-M5.
- OCSD-XL has not been run on single-object scenes before (18 of the 72). The code path is tested on CPU
  (`tests/test_xl_paper.py`), not its image quality.

## Cost

About 7 hours on an A100: 576 SDXL images plus M2 objects (about 5 hours), 144 SD 1.5 images (about 20 minutes),
scoring of 1,584 images (about 1.5 to 2 hours). Resumable: finished images and scores are kept, and once all SDXL
images exist the remaining steps also run on an L4 or T4.
