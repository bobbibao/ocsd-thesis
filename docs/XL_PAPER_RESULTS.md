# OCSD-XL on the paper benchmark (2026-11-17): result

**Verdict under the locked rules: both claims hold on the 72 paper scenes.** The faithful setting of OCSD-XL keeps
objects better than the original OCSD (OPR Holm p = 0.0003, mIoU Holm p = 0.0005). The natural setting looks more
natural than the original OCSD (Holm p = 0.0001) and also keeps more objects (Holm p = 0.0003).

On held-out v3 ([result](XL_CONFIRM_RESULTS.md)), C-natural held and C-faithful did not reach significance with 18
scenes. The two sets are reported side by side and are not pooled (table below).

Protocol: [XL_PAPER_PROTOCOL.md](XL_PAPER_PROTOCOL.md), locked in `assets/xl_paper_v1/protocol.json` at commit
`90cb976` before any image. Evidence: `results/xl_paper_v1/` (tables, key, blind ratings, `analysis.md`).

## Run audit

- Code `90cb976`, A100. 576 of 576 SDXL images in 249 minutes, 0 failures, no arm switched off. 144 of 144 OCSD-v2
  images, 0 failures.
- Scenes: all 72 copied from the paper benchmark on Drive matched the expected list, and their dataset identity
  (`bcd46251…`) equals the one recorded by the 2026-11-05 paper benchmark run: same files.
- 864 of 864 images of the 2026-11-05 run were found on Drive with matching checksums. Scored again here, 862 of 864
  have the same OPR as in `docs/paper3_20261105/per_image_*.csv` (two images differ by one detected object), and the
  arm means are equal to three decimals except original OCSD (0.713 then, 0.711 now) and GLIGEN mIoU (0.652, 0.653).
- Blind ratings: 36 sheets (scenes numbered 003 to 005, seed 0, as locked), 11 panels each, written before the key or
  any table was opened. File `blind_naturalness_ratings.json`, sha256 `0cbd3293…19bb6b`.

## Primary family (exact sign test over scenes, Holm over four)

OPR and mIoU over 72 scenes (seeds averaged); naturalness over the 36 rated scenes.

| id | comparison | measure | wins / losses / ties | difference [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|
| **P1** | **faithful vs original OCSD** | **OPR** | **35 / 9 / 28** | **+0.113 [0.056, 0.172]** | 0.0001 | **0.0003** |
| **P2** | **faithful vs original OCSD** | **mIoU** | **51 / 21 / 0** | **+0.094 [0.034, 0.153]** | 0.0005 | **0.0005** |
| **P3** | **natural vs original OCSD** | **naturalness** | **23 / 2 / 11** | **+0.86 [0.58, 1.14]** | < 0.0001 | **0.0001** |
| **P4** | **natural vs original OCSD** | **OPR** | **34 / 9 / 29** | **+0.108 [0.060, 0.158]** | 0.0002 | **0.0003** |

## Both test sets side by side (not pooled)

| claim | held-out v3 (18 scenes) | paper benchmark (72 scenes) |
|---|---|---|
| C-faithful: OPR / mIoU vs original OCSD | +0.145 / +0.154, Holm p 0.19 / 0.19: **not significant** | +0.113 / +0.094, Holm p 0.0003 / 0.0005: **holds** |
| C-natural: naturalness / OPR vs original OCSD | +0.78 / +0.128, Holm p 0.0002 / 0.019: **holds** | +0.86 / +0.108, Holm p 0.0001 / 0.0003: **holds** |

The effect sizes agree across the two sets; the faithful claim failed on held-out v3 for lack of scenes, not because
the direction changed.

## All arms (means over 72 scenes, 144 images each; naturalness over 36 rated images)

| arm | OPR | mIoU | chamfer | naturalness (0-2) | images rated 2 / 0 | PickScore | count acc. | relations | s / image |
|---|---|---|---|---|---|---|---|---|---|
| **OCSD-XL natural** | 0.819 | 0.519 | 0.054 | 1.08 | 13 / 10 | **22.49** | 0.37 | 0.60 | 18.9 |
| **OCSD-XL faithful** | 0.825 | 0.623 | 0.041 | 0.75 | 7 / 16 | 21.77 | 0.41 | 0.61 | 10.9 |
| original OCSD (v1, SD 1.5) | 0.711 | 0.529 | 0.055 | 0.22 | 1 / 29 | 21.16 | 0.33 | 0.45 | 6.7 |
| OCSD-v2 (SD 1.5) | 0.744 | 0.408 | 0.066 | 0.89 | 7 / 11 | 22.18 | 0.36 | 0.46 | 7.1 |
| SDXL ControlNet | 0.580 | 0.436 | 0.028 | 0.61 | 3 / 17 | 21.37 | 0.25 | 0.27 | 6.2 |
| SDXL Collage | 0.805 | 0.610 | 0.048 | 0.42 | 3 / 24 | 21.53 | 0.40 | 0.57 | 10.5 |
| GLIGEN | 0.801 | 0.653 | 0.066 | 1.44 | 17 / 1 | 21.41 | 0.47 | 0.56 | 2.9 |
| Collage (SD 1.5) | 0.824 | 0.607 | 0.049 | 0.28 | 2 / 28 | 21.52 | 0.45 | 0.62 | 4.0 |
| ControlNet (SD 1.5) | 0.655 | 0.492 | 0.038 | 0.75 | 3 / 12 | 21.67 | 0.26 | 0.32 | 2.8 |
| OCSD v3 (InstanceDiffusion core) | 0.812 | 0.682 | 0.050 | 1.69 | 25 / 0 | 22.04 | 0.38 | 0.59 | 23.6 |
| InstanceDiffusion boxes | 0.829 | 0.730 | 0.060 | 1.69 | 25 / 0 | 21.88 | 0.45 | 0.61 | 23.6 |

## Secondary (exploratory, unadjusted; full table in `results/xl_paper_v1/analysis.md`)

- **Against OCSD-v2.** Both settings keep more objects (OPR +0.075 natural, +0.081 faithful) and fit the layout better
  (mIoU +0.111, +0.215). Naturalness is level (natural +0.19 [-0.11, 0.50]; faithful -0.14 [-0.44, 0.17]). PickScore:
  natural higher (+0.30), faithful lower (-0.41).
- **Against the same backbone without M2-M5 (SDXL ControlNet).** OPR +0.24 for both settings (51 / 2 and 54 / 3);
  natural is also more natural (+0.47). The gain is not the backbone alone.
- **Against GLIGEN, Collage, OCSD v3 and InstanceDiffusion.** OPR is level with all four (every interval includes
  zero). The faithful setting follows the drawn shapes better than all four (chamfer 0.041 against 0.049 to 0.066;
  63 / 9 against GLIGEN, 56 / 16 against OCSD v3) and is level with GLIGEN and Collage on mIoU, below OCSD v3 and
  InstanceDiffusion. Both settings are less natural than GLIGEN, OCSD v3 and InstanceDiffusion (natural -0.36 to
  -0.61; faithful -0.69 to -0.94) and more natural than Collage. The natural setting has the highest PickScore of all
  eleven arms (+0.45 over OCSD v3, +1.08 over GLIGEN).
- **By object count.** The OCSD-XL gain grows with the number of objects. With 8 or more objects: OPR 0.680 natural /
  0.689 faithful against 0.515 original OCSD, 0.552 GLIGEN, 0.572 OCSD v3, 0.686 Collage; mIoU of the faithful setting
  0.546, the highest of all arms (GLIGEN 0.411, OCSD v3 0.442, Collage 0.501). Naturalness falls with object count for
  both settings (natural 1.67 with one object, 0.44 with 8+), while GLIGEN and OCSD v3 stay above 1.2.
- **Natural against faithful.** The trade-off seen in development and on held-out v3 repeats: natural is more natural
  and has higher PickScore, faithful has higher mIoU and lower chamfer, OPR is the same.

## What can and cannot be said

Can be said: the unchanged M1-M5 pipeline (M3 off) on SDXL is an improved OCSD. On the paper benchmark, at scene
ControlNet weight 0.8 it keeps objects and layout significantly better than the original OCSD; at weight 0.4 it is
significantly more natural and still keeps more objects. Both results are pre-registered; the natural one is
confirmed on two scene sets, the faithful one on the larger set only. Most of the gain is not the backbone: SDXL
ControlNet alone is far below both settings.

Cannot be said: that OCSD-XL is the best method overall. GLIGEN, OCSD v3 and InstanceDiffusion produce more natural
images, and OCSD v3 and InstanceDiffusion fit the layout better. OCSD-XL is level with the strongest baselines on
object preservation, best on sketch-shape fidelity (faithful), best on PickScore (natural) and best on scenes with
many objects; none of these comparisons were primary tests.

## Limits

- One naturalness rater, who also developed the settings; sheets were blind and shuffled, but SDXL images can be
  recognisable by their look. Naturalness covers half of the scenes and one seed.
- OCSD-XL runs without M3 (identity learning does not train on SDXL in this code); the original OCSD uses it.
- The 72 scenes have been used before to report the SD 1.5 methods and OCSD v3; no OCSD-XL setting was chosen on them.
- One setting cannot have both advantages: the user picks the scene ControlNet weight (0.4 natural, 0.8 faithful).
