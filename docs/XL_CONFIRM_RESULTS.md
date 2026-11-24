# OCSD-XL confirmation on held-out v3 (2026-11-17): result

**Verdict under the locked rules: C-natural holds, C-faithful does not.** The natural setting of OCSD-XL looks more
natural than the original OCSD (Holm p = 0.0002) and also keeps more objects (Holm p = 0.019). The faithful setting has
higher OPR and mIoU than the original OCSD with confidence intervals above zero, but its sign tests do not survive
correction (Holm p = 0.19), so no claim is made for it.

Protocol: [XL_CONFIRM_PROTOCOL.md](XL_CONFIRM_PROTOCOL.md), locked in `assets/xl_confirm_v1/protocol.json` at commit
`dfa9ce6` before any image. Evidence: `results/xl_confirm_v1/` (tables, key, blind ratings, `analysis.md`).

## Run audit

- Code `dfa9ce6`, A100. 144 of 144 SDXL images in 72 minutes, 0 failures. 36 of 36 OCSD-v2 images, 0 failures.
- 216 of 216 images of the 2026-11-05 held-out v3 run were found on Drive with matching checksums. Scored again here,
  their OPR, mIoU and chamfer equal the values in [HELDOUT3_20261105_RESULTS.md](HELDOUT3_20261105_RESULTS.md) to three
  decimals (for example original OCSD 0.664 / 0.475 / 0.055), so the two runs share one scale.
- Blind ratings: all 36 sheets, 11 panels each, written before the key or any table was opened. File
  `blind_naturalness_ratings.json`, sha256 `5bcf25fe…a351d`.

## Primary family (18 scenes, exact sign test, Holm over four)

| id | comparison | measure | wins / losses / ties | difference [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|
| P1 | faithful vs original OCSD | OPR | 11 / 4 / 3 | +0.145 [0.038, 0.256] | 0.119 | 0.19 |
| P2 | faithful vs original OCSD | mIoU | 13 / 5 / 0 | +0.154 [0.056, 0.255] | 0.096 | 0.19 |
| **P3** | **natural vs original OCSD** | **naturalness** | **15 / 0 / 3** | **+0.78 [0.56, 0.97]** | 0.0001 | **0.0002** |
| **P4** | **natural vs original OCSD** | **OPR** | **11 / 1 / 6** | **+0.128 [0.053, 0.206]** | 0.006 | **0.019** |

## All arms (means over 18 scenes, 36 images each)

| arm | OPR | mIoU | chamfer | naturalness (0-2) | images rated 2 / 0 | PickScore | count acc. | relations | s / image |
|---|---|---|---|---|---|---|---|---|---|
| **OCSD-XL natural** | 0.792 | 0.511 | 0.058 | 0.92 | 7 / 10 | **22.06** | 0.25 | 0.64 | 20.6 |
| **OCSD-XL faithful** | 0.809 | 0.629 | 0.039 | 0.61 | 4 / 18 | 21.35 | 0.39 | 0.67 | 11.8 |
| original OCSD (v1, SD 1.5) | 0.664 | 0.475 | 0.055 | 0.14 | 0 / 31 | 20.91 | 0.19 | 0.43 | 7.7 |
| OCSD-v2 (SD 1.5) | 0.717 | 0.396 | 0.069 | 1.00 | 8 / 8 | 21.64 | 0.22 | 0.51 | 7.8 |
| SDXL ControlNet | 0.498 | 0.394 | 0.029 | 0.36 | 1 / 24 | 21.12 | 0.06 | 0.30 | 6.1 |
| SDXL Collage | 0.814 | 0.619 | 0.052 | 0.44 | 2 / 22 | 21.30 | 0.36 | 0.67 | 10.6 |
| GLIGEN | 0.700 | 0.565 | 0.066 | 1.50 | 22 / 4 | 21.25 | 0.36 | 0.55 | 7.9 |
| Collage (SD 1.5) | 0.794 | 0.614 | 0.049 | 0.31 | 0 / 25 | 21.32 | 0.28 | 0.63 | 4.1 |
| ControlNet (SD 1.5) | 0.599 | 0.457 | 0.036 | 0.72 | 5 / 15 | 21.40 | 0.14 | 0.37 | 2.8 |
| OCSD v3 (InstanceDiffusion core) | 0.792 | 0.670 | 0.050 | 1.36 | 14 / 1 | 21.72 | 0.31 | 0.66 | 26.9 |
| InstanceDiffusion boxes | 0.768 | 0.679 | 0.058 | 1.42 | 15 / 0 | 21.58 | 0.28 | 0.64 | 27.0 |

## Secondary (exploratory, unadjusted; full table in `results/xl_confirm_v1/analysis.md`)

- **Against OCSD-v2.** Natural: same naturalness (-0.08 [-0.36, 0.22]), mIoU +0.115 [0.03, 0.20], chamfer better
  (-0.011), OPR +0.08 (interval includes zero). Faithful: mIoU +0.233 (16 / 2, p 0.001), chamfer better in 18 of 18
  scenes, naturalness lower (-0.39).
- **Against the same backbone without M2-M5 (SDXL ControlNet).** Natural: OPR +0.29 (15 / 1), naturalness +0.56,
  PickScore +0.94. Faithful: OPR +0.31 (17 / 0), mIoU +0.24 (17 / 1). The gain is not the backbone alone.
- **Against GLIGEN and OCSD v3.** Both OCSD-XL settings are less natural (natural: -0.58 and -0.44; faithful: -0.89 and
  -0.75). OPR is level with OCSD v3 and above GLIGEN by 0.09 to 0.11 (intervals include zero). The faithful setting
  follows the drawn shapes better than both (chamfer 18 / 0 against GLIGEN, 16 / 2 against OCSD v3).
- **Natural against faithful.** The trade-off found in development repeats on new scenes: natural has higher
  PickScore (17 / 1) and naturalness (+0.31), faithful has higher mIoU (15 / 3) and lower chamfer (17 / 1).

## What can and cannot be said

Can be said: on 18 untouched scenes, running the unchanged M1-M5 pipeline on SDXL at scene ControlNet weight 0.4 gives
images that are more natural than those of the original OCSD and keep more of the drawn objects. This is the first
OCSD configuration in the project that improves on the original method in a pre-registered test without replacing
M2-M5.

Cannot be said: that OCSD-XL is the most natural method (GLIGEN, InstanceDiffusion and OCSD v3 are rated higher), that
the faithful setting is better than the original OCSD (direction and size are favourable, the test is not), or that
one setting wins on both naturalness and shape. A naturalness of 0.92 means the typical image is still "fake but
readable": 10 of 36 images were rated broken or pasted.

## Limits

One rater, who is also the developer; sheets were blind and shuffled, but SDXL images can be recognisable. 18 scenes.
OCSD-XL runs without M3. The scenes are synthetic QuickDraw compositions. The 72 paper scenes have not been run.
