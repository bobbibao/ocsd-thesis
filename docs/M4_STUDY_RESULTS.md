# OCSD with harmonised M4 (`m4_study_v1`): results

Run: one Colab Run all (A100), frozen 2026-11-24 23:32, evaluated 2026-11-25 03:42. Protocol: `docs/M4_STUDY_PROTOCOL.md`.
Files: `results/m4_study_v1/` (summary, tables, selection, lock, blind ratings and the review key).

## 1. Frozen configuration (development split only)

The combined arm won the pre-registered rule (`SELECTION.md`): `ocsd_m4_combo` = harmonised M4 with
alpha 0.2 (anchor released by 80% of the steps), release over 40% of the steps, scene ControlNet omega 0.6 and the M3
LoRA only from 45% of the steps. M1, M2, M3 and M5 are the thesis modules. This setting is `ocsd_final` (= OCSD from
now on). Development numbers are not results.

## 2. Paper benchmark (72 scenes x seeds 0/1, 144/144 images for every arm)

| method | OPR | mIoU | RA | count acc | chamfer (lower better) | CLIP | PickScore |
|---|---|---|---|---|---|---|---|
| **OCSD (harmonised M4)** | 0.785 | 0.484 | 0.508 | 0.333 | 0.061 | 28.69 | 21.89 |
| OCSD (thesis M4) | 0.711 | 0.529 | 0.445 | 0.333 | 0.055 | 28.11 | 21.16 |
| OCSD-v2 | 0.744 | 0.408 | 0.455 | 0.361 | 0.066 | 28.87 | 22.18 |
| GLIGEN | 0.801 | 0.653 | 0.563 | 0.465 | 0.066 | 28.23 | 21.41 |
| Collage | 0.824 | 0.607 | 0.616 | 0.451 | 0.049 | 28.80 | 21.52 |
| ControlNet | 0.655 | 0.492 | 0.321 | 0.257 | 0.038 | 27.84 | 21.67 |
| T2I-Adapter | 0.624 | 0.437 | 0.282 | 0.312 | 0.035 | 28.24 | 21.53 |
| Zhang et al. (2025) | 0.633 | 0.435 | 0.337 | 0.278 | 0.059 | 25.55 | 20.73 |

95% CIs: `results/m4_study_v1/summary.md`.

## 3. Primary tests (new OCSD vs thesis OCSD, sign test, Holm over P1-P4)

| test | diff [95% CI] | W/L/T | p Holm |
|---|---|---|---|
| P1 OPR | +0.074 [+0.023, +0.129] | 32/12/28 | **0.011** |
| P2 mIoU | -0.044 [-0.091, +0.002] | 30/42/0 | 0.27 (n.s.) |
| P3 RA | +0.063 [-0.023, +0.145] | 28/17/9 | 0.27 (n.s.) |
| P4 PickScore | +0.725 [+0.592, +0.861] | 65/7/0 | **<0.0001** |

Both pre-registered claims hold: **C-structure** (P1 significant, nothing significantly worse) and **C-natural**
(P4 significant, OPR and mIoU not significantly worse). mIoU is lower on average but not significantly.

## 4. Secondary comparisons (Holm over the secondary family)

Significantly better than the baselines:
- ControlNet: OPR, RA.
- T2I-Adapter: OPR, RA.
- Zhang et al. (2025): OPR, RA, PickScore, CLIP.
- OCSD-v2: mIoU.
- GLIGEN and Collage: PickScore.

Significantly worse:
- GLIGEN: mIoU.
- Collage: mIoU and sketch chamfer.
- ControlNet and T2I-Adapter: sketch chamfer.
- OCSD-v2: PickScore.

Not significantly different:
- GLIGEN on OPR (0.785 vs 0.801), RA and count.
- Collage on OPR (0.785 vs 0.824), RA (Holm p 0.065) and count.

**There is no claim that OCSD beats GLIGEN or Collage on structure.** The supported claim is: OCSD reaches the
object presence of GLIGEN and Collage (no significant difference in OPR), it is significantly more natural than both
by PickScore and by blind rating, and it beats ControlNet, T2I-Adapter and Zhang et al. on OPR and RA.

By object count (`tables/by_count_bin.csv`), OCSD has OPR 1.0 at one object. At 8+ objects it is second with 0.625,
behind Collage at 0.686 and ahead of GLIGEN at 0.552. It also has the second-best PickScore in every count bin, after
OCSD-v2.

## 5. Blind naturalness rating (secondary)

Analyst ratings: 36 sheets, scenes 003-005, seed 0, 8 shuffled arms. Scale: 0 = broken or pasted, 1 = fake but
readable, 2 = natural. All ratings were written before the key was opened (`naturalness_blind_ratings.txt`).

| method | mean | #2 | #0 |
|---|---|---|---|
| GLIGEN | 1.81 | 30 | 1 |
| **OCSD (harmonised M4)** | **1.36** | 16 | 3 |
| T2I-Adapter | 1.19 | 10 | 3 |
| ControlNet | 1.17 | 10 | 4 |
| OCSD-v2 | 1.17 | 11 | 5 |
| Collage | 0.75 | 2 | 11 |
| Zhang et al. (2025) | 0.47 | 2 | 21 |
| OCSD (thesis M4) | 0.44 | 0 | 20 |

Sign tests of the new OCSD against each arm (Holm over 7):

| comparison | W/L | p | significant after Holm? |
|---|---|---|---|
| vs thesis OCSD | 25/0 | 6e-8 | yes, better |
| vs Collage | 20/0 | 2e-6 | yes, better |
| vs Zhang et al. | 26/2 | 3e-6 | yes, better |
| vs GLIGEN | 3/18 | 0.0015 | yes, **worse** |
| vs ControlNet | 16/8 | 0.15 | no |
| vs T2I-Adapter | 11/7 | 0.48 | no |
| vs OCSD-v2 | 9/2 | 0.065 | no |

The halos and pasted look of the thesis M4 are mostly gone. On a single rater, GLIGEN still looks more natural.
Many-object scenes (8+) remain the hardest for every method.

## 6. Limits

- One rater (the analyst), not a user study.
- QuickDraw-Scenes only.
- PickScore is a proxy.
- mIoU dropped against the thesis OCSD (not significant), because the released objects are redrawn and shift
  slightly from the sketch masks. Sketch chamfer is also slightly worse (0.061 vs 0.055).
