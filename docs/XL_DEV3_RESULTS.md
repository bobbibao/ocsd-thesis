# OCSD-XL development round 3 (`xl_dev_v3`): result and decision

Date: 2026-11-13. Protocol `assets/xl_dev_v3/protocol.json`, code `4baac96`, 9 held-out v2 `_001` scenes x seeds 0, 1,
12 arms completed (216 images, 100 A100 minutes). Raw files: `results/xl_dev_v3/`. These are development numbers. None
of them is a result to report.

## Numbers

N = blind naturalness (0 broken or pasted, 1 fake but readable, 2 natural), rated on shuffled unlabeled sheets before
the key was opened. S = (OPR + mIoU) / 2.

| arm | N | images rated 2 / 0 | OPR | mIoU | S | chamfer | PickScore | s / image |
|---|---|---|---|---|---|---|---|---|
| `ocsd_xl_region` (ablation, no M2 anchoring) | 1.56 | 12 / 2 | 0.703 | 0.459 | 0.581 | 0.046 | 21.96 | 23.6 |
| `ocsd_xl_w04` | 1.22 | 7 / 3 | 0.760 | 0.470 | 0.615 | 0.055 | 21.94 | 25.1 |
| `ocsd_xl_w04_noenergy` | 1.06 | 6 / 5 | 0.769 | 0.488 | 0.628 | 0.053 | 21.93 | 15.2 |
| `ocsd_xl_norepair` (ablation) | 1.00 | 1 / 1 | 0.708 | 0.496 | 0.602 | 0.044 | 21.79 | 10.9 |
| `ocsd_xl_a06` | 0.78 | 1 / 5 | 0.726 | 0.491 | 0.608 | 0.046 | 21.82 | 25.1 |
| `ocsd_xl_noenergy` | 0.67 | 2 / 8 | 0.804 | 0.579 | 0.692 | 0.045 | 21.53 | 15.6 |
| `ocsd_xl` (centre, omega 0.6, eta 0.02) | 0.61 | 1 / 8 | 0.777 | 0.566 | 0.671 | 0.045 | 21.56 | 25.2 |
| `ocsd_xl_a02` | 0.61 | 1 / 8 | 0.770 | 0.553 | 0.662 | 0.046 | 21.42 | 25.2 |
| `ocsd_xl_e05` | 0.61 | 1 / 8 | 0.727 | 0.474 | 0.601 | 0.049 | 21.32 | 24.3 |
| `ocsd_xl_w08_noenergy` | 0.56 | 3 / 11 | 0.832 | 0.644 | 0.738 | 0.036 | 21.25 | 14.4 |
| `xl_controlnet` (baseline) | 0.28 | 0 / 13 | 0.533 | 0.433 | 0.483 | 0.028 | 21.04 | 6.4 |
| `xl_collage` (baseline) | 0.22 | 0 / 14 | 0.760 | 0.586 | 0.673 | 0.041 | 21.02 | 11.0 |
| `ocsd_xl_id` (M3) | failed | | | | | | | |

## Decision under the pre-registered rule

Best candidate S is 0.738 (`ocsd_xl_w08_noenergy`), so an admissible candidate needs N >= 1.0 and S >= 0.708.

- Candidates with N >= 1.0: `ocsd_xl_w04` (S 0.615) and `ocsd_xl_w04_noenergy` (S 0.628). Both miss the S bar.
- Candidates with S >= 0.708: `ocsd_xl_w08_noenergy` only (N 0.56). It misses the N bar.

No candidate is admissible. This is the third round without one, so the SDXL line stops here, as committed before the
round. Nothing in the thesis, paper or slides changes.

## What the three rounds show

1. Naturalness and sketch fidelity trade off through one knob, the scene ControlNet weight omega. With the QuickDraw
   scribbles used here, a high omega forces SDXL to reproduce the doodle shapes (flat cat faces, pasted heads, halos),
   and a low omega gives natural images whose objects drift from the drawn shapes (mIoU 0.47 against 0.64). No setting
   of alpha, eta or repair moved an arm off this frontier.
2. M2 anchoring is the main source of the pasted look on SDXL. The arm without anchoring (`ocsd_xl_region`) is the most
   natural in rounds 2 and 3, and longer anchoring (alpha 0.2) is less natural than shorter (alpha 0.6).
3. The M5 mechanisms do transfer. Every OCSD-XL arm beats the same-backbone ControlNet baseline on OPR by 0.17 to 0.30
   with paired 95% intervals above zero, and the low-omega arms also beat it on naturalness (1.22 against 0.28) and
   PickScore (+0.9, interval above zero). The energy step at eta 0.02 raised naturalness at omega 0.4 (1.22 against
   1.06) and did nothing at omega 0.6; at eta 0.05 it lowered mIoU.
4. Verification and repair (M5d) raises OPR by 0.07 (0.708 to 0.777) and costs naturalness (1.00 to 0.61): repaired
   regions are where most pasted patches come from.
5. M3 does not train on SDXL in this code. After the autocast fix the first loss is finite (0.23 to 0.32) and the other
   199 of 200 steps are non-finite, so the first optimizer step itself produces the overflow (half-precision trainable
   parameters). It needs float32 trainable copies; not pursued because the line stops.

## Limits

One rater (the analyst), 18 images per arm, three scene sets that all come from the same QuickDraw generator. The
naturalness gaps between neighbouring arms (for example 1.22 against 1.06) are within rating noise; the gaps to the
baselines are not. The rule was fixed before round 1 and was not changed after seeing any round.

## A possible follow-up, outside this protocol

`ocsd_xl_w04` is better than the SDXL ControlNet baseline on both axes, which is a different question from the one the
rule asked (one configuration that is both natural and within 0.03 of the best structure score). Testing that claim
would need its own locked protocol on untouched scenes with two declared operating points (omega 0.4 "natural",
omega 0.8 without energy "faithful") against OCSD v1/v2 on SD 1.5. It is not started.
