# Held-out v3: complete OCSD v3 vs original OCSD and baselines

Locked protocol: docs/HELDOUT3_PROTOCOL.md. Per-scene means (seeds averaged); exact sign test; a win = a better.

## Primary: OCSD v3 vs original OCSD (Holm over three metrics)

| a | b | metric | scenes | wins / losses / ties | diff a-b [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| ocsd_v3 | ocsd | opr | 18 | 12 / 3 / 3 | 0.128 [0.023, 0.235] | 0.0352 | 0.0703 |
| ocsd_v3 | ocsd | miou | 18 | 15 / 3 / 0 | 0.195 [0.095, 0.297] | 0.0075 | 0.0226 |
| ocsd_v3 | ocsd | sketch_chamfer | 18 | 11 / 7 / 0 | -0.005 [-0.010, -0.000] | 0.4807 | 0.4807 |

## Key secondary: OCSD v3 vs GLIGEN and Collage (Holm over six)

| a | b | metric | scenes | wins / losses / ties | diff a-b [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| ocsd_v3 | gligen | opr | 18 | 10 / 3 / 5 | 0.093 [-0.002, 0.189] | 0.0923 | 0.3691 |
| ocsd_v3 | gligen | miou | 18 | 14 / 4 / 0 | 0.104 [0.028, 0.184] | 0.0309 | 0.1544 |
| ocsd_v3 | gligen | sketch_chamfer | 18 | 17 / 1 / 0 | -0.016 [-0.024, -0.009] | 0.0001 | 0.0009 |
| ocsd_v3 | collage | opr | 18 | 5 / 7 / 6 | -0.001 [-0.098, 0.102] | 0.7744 | 1.0000 |
| ocsd_v3 | collage | miou | 18 | 9 / 9 / 0 | 0.056 [-0.039, 0.156] | 1.0000 | 1.0000 |
| ocsd_v3 | collage | sketch_chamfer | 18 | 7 / 11 / 0 | 0.002 [-0.004, 0.007] | 0.4807 | 1.0000 |

opr, miou: higher is better; sketch_chamfer: lower is better.

## All arms

18 scenes x seeds 0/1. Means over scenes (seeds averaged) with 95% bootstrap CIs over scenes.
OPR/mIoU (OWLv2) and sketch chamfer (SAM) vs the original OCSD are primary; other metrics, other pairs and the
blind review of review/*.png are secondary.

| method | sketch_chamfer | facing_acc | found_rate | count_acc | opr | oce_c | miou | ra | joint_layout_success | clip | obj_clip | time_s | sampling_work |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ocsd_v3 | 0.050 [0.045, 0.056] | 0.830 [0.689, 0.949] | 1.000 [1.000, 1.000] | 0.306 [0.139, 0.500] | 0.792 [0.691, 0.889] | 2.306 [1.306, 3.500] | 0.670 [0.570, 0.766] | 0.661 [0.507, 0.814] | 0.278 [0.111, 0.472] | 27.667 [26.481, 28.837] | 23.522 [22.655, 24.429] | 26.873 [23.447, 30.477] | 313.111 [273.000, 355.333] |
| direct_box | 0.058 [0.055, 0.062] | 0.551 [0.333, 0.763] | 1.000 [1.000, 1.000] | 0.278 [0.111, 0.472] | 0.768 [0.678, 0.857] | 2.556 [1.472, 3.861] | 0.679 [0.590, 0.771] | 0.637 [0.504, 0.774] | 0.222 [0.056, 0.417] | 28.739 [27.290, 30.220] | 25.157 [24.330, 25.989] | 26.979 [23.565, 30.608] | 313.111 [273.000, 355.333] |
| ocsd | 0.055 [0.050, 0.061] | 0.901 [0.801, 0.981] | 1.000 [1.000, 1.000] | 0.194 [0.056, 0.389] | 0.664 [0.566, 0.761] | 2.417 [1.528, 3.389] | 0.475 [0.396, 0.555] | 0.431 [0.298, 0.578] | 0.139 [0.000, 0.306] | 27.999 [27.175, 28.822] | 23.242 [22.492, 23.983] | 7.725 [6.502, 8.781] | 178.000 [150.000, 202.000] |
| gligen | 0.066 [0.059, 0.073] | 0.497 [0.359, 0.638] | 1.000 [1.000, 1.000] | 0.361 [0.194, 0.556] | 0.700 [0.602, 0.801] | 2.556 [1.528, 3.639] | 0.565 [0.475, 0.658] | 0.553 [0.419, 0.692] | 0.361 [0.194, 0.556] | 27.891 [26.312, 29.358] | 23.624 [22.435, 24.799] | 7.946 [2.769, 18.296] | 60.000 [60.000, 60.000] |
| collage | 0.049 [0.044, 0.053] | 0.894 [0.808, 0.971] | 1.000 [1.000, 1.000] | 0.278 [0.139, 0.417] | 0.794 [0.713, 0.871] | 1.528 [0.972, 2.167] | 0.614 [0.542, 0.687] | 0.633 [0.502, 0.767] | 0.194 [0.083, 0.333] | 30.119 [28.624, 31.545] | 24.009 [23.499, 24.531] | 4.099 [4.019, 4.183] | 144.000 [144.000, 144.000] |
| controlnet | 0.036 [0.031, 0.041] | 0.984 [0.962, 1.000] | 1.000 [1.000, 1.000] | 0.139 [0.000, 0.306] | 0.599 [0.495, 0.707] | 3.056 [2.000, 4.194] | 0.457 [0.360, 0.562] | 0.366 [0.235, 0.516] | 0.111 [0.000, 0.250] | 28.962 [27.840, 30.091] | 22.277 [21.216, 23.370] | 2.836 [2.831, 2.845] | 120.000 [120.000, 120.000] |

## Paired differences (a - b)

| a | b | metric | diff [95% CI] | role |
|---|---|---|---|---|
| ocsd_v3 | direct_box | sketch_chamfer | -0.008 [-0.012, -0.004] | secondary (automatic) |
| ocsd_v3 | direct_box | facing_acc | 0.279 [0.058, 0.510] | secondary (automatic) |
| ocsd_v3 | direct_box | count_acc | 0.028 [-0.139, 0.194] | secondary (automatic) |
| ocsd_v3 | direct_box | opr | 0.025 [-0.053, 0.108] | secondary (automatic) |
| ocsd_v3 | direct_box | oce_c | -0.250 [-1.056, 0.417] | secondary (automatic) |
| ocsd_v3 | direct_box | miou | -0.010 [-0.081, 0.069] | secondary (automatic) |
| ocsd_v3 | direct_box | ra | 0.024 [-0.081, 0.128] | secondary (automatic) |
| ocsd_v3 | direct_box | joint_layout_success | 0.056 [-0.083, 0.222] | secondary (automatic) |
| ocsd_v3 | direct_box | clip | -1.073 [-2.300, -0.033] | secondary (automatic) |
| ocsd_v3 | ocsd | sketch_chamfer | -0.005 [-0.010, -0.000] | secondary (automatic) |
| ocsd_v3 | ocsd | facing_acc | -0.071 [-0.212, 0.038] | secondary (automatic) |
| ocsd_v3 | ocsd | count_acc | 0.111 [-0.167, 0.361] | secondary (automatic) |
| ocsd_v3 | ocsd | opr | 0.128 [0.023, 0.235] | secondary (automatic) |
| ocsd_v3 | ocsd | oce_c | -0.111 [-1.083, 1.000] | secondary (automatic) |
| ocsd_v3 | ocsd | miou | 0.195 [0.095, 0.297] | secondary (automatic) |
| ocsd_v3 | ocsd | ra | 0.230 [0.058, 0.404] | secondary (automatic) |
| ocsd_v3 | ocsd | joint_layout_success | 0.139 [-0.111, 0.389] | secondary (automatic) |
| ocsd_v3 | ocsd | clip | -0.333 [-1.592, 0.947] | secondary (automatic) |
| ocsd_v3 | gligen | sketch_chamfer | -0.016 [-0.024, -0.009] | secondary (automatic) |
| ocsd_v3 | gligen | facing_acc | 0.333 [0.173, 0.478] | secondary (automatic) |
| ocsd_v3 | gligen | count_acc | -0.056 [-0.194, 0.083] | secondary (automatic) |
| ocsd_v3 | gligen | opr | 0.093 [-0.002, 0.189] | secondary (automatic) |
| ocsd_v3 | gligen | oce_c | -0.250 [-1.167, 0.806] | secondary (automatic) |
| ocsd_v3 | gligen | miou | 0.104 [0.028, 0.184] | secondary (automatic) |
| ocsd_v3 | gligen | ra | 0.108 [-0.013, 0.235] | secondary (automatic) |
| ocsd_v3 | gligen | joint_layout_success | -0.083 [-0.222, 0.056] | secondary (automatic) |
| ocsd_v3 | gligen | clip | -0.225 [-1.544, 1.144] | secondary (automatic) |
| ocsd_v3 | collage | sketch_chamfer | 0.002 [-0.004, 0.007] | secondary (automatic) |
| ocsd_v3 | collage | facing_acc | -0.064 [-0.170, 0.026] | secondary (automatic) |
| ocsd_v3 | collage | count_acc | 0.028 [-0.194, 0.250] | secondary (automatic) |
| ocsd_v3 | collage | opr | -0.001 [-0.098, 0.102] | secondary (automatic) |
| ocsd_v3 | collage | oce_c | 0.778 [0.028, 1.583] | secondary (automatic) |
| ocsd_v3 | collage | miou | 0.056 [-0.039, 0.156] | secondary (automatic) |
| ocsd_v3 | collage | ra | 0.028 [-0.135, 0.202] | secondary (automatic) |
| ocsd_v3 | collage | joint_layout_success | 0.083 [-0.139, 0.278] | secondary (automatic) |
| ocsd_v3 | collage | clip | -2.452 [-3.984, -0.941] | secondary (automatic) |
| ocsd_v3 | controlnet | sketch_chamfer | 0.014 [0.008, 0.020] | secondary (automatic) |
| ocsd_v3 | controlnet | facing_acc | -0.154 [-0.298, -0.038] | secondary (automatic) |
| ocsd_v3 | controlnet | count_acc | 0.167 [-0.056, 0.389] | secondary (automatic) |
| ocsd_v3 | controlnet | opr | 0.193 [0.106, 0.288] | secondary (automatic) |
| ocsd_v3 | controlnet | oce_c | -0.750 [-1.611, 0.194] | secondary (automatic) |
| ocsd_v3 | controlnet | miou | 0.212 [0.137, 0.298] | secondary (automatic) |
| ocsd_v3 | controlnet | ra | 0.294 [0.163, 0.435] | secondary (automatic) |
| ocsd_v3 | controlnet | joint_layout_success | 0.167 [-0.028, 0.389] | secondary (automatic) |
| ocsd_v3 | controlnet | clip | -1.296 [-2.747, 0.258] | secondary (automatic) |
| direct_box | ocsd | sketch_chamfer | 0.003 [-0.002, 0.008] | secondary (automatic) |
| direct_box | ocsd | facing_acc | -0.349 [-0.561, -0.154] | secondary (automatic) |
| direct_box | ocsd | count_acc | 0.083 [-0.194, 0.361] | secondary (automatic) |
| direct_box | ocsd | opr | 0.103 [0.002, 0.203] | secondary (automatic) |
| direct_box | ocsd | oce_c | 0.139 [-0.833, 1.139] | secondary (automatic) |
| direct_box | ocsd | miou | 0.205 [0.123, 0.284] | secondary (automatic) |
| direct_box | ocsd | ra | 0.206 [0.033, 0.373] | secondary (automatic) |
| direct_box | ocsd | joint_layout_success | 0.083 [-0.139, 0.333] | secondary (automatic) |
| direct_box | ocsd | clip | 0.740 [-0.670, 2.171] | secondary (automatic) |
| direct_box | gligen | sketch_chamfer | -0.008 [-0.015, -0.003] | secondary (automatic) |
| direct_box | gligen | facing_acc | 0.054 [-0.192, 0.269] | secondary (automatic) |
| direct_box | gligen | count_acc | -0.083 [-0.194, 0.028] | secondary (automatic) |
| direct_box | gligen | opr | 0.068 [0.008, 0.129] | secondary (automatic) |
| direct_box | gligen | oce_c | 0.000 [-0.694, 0.750] | secondary (automatic) |
| direct_box | gligen | miou | 0.114 [0.065, 0.169] | secondary (automatic) |
| direct_box | gligen | ra | 0.084 [0.000, 0.176] | secondary (automatic) |
| direct_box | gligen | joint_layout_success | -0.139 [-0.250, -0.056] | secondary (automatic) |
| direct_box | gligen | clip | 0.848 [-0.364, 2.089] | secondary (automatic) |
