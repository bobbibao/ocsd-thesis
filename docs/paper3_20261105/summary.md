# Paper benchmark v3: frozen OCSD v3 vs original OCSD and baselines (72 paper_v2 scenes)

Locked protocol: docs/PAPER3_PROTOCOL.md. Per-scene means (seeds averaged); exact sign test; a win = a better.

## Primary: OCSD v3 vs original OCSD (Holm over three metrics)

| a | b | metric | scenes | wins / losses / ties | diff a-b [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| ocsd_v3 | ocsd | opr | 72 | 35 / 15 / 22 | 0.100 [0.047, 0.153] | 0.0066 | 0.0132 |
| ocsd_v3 | ocsd | miou | 72 | 59 / 13 / 0 | 0.152 [0.109, 0.196] | 0.0000 | 0.0000 |
| ocsd_v3 | ocsd | sketch_chamfer | 72 | 38 / 34 / 0 | -0.005 [-0.009, -0.001] | 0.7239 | 0.7239 |

## Key secondary: OCSD v3 vs GLIGEN and Collage (Holm over six)

| a | b | metric | scenes | wins / losses / ties | diff a-b [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| ocsd_v3 | gligen | opr | 72 | 21 / 15 / 36 | 0.011 [-0.021, 0.044] | 0.4050 | 1.0000 |
| ocsd_v3 | gligen | miou | 72 | 49 / 23 / 0 | 0.030 [0.001, 0.058] | 0.0029 | 0.0147 |
| ocsd_v3 | gligen | sketch_chamfer | 72 | 65 / 7 / 0 | -0.016 [-0.020, -0.013] | 0.0000 | 0.0000 |
| ocsd_v3 | collage | opr | 72 | 18 / 23 / 31 | -0.012 [-0.058, 0.035] | 0.5327 | 1.0000 |
| ocsd_v3 | collage | miou | 72 | 47 / 25 / 0 | 0.075 [0.030, 0.120] | 0.0128 | 0.0511 |
| ocsd_v3 | collage | sketch_chamfer | 72 | 34 / 38 / 0 | 0.000 [-0.003, 0.004] | 0.7239 | 1.0000 |

opr, miou: higher is better; sketch_chamfer: lower is better.

## All arms

72 scenes x seeds 0/1. Means over scenes (seeds averaged) with 95% bootstrap CIs over scenes.
OPR/mIoU (OWLv2) and sketch chamfer (SAM) vs the original OCSD are primary; other metrics, other pairs and the
blind review of review/*.png are secondary.

| method | sketch_chamfer | facing_acc | found_rate | count_acc | opr | oce_c | miou | ra | joint_layout_success | clip | obj_clip | time_s | sampling_work |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ocsd_v3 | 0.050 [0.047, 0.053] | 0.822 [0.741, 0.893] | 1.000 [1.000, 1.000] | 0.382 [0.285, 0.479] | 0.812 [0.766, 0.856] | 2.160 [1.500, 2.972] | 0.682 [0.636, 0.728] | 0.589 [0.514, 0.664] | 0.368 [0.271, 0.465] | 28.088 [27.405, 28.780] | 24.728 [24.216, 25.249] | 23.593 [21.432, 25.887] | 273.000 [247.667, 299.917] |
| direct_box | 0.060 [0.058, 0.063] | 0.512 [0.423, 0.600] | 1.000 [1.000, 1.000] | 0.451 [0.354, 0.556] | 0.829 [0.786, 0.870] | 1.938 [1.243, 2.812] | 0.730 [0.685, 0.774] | 0.607 [0.535, 0.681] | 0.424 [0.326, 0.521] | 28.128 [27.404, 28.868] | 26.146 [25.603, 26.701] | 23.589 [21.456, 25.863] | 273.000 [247.667, 299.917] |
| ocsd | 0.055 [0.050, 0.059] | 0.916 [0.862, 0.963] | 1.000 [1.000, 1.000] | 0.333 [0.229, 0.438] | 0.712 [0.649, 0.772] | 2.188 [1.653, 2.771] | 0.530 [0.476, 0.584] | 0.447 [0.367, 0.530] | 0.271 [0.174, 0.375] | 28.112 [27.247, 28.959] | 24.009 [23.616, 24.397] | 6.681 [6.005, 7.342] | 152.000 [136.500, 167.000] |
| gligen | 0.066 [0.063, 0.069] | 0.509 [0.418, 0.599] | 1.000 [1.000, 1.000] | 0.472 [0.368, 0.576] | 0.801 [0.749, 0.850] | 2.062 [1.451, 2.757] | 0.652 [0.604, 0.698] | 0.562 [0.478, 0.647] | 0.451 [0.347, 0.556] | 28.227 [27.389, 29.091] | 24.598 [23.943, 25.263] | 2.925 [2.782, 3.209] | 60.000 [60.000, 60.000] |
| collage | 0.049 [0.045, 0.053] | 0.902 [0.845, 0.953] | 1.000 [1.000, 1.000] | 0.451 [0.354, 0.549] | 0.824 [0.775, 0.869] | 1.500 [1.139, 1.903] | 0.607 [0.566, 0.648] | 0.616 [0.532, 0.699] | 0.319 [0.229, 0.417] | 28.800 [27.924, 29.647] | 24.778 [24.294, 25.271] | 4.013 [3.959, 4.071] | 144.000 [144.000, 144.000] |
| controlnet | 0.038 [0.035, 0.041] | 0.977 [0.948, 0.997] | 1.000 [1.000, 1.000] | 0.257 [0.174, 0.347] | 0.655 [0.588, 0.719] | 2.514 [1.951, 3.139] | 0.492 [0.435, 0.549] | 0.321 [0.249, 0.393] | 0.222 [0.146, 0.306] | 27.843 [27.105, 28.569] | 23.248 [22.596, 23.905] | 2.845 [2.843, 2.848] | 120.000 [120.000, 120.000] |

## Paired differences (a - b)

| a | b | metric | diff [95% CI] | role |
|---|---|---|---|---|
| ocsd_v3 | direct_box | sketch_chamfer | -0.011 [-0.013, -0.008] | secondary (automatic) |
| ocsd_v3 | direct_box | facing_acc | 0.311 [0.196, 0.427] | secondary (automatic) |
| ocsd_v3 | direct_box | count_acc | -0.069 [-0.153, 0.014] | secondary (automatic) |
| ocsd_v3 | direct_box | opr | -0.017 [-0.047, 0.013] | secondary (automatic) |
| ocsd_v3 | direct_box | oce_c | 0.222 [-0.062, 0.493] | secondary (automatic) |
| ocsd_v3 | direct_box | miou | -0.048 [-0.074, -0.021] | secondary (automatic) |
| ocsd_v3 | direct_box | ra | -0.018 [-0.072, 0.037] | secondary (automatic) |
| ocsd_v3 | direct_box | joint_layout_success | -0.056 [-0.132, 0.021] | secondary (automatic) |
| ocsd_v3 | direct_box | clip | -0.040 [-0.603, 0.521] | secondary (automatic) |
| ocsd_v3 | ocsd | sketch_chamfer | -0.005 [-0.009, -0.001] | secondary (automatic) |
| ocsd_v3 | ocsd | facing_acc | -0.094 [-0.172, -0.020] | secondary (automatic) |
| ocsd_v3 | ocsd | count_acc | 0.049 [-0.056, 0.146] | secondary (automatic) |
| ocsd_v3 | ocsd | opr | 0.100 [0.047, 0.153] | secondary (automatic) |
| ocsd_v3 | ocsd | oce_c | -0.028 [-0.611, 0.743] | secondary (automatic) |
| ocsd_v3 | ocsd | miou | 0.152 [0.109, 0.196] | secondary (automatic) |
| ocsd_v3 | ocsd | ra | 0.142 [0.054, 0.228] | secondary (automatic) |
| ocsd_v3 | ocsd | joint_layout_success | 0.097 [0.007, 0.181] | secondary (automatic) |
| ocsd_v3 | ocsd | clip | -0.024 [-0.872, 0.846] | secondary (automatic) |
| ocsd_v3 | gligen | sketch_chamfer | -0.016 [-0.020, -0.013] | secondary (automatic) |
| ocsd_v3 | gligen | facing_acc | 0.314 [0.213, 0.414] | secondary (automatic) |
| ocsd_v3 | gligen | count_acc | -0.090 [-0.181, -0.007] | secondary (automatic) |
| ocsd_v3 | gligen | opr | 0.011 [-0.021, 0.044] | secondary (automatic) |
| ocsd_v3 | gligen | oce_c | 0.097 [-0.444, 0.764] | secondary (automatic) |
| ocsd_v3 | gligen | miou | 0.030 [0.001, 0.058] | secondary (automatic) |
| ocsd_v3 | gligen | ra | 0.027 [-0.036, 0.088] | secondary (automatic) |
| ocsd_v3 | gligen | joint_layout_success | -0.083 [-0.167, -0.007] | secondary (automatic) |
| ocsd_v3 | gligen | clip | -0.139 [-0.849, 0.630] | secondary (automatic) |
| ocsd_v3 | collage | sketch_chamfer | 0.000 [-0.003, 0.004] | secondary (automatic) |
| ocsd_v3 | collage | facing_acc | -0.079 [-0.166, 0.005] | secondary (automatic) |
| ocsd_v3 | collage | count_acc | -0.069 [-0.167, 0.028] | secondary (automatic) |
| ocsd_v3 | collage | opr | -0.012 [-0.058, 0.035] | secondary (automatic) |
| ocsd_v3 | collage | oce_c | 0.660 [0.118, 1.403] | secondary (automatic) |
| ocsd_v3 | collage | miou | 0.075 [0.030, 0.120] | secondary (automatic) |
| ocsd_v3 | collage | ra | -0.027 [-0.109, 0.054] | secondary (automatic) |
| ocsd_v3 | collage | joint_layout_success | 0.049 [-0.035, 0.132] | secondary (automatic) |
| ocsd_v3 | collage | clip | -0.711 [-1.360, -0.029] | secondary (automatic) |
| ocsd_v3 | controlnet | sketch_chamfer | 0.012 [0.009, 0.015] | secondary (automatic) |
| ocsd_v3 | controlnet | facing_acc | -0.154 [-0.240, -0.076] | secondary (automatic) |
| ocsd_v3 | controlnet | count_acc | 0.125 [0.042, 0.208] | secondary (automatic) |
| ocsd_v3 | controlnet | opr | 0.157 [0.102, 0.213] | secondary (automatic) |
| ocsd_v3 | controlnet | oce_c | -0.354 [-0.924, 0.319] | secondary (automatic) |
| ocsd_v3 | controlnet | miou | 0.190 [0.142, 0.238] | secondary (automatic) |
| ocsd_v3 | controlnet | ra | 0.269 [0.185, 0.352] | secondary (automatic) |
| ocsd_v3 | controlnet | joint_layout_success | 0.146 [0.069, 0.222] | secondary (automatic) |
| ocsd_v3 | controlnet | clip | 0.246 [-0.421, 0.975] | secondary (automatic) |
| direct_box | ocsd | sketch_chamfer | 0.006 [0.001, 0.010] | secondary (automatic) |
| direct_box | ocsd | facing_acc | -0.404 [-0.511, -0.299] | secondary (automatic) |
| direct_box | ocsd | count_acc | 0.118 [0.007, 0.222] | secondary (automatic) |
| direct_box | ocsd | opr | 0.116 [0.061, 0.175] | secondary (automatic) |
| direct_box | ocsd | oce_c | -0.250 [-0.896, 0.577] | secondary (automatic) |
| direct_box | ocsd | miou | 0.200 [0.154, 0.247] | secondary (automatic) |
| direct_box | ocsd | ra | 0.160 [0.072, 0.246] | secondary (automatic) |
| direct_box | ocsd | joint_layout_success | 0.153 [0.056, 0.250] | secondary (automatic) |
| direct_box | ocsd | clip | 0.016 [-0.862, 0.891] | secondary (automatic) |
| direct_box | gligen | sketch_chamfer | -0.006 [-0.008, -0.003] | secondary (automatic) |
| direct_box | gligen | facing_acc | 0.003 [-0.105, 0.109] | secondary (automatic) |
| direct_box | gligen | count_acc | -0.021 [-0.104, 0.062] | secondary (automatic) |
| direct_box | gligen | opr | 0.028 [0.001, 0.055] | secondary (automatic) |
| direct_box | gligen | oce_c | -0.125 [-0.694, 0.590] | secondary (automatic) |
| direct_box | gligen | miou | 0.078 [0.054, 0.102] | secondary (automatic) |
| direct_box | gligen | ra | 0.045 [-0.017, 0.106] | secondary (automatic) |
| direct_box | gligen | joint_layout_success | -0.028 [-0.104, 0.049] | secondary (automatic) |
| direct_box | gligen | clip | -0.099 [-0.862, 0.704] | secondary (automatic) |
