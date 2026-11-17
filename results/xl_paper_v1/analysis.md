# OCSD-XL confirmation: pre-registered analysis

## Primary family (Holm over four)

| id | comparison | metric | wins / losses / ties | difference [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|
| P1 | ocsd_xl_faithful vs ocsd | opr | 35 / 9 / 28 | +0.113 [+0.056, +0.172] | 0.0001 | 0.0003 |
| P2 | ocsd_xl_faithful vs ocsd | miou | 51 / 21 / 0 | +0.094 [+0.034, +0.153] | 0.0005 | 0.0005 |
| P3 | ocsd_xl_natural vs ocsd | naturalness | 23 / 2 / 11 | +0.861 [+0.583, +1.139] | 0.0000 | 0.0001 |
| P4 | ocsd_xl_natural vs ocsd | opr | 34 / 9 / 29 | +0.108 [+0.060, +0.158] | 0.0002 | 0.0003 |

## Claims

* C-faithful: HOLDS
* C-natural: HOLDS

## Means over scenes

| arm | images | opr | miou | sketch_chamfer | naturalness | pickscore |
|---|---|---|---|---|---|---|
| collage | 144 | 0.824 | 0.607 | 0.049 | 0.278 | 21.518 |
| controlnet | 144 | 0.655 | 0.492 | 0.038 | 0.750 | 21.671 |
| direct_box | 144 | 0.829 | 0.730 | 0.060 | 1.694 | 21.881 |
| gligen | 144 | 0.801 | 0.653 | 0.066 | 1.444 | 21.409 |
| ocsd | 144 | 0.711 | 0.529 | 0.055 | 0.222 | 21.164 |
| ocsd_v2 | 144 | 0.744 | 0.408 | 0.066 | 0.889 | 22.182 |
| ocsd_v3 | 144 | 0.812 | 0.682 | 0.050 | 1.694 | 22.037 |
| ocsd_xl_faithful | 144 | 0.825 | 0.623 | 0.041 | 0.750 | 21.773 |
| ocsd_xl_natural | 144 | 0.819 | 0.519 | 0.054 | 1.083 | 22.486 |
| xl_collage | 144 | 0.805 | 0.610 | 0.048 | 0.417 | 21.532 |
| xl_controlnet | 144 | 0.580 | 0.436 | 0.028 | 0.611 | 21.367 |

## Secondary (exploratory, unadjusted)

| a | b | metric | w / l / t | difference [95% CI] | p |
|---|---|---|---|---|---|
| ocsd_xl_natural | collage | opr | 20 / 23 / 29 | -0.005 [-0.049, +0.038] | 0.7608 |
| ocsd_xl_natural | collage | miou | 29 / 43 / 0 | -0.089 [-0.132, -0.046] | 0.1249 |
| ocsd_xl_natural | collage | sketch_chamfer | 27 / 45 / 0 | +0.005 [+0.001, +0.009] | 0.0444 |
| ocsd_xl_natural | collage | naturalness | 24 / 4 / 8 | +0.806 [+0.500, +1.111] | 0.0002 |
| ocsd_xl_natural | collage | pickscore | 65 / 7 / 0 | +0.968 [+0.789, +1.142] | 0.0000 |
| ocsd_xl_natural | controlnet | opr | 42 / 7 / 23 | +0.165 [+0.108, +0.219] | 0.0000 |
| ocsd_xl_natural | controlnet | miou | 43 / 29 / 0 | +0.027 [-0.023, +0.075] | 0.1249 |
| ocsd_xl_natural | controlnet | sketch_chamfer | 11 / 61 / 0 | +0.017 [+0.013, +0.020] | 0.0000 |
| ocsd_xl_natural | controlnet | naturalness | 18 / 10 / 8 | +0.333 [+0.000, +0.667] | 0.1849 |
| ocsd_xl_natural | controlnet | pickscore | 61 / 11 / 0 | +0.814 [+0.623, +1.003] | 0.0000 |
| ocsd_xl_natural | direct_box | opr | 22 / 20 / 30 | -0.009 [-0.055, +0.035] | 0.8776 |
| ocsd_xl_natural | direct_box | miou | 8 / 64 / 0 | -0.211 [-0.261, -0.164] | 0.0000 |
| ocsd_xl_natural | direct_box | sketch_chamfer | 48 / 24 / 0 | -0.006 [-0.010, -0.002] | 0.0063 |
| ocsd_xl_natural | direct_box | naturalness | 3 / 17 / 16 | -0.611 [-0.917, -0.306] | 0.0026 |
| ocsd_xl_natural | direct_box | pickscore | 52 / 20 / 0 | +0.605 [+0.398, +0.815] | 0.0002 |
| ocsd_xl_natural | gligen | opr | 24 / 12 / 36 | +0.018 [-0.030, +0.063] | 0.0652 |
| ocsd_xl_natural | gligen | miou | 18 / 54 / 0 | -0.134 [-0.181, -0.089] | 0.0000 |
| ocsd_xl_natural | gligen | sketch_chamfer | 54 / 18 / 0 | -0.012 [-0.016, -0.007] | 0.0000 |
| ocsd_xl_natural | gligen | naturalness | 6 / 17 / 13 | -0.361 [-0.667, -0.056] | 0.0347 |
| ocsd_xl_natural | gligen | pickscore | 66 / 6 / 0 | +1.077 [+0.873, +1.281] | 0.0000 |
| ocsd_xl_natural | ocsd | opr | 34 / 9 / 29 | +0.108 [+0.060, +0.158] | 0.0002 |
| ocsd_xl_natural | ocsd | miou | 34 / 38 / 0 | -0.010 [-0.066, +0.042] | 0.7239 |
| ocsd_xl_natural | ocsd | sketch_chamfer | 33 / 39 / 0 | -0.000 [-0.005, +0.005] | 0.5560 |
| ocsd_xl_natural | ocsd | naturalness | 23 / 2 / 11 | +0.861 [+0.583, +1.139] | 0.0000 |
| ocsd_xl_natural | ocsd | pickscore | 71 / 1 / 0 | +1.322 [+1.161, +1.486] | 0.0000 |
| ocsd_xl_natural | ocsd_v2 | opr | 32 / 6 / 34 | +0.075 [+0.041, +0.110] | 0.0000 |
| ocsd_xl_natural | ocsd_v2 | miou | 57 / 15 / 0 | +0.111 [+0.072, +0.149] | 0.0000 |
| ocsd_xl_natural | ocsd_v2 | sketch_chamfer | 50 / 22 / 0 | -0.012 [-0.016, -0.007] | 0.0013 |
| ocsd_xl_natural | ocsd_v2 | naturalness | 15 / 9 / 12 | +0.194 [-0.111, +0.500] | 0.3075 |
| ocsd_xl_natural | ocsd_v2 | pickscore | 48 / 24 / 0 | +0.304 [+0.153, +0.449] | 0.0063 |
| ocsd_xl_natural | ocsd_v3 | opr | 23 / 18 / 31 | +0.007 [-0.030, +0.044] | 0.5327 |
| ocsd_xl_natural | ocsd_v3 | miou | 12 / 60 / 0 | -0.164 [-0.209, -0.121] | 0.0000 |
| ocsd_xl_natural | ocsd_v3 | sketch_chamfer | 29 / 43 / 0 | +0.005 [+0.001, +0.008] | 0.1249 |
| ocsd_xl_natural | ocsd_v3 | naturalness | 5 / 17 / 14 | -0.611 [-0.944, -0.278] | 0.0169 |
| ocsd_xl_natural | ocsd_v3 | pickscore | 46 / 26 / 0 | +0.449 [+0.229, +0.668] | 0.0245 |
| ocsd_xl_natural | ocsd_xl_faithful | opr | 17 / 21 / 34 | -0.005 [-0.041, +0.031] | 0.6271 |
| ocsd_xl_natural | ocsd_xl_faithful | miou | 16 / 56 / 0 | -0.104 [-0.143, -0.064] | 0.0000 |
| ocsd_xl_natural | ocsd_xl_faithful | sketch_chamfer | 8 / 64 / 0 | +0.013 [+0.009, +0.017] | 0.0000 |
| ocsd_xl_natural | ocsd_xl_faithful | naturalness | 12 / 4 / 20 | +0.333 [+0.083, +0.611] | 0.0768 |
| ocsd_xl_natural | ocsd_xl_faithful | pickscore | 61 / 11 / 0 | +0.713 [+0.534, +0.899] | 0.0000 |
| ocsd_xl_natural | xl_collage | opr | 18 / 17 / 37 | +0.015 [-0.017, +0.048] | 1.0000 |
| ocsd_xl_natural | xl_collage | miou | 21 / 51 / 0 | -0.091 [-0.126, -0.057] | 0.0005 |
| ocsd_xl_natural | xl_collage | sketch_chamfer | 25 / 47 / 0 | +0.007 [+0.004, +0.010] | 0.0128 |
| ocsd_xl_natural | xl_collage | naturalness | 19 / 1 / 16 | +0.667 [+0.417, +0.917] | 0.0000 |
| ocsd_xl_natural | xl_collage | pickscore | 67 / 5 / 0 | +0.954 [+0.781, +1.133] | 0.0000 |
| ocsd_xl_natural | xl_controlnet | opr | 51 / 2 / 19 | +0.239 [+0.190, +0.293] | 0.0000 |
| ocsd_xl_natural | xl_controlnet | miou | 47 / 25 / 0 | +0.082 [+0.037, +0.130] | 0.0128 |
| ocsd_xl_natural | xl_controlnet | sketch_chamfer | 5 / 67 / 0 | +0.026 [+0.022, +0.030] | 0.0000 |
| ocsd_xl_natural | xl_controlnet | naturalness | 17 / 5 / 14 | +0.472 [+0.167, +0.778] | 0.0169 |
| ocsd_xl_natural | xl_controlnet | pickscore | 66 / 6 / 0 | +1.119 [+0.891, +1.346] | 0.0000 |
| ocsd_xl_faithful | collage | opr | 21 / 25 / 26 | +0.000 [-0.050, +0.049] | 0.6587 |
| ocsd_xl_faithful | collage | miou | 39 / 33 / 0 | +0.015 [-0.029, +0.059] | 0.5560 |
| ocsd_xl_faithful | collage | sketch_chamfer | 51 / 21 / 0 | -0.008 [-0.013, -0.004] | 0.0005 |
| ocsd_xl_faithful | collage | naturalness | 16 / 4 / 16 | +0.472 [+0.194, +0.750] | 0.0118 |
| ocsd_xl_faithful | collage | pickscore | 43 / 29 / 0 | +0.255 [+0.079, +0.437] | 0.1249 |
| ocsd_xl_faithful | controlnet | opr | 44 / 9 / 19 | +0.170 [+0.111, +0.228] | 0.0000 |
| ocsd_xl_faithful | controlnet | miou | 50 / 22 / 0 | +0.131 [+0.076, +0.185] | 0.0013 |
| ocsd_xl_faithful | controlnet | sketch_chamfer | 30 / 42 / 0 | +0.003 [-0.000, +0.007] | 0.1945 |
| ocsd_xl_faithful | controlnet | naturalness | 12 / 13 / 11 | +0.000 [-0.278, +0.278] | 1.0000 |
| ocsd_xl_faithful | controlnet | pickscore | 42 / 30 / 0 | +0.102 [-0.062, +0.263] | 0.1945 |
| ocsd_xl_faithful | direct_box | opr | 25 / 19 / 28 | -0.004 [-0.058, +0.047] | 0.4514 |
| ocsd_xl_faithful | direct_box | miou | 26 / 46 / 0 | -0.107 [-0.165, -0.051] | 0.0245 |
| ocsd_xl_faithful | direct_box | sketch_chamfer | 62 / 10 / 0 | -0.019 [-0.023, -0.015] | 0.0000 |
| ocsd_xl_faithful | direct_box | naturalness | 1 / 25 / 10 | -0.944 [-1.194, -0.667] | 0.0000 |
| ocsd_xl_faithful | direct_box | pickscore | 29 / 43 / 0 | -0.107 [-0.317, +0.105] | 0.1249 |
| ocsd_xl_faithful | gligen | opr | 25 / 18 / 29 | +0.023 [-0.033, +0.077] | 0.3604 |
| ocsd_xl_faithful | gligen | miou | 33 / 39 / 0 | -0.030 [-0.087, +0.025] | 0.5560 |
| ocsd_xl_faithful | gligen | sketch_chamfer | 63 / 9 / 0 | -0.025 [-0.029, -0.021] | 0.0000 |
| ocsd_xl_faithful | gligen | naturalness | 3 / 21 / 12 | -0.694 [-0.972, -0.389] | 0.0003 |
| ocsd_xl_faithful | gligen | pickscore | 46 / 26 / 0 | +0.364 [+0.163, +0.566] | 0.0245 |
| ocsd_xl_faithful | ocsd | opr | 35 / 9 / 28 | +0.113 [+0.056, +0.172] | 0.0001 |
| ocsd_xl_faithful | ocsd | miou | 51 / 21 / 0 | +0.094 [+0.034, +0.153] | 0.0005 |
| ocsd_xl_faithful | ocsd | sketch_chamfer | 58 / 14 / 0 | -0.014 [-0.018, -0.009] | 0.0000 |
| ocsd_xl_faithful | ocsd | naturalness | 15 / 2 / 19 | +0.528 [+0.250, +0.806] | 0.0023 |
| ocsd_xl_faithful | ocsd | pickscore | 59 / 13 / 0 | +0.609 [+0.443, +0.776] | 0.0000 |
| ocsd_xl_faithful | ocsd_v2 | opr | 30 / 9 / 33 | +0.081 [+0.035, +0.129] | 0.0011 |
| ocsd_xl_faithful | ocsd_v2 | miou | 62 / 10 / 0 | +0.215 [+0.170, +0.260] | 0.0000 |
| ocsd_xl_faithful | ocsd_v2 | sketch_chamfer | 66 / 6 / 0 | -0.025 [-0.030, -0.020] | 0.0000 |
| ocsd_xl_faithful | ocsd_v2 | naturalness | 9 / 12 / 15 | -0.139 [-0.444, +0.167] | 0.6636 |
| ocsd_xl_faithful | ocsd_v2 | pickscore | 19 / 53 / 0 | -0.408 [-0.582, -0.248] | 0.0001 |
| ocsd_xl_faithful | ocsd_v3 | opr | 28 / 18 / 26 | +0.013 [-0.039, +0.060] | 0.1839 |
| ocsd_xl_faithful | ocsd_v3 | miou | 31 / 41 / 0 | -0.060 [-0.117, -0.005] | 0.2888 |
| ocsd_xl_faithful | ocsd_v3 | sketch_chamfer | 56 / 16 / 0 | -0.009 [-0.013, -0.005] | 0.0000 |
| ocsd_xl_faithful | ocsd_v3 | naturalness | 3 / 24 / 9 | -0.944 [-1.250, -0.639] | 0.0000 |
| ocsd_xl_faithful | ocsd_v3 | pickscore | 23 / 49 / 0 | -0.264 [-0.503, -0.021] | 0.0029 |
| ocsd_xl_faithful | ocsd_xl_natural | opr | 21 / 17 / 34 | +0.005 [-0.031, +0.041] | 0.6271 |
| ocsd_xl_faithful | ocsd_xl_natural | miou | 56 / 16 / 0 | +0.104 [+0.064, +0.143] | 0.0000 |
| ocsd_xl_faithful | ocsd_xl_natural | sketch_chamfer | 64 / 8 / 0 | -0.013 [-0.017, -0.009] | 0.0000 |
| ocsd_xl_faithful | ocsd_xl_natural | naturalness | 4 / 12 / 20 | -0.333 [-0.611, -0.083] | 0.0768 |
| ocsd_xl_faithful | ocsd_xl_natural | pickscore | 11 / 61 / 0 | -0.713 [-0.899, -0.534] | 0.0000 |
| ocsd_xl_faithful | xl_collage | opr | 22 / 17 / 33 | +0.020 [-0.020, +0.059] | 0.5224 |
| ocsd_xl_faithful | xl_collage | miou | 35 / 37 / 0 | +0.013 [-0.026, +0.051] | 0.9063 |
| ocsd_xl_faithful | xl_collage | sketch_chamfer | 55 / 17 / 0 | -0.007 [-0.010, -0.003] | 0.0000 |
| ocsd_xl_faithful | xl_collage | naturalness | 11 / 1 / 24 | +0.333 [+0.139, +0.556] | 0.0063 |
| ocsd_xl_faithful | xl_collage | pickscore | 51 / 21 / 0 | +0.241 [+0.118, +0.361] | 0.0005 |
| ocsd_xl_faithful | xl_controlnet | opr | 54 / 3 / 15 | +0.245 [+0.182, +0.309] | 0.0000 |
| ocsd_xl_faithful | xl_controlnet | miou | 59 / 13 / 0 | +0.186 [+0.129, +0.243] | 0.0000 |
| ocsd_xl_faithful | xl_controlnet | sketch_chamfer | 6 / 66 / 0 | +0.013 [+0.010, +0.016] | 0.0000 |
| ocsd_xl_faithful | xl_controlnet | naturalness | 10 / 7 / 19 | +0.139 [-0.111, +0.389] | 0.6291 |
| ocsd_xl_faithful | xl_controlnet | pickscore | 57 / 15 / 0 | +0.407 [+0.232, +0.574] | 0.0000 |
