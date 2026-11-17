# OCSD-XL confirmation: pre-registered analysis

## Primary family (Holm over four)

| id | comparison | metric | wins / losses / ties | difference [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|
| P1 | ocsd_xl_faithful vs ocsd | opr | 11 / 4 / 3 | +0.145 [+0.038, +0.256] | 0.1185 | 0.1925 |
| P2 | ocsd_xl_faithful vs ocsd | miou | 13 / 5 / 0 | +0.154 [+0.056, +0.255] | 0.0963 | 0.1925 |
| P3 | ocsd_xl_natural vs ocsd | naturalness | 15 / 0 / 3 | +0.778 [+0.556, +0.972] | 0.0001 | 0.0002 |
| P4 | ocsd_xl_natural vs ocsd | opr | 11 / 1 / 6 | +0.128 [+0.053, +0.206] | 0.0063 | 0.0190 |

## Claims

* C-faithful: does not hold
* C-natural: HOLDS

## Means over scenes

| arm | images | opr | miou | sketch_chamfer | naturalness | pickscore |
|---|---|---|---|---|---|---|
| collage | 36 | 0.794 | 0.614 | 0.049 | 0.306 | 21.324 |
| controlnet | 36 | 0.599 | 0.457 | 0.036 | 0.722 | 21.402 |
| direct_box | 36 | 0.768 | 0.679 | 0.058 | 1.417 | 21.579 |
| gligen | 36 | 0.700 | 0.565 | 0.066 | 1.500 | 21.249 |
| ocsd | 36 | 0.664 | 0.475 | 0.055 | 0.139 | 20.912 |
| ocsd_v2 | 36 | 0.717 | 0.396 | 0.069 | 1.000 | 21.641 |
| ocsd_v3 | 36 | 0.792 | 0.670 | 0.050 | 1.361 | 21.724 |
| ocsd_xl_faithful | 36 | 0.809 | 0.629 | 0.039 | 0.611 | 21.349 |
| ocsd_xl_natural | 36 | 0.792 | 0.511 | 0.058 | 0.917 | 22.056 |
| xl_collage | 36 | 0.814 | 0.619 | 0.052 | 0.444 | 21.300 |
| xl_controlnet | 36 | 0.498 | 0.394 | 0.029 | 0.361 | 21.120 |

## Secondary (exploratory, unadjusted)

| a | b | metric | w / l / t | difference [95% CI] | p |
|---|---|---|---|---|---|
| ocsd_xl_natural | collage | opr | 7 / 6 / 5 | -0.001 [-0.081, +0.077] | 1.0000 |
| ocsd_xl_natural | collage | miou | 4 / 14 / 0 | -0.103 [-0.177, -0.032] | 0.0309 |
| ocsd_xl_natural | collage | sketch_chamfer | 3 / 15 / 0 | +0.009 [+0.004, +0.014] | 0.0075 |
| ocsd_xl_natural | collage | naturalness | 12 / 1 / 5 | +0.611 [+0.361, +0.861] | 0.0034 |
| ocsd_xl_natural | collage | pickscore | 18 / 0 / 0 | +0.732 [+0.516, +0.967] | 0.0000 |
| ocsd_xl_natural | controlnet | opr | 11 / 2 / 5 | +0.193 [+0.079, +0.303] | 0.0225 |
| ocsd_xl_natural | controlnet | miou | 11 / 7 / 0 | +0.053 [-0.046, +0.144] | 0.4807 |
| ocsd_xl_natural | controlnet | sketch_chamfer | 0 / 18 / 0 | +0.021 [+0.016, +0.028] | 0.0000 |
| ocsd_xl_natural | controlnet | naturalness | 7 / 4 / 7 | +0.194 [-0.139, +0.528] | 0.5488 |
| ocsd_xl_natural | controlnet | pickscore | 15 / 3 / 0 | +0.654 [+0.372, +0.939] | 0.0075 |
| ocsd_xl_natural | direct_box | opr | 8 / 7 / 3 | +0.025 [-0.109, +0.155] | 1.0000 |
| ocsd_xl_natural | direct_box | miou | 6 / 12 / 0 | -0.169 [-0.282, -0.054] | 0.2379 |
| ocsd_xl_natural | direct_box | sketch_chamfer | 10 / 8 / 0 | -0.001 [-0.006, +0.005] | 0.8145 |
| ocsd_xl_natural | direct_box | naturalness | 1 / 10 / 7 | -0.500 [-0.778, -0.222] | 0.0117 |
| ocsd_xl_natural | direct_box | pickscore | 13 / 5 / 0 | +0.477 [+0.133, +0.840] | 0.0963 |
| ocsd_xl_natural | gligen | opr | 11 / 6 / 1 | +0.092 [-0.044, +0.229] | 0.3323 |
| ocsd_xl_natural | gligen | miou | 8 / 10 / 0 | -0.055 [-0.173, +0.065] | 0.8145 |
| ocsd_xl_natural | gligen | sketch_chamfer | 13 / 5 / 0 | -0.008 [-0.015, -0.002] | 0.0963 |
| ocsd_xl_natural | gligen | naturalness | 3 / 11 / 4 | -0.583 [-0.972, -0.194] | 0.0574 |
| ocsd_xl_natural | gligen | pickscore | 16 / 2 / 0 | +0.807 [+0.377, +1.276] | 0.0013 |
| ocsd_xl_natural | ocsd | opr | 11 / 1 / 6 | +0.128 [+0.053, +0.206] | 0.0063 |
| ocsd_xl_natural | ocsd | miou | 10 / 8 / 0 | +0.036 [-0.044, +0.118] | 0.8145 |
| ocsd_xl_natural | ocsd | sketch_chamfer | 7 / 11 / 0 | +0.002 [-0.003, +0.007] | 0.4807 |
| ocsd_xl_natural | ocsd | naturalness | 15 / 0 / 3 | +0.778 [+0.556, +0.972] | 0.0001 |
| ocsd_xl_natural | ocsd | pickscore | 18 / 0 / 0 | +1.144 [+0.929, +1.374] | 0.0000 |
| ocsd_xl_natural | ocsd_v2 | opr | 10 / 6 / 2 | +0.076 [-0.033, +0.186] | 0.4545 |
| ocsd_xl_natural | ocsd_v2 | miou | 12 / 6 / 0 | +0.115 [+0.028, +0.202] | 0.2379 |
| ocsd_xl_natural | ocsd_v2 | sketch_chamfer | 14 / 4 / 0 | -0.011 [-0.017, -0.005] | 0.0309 |
| ocsd_xl_natural | ocsd_v2 | naturalness | 6 / 8 / 4 | -0.083 [-0.361, +0.222] | 0.7905 |
| ocsd_xl_natural | ocsd_v2 | pickscore | 13 / 5 / 0 | +0.415 [+0.117, +0.723] | 0.0963 |
| ocsd_xl_natural | ocsd_v3 | opr | 6 / 6 / 6 | -0.000 [-0.108, +0.109] | 1.0000 |
| ocsd_xl_natural | ocsd_v3 | miou | 4 / 14 / 0 | -0.159 [-0.258, -0.058] | 0.0309 |
| ocsd_xl_natural | ocsd_v3 | sketch_chamfer | 7 / 11 / 0 | +0.007 [+0.001, +0.013] | 0.4807 |
| ocsd_xl_natural | ocsd_v3 | naturalness | 3 / 10 / 5 | -0.444 [-0.806, -0.111] | 0.0923 |
| ocsd_xl_natural | ocsd_v3 | pickscore | 11 / 7 / 0 | +0.332 [-0.036, +0.678] | 0.4807 |
| ocsd_xl_natural | ocsd_xl_faithful | opr | 6 / 7 / 5 | -0.017 [-0.100, +0.060] | 1.0000 |
| ocsd_xl_natural | ocsd_xl_faithful | miou | 3 / 15 / 0 | -0.118 [-0.177, -0.059] | 0.0075 |
| ocsd_xl_natural | ocsd_xl_faithful | sketch_chamfer | 1 / 17 / 0 | +0.019 [+0.013, +0.024] | 0.0001 |
| ocsd_xl_natural | ocsd_xl_faithful | naturalness | 9 / 3 / 6 | +0.306 [+0.056, +0.583] | 0.1460 |
| ocsd_xl_natural | ocsd_xl_faithful | pickscore | 17 / 1 / 0 | +0.707 [+0.453, +0.986] | 0.0001 |
| ocsd_xl_natural | xl_collage | opr | 5 / 7 / 6 | -0.022 [-0.095, +0.053] | 0.7744 |
| ocsd_xl_natural | xl_collage | miou | 5 / 13 / 0 | -0.108 [-0.167, -0.049] | 0.0963 |
| ocsd_xl_natural | xl_collage | sketch_chamfer | 8 / 10 / 0 | +0.005 [+0.000, +0.011] | 0.8145 |
| ocsd_xl_natural | xl_collage | naturalness | 11 / 1 / 6 | +0.472 [+0.222, +0.722] | 0.0063 |
| ocsd_xl_natural | xl_collage | pickscore | 17 / 1 / 0 | +0.756 [+0.485, +1.052] | 0.0001 |
| ocsd_xl_natural | xl_controlnet | opr | 15 / 1 / 2 | +0.294 [+0.195, +0.393] | 0.0005 |
| ocsd_xl_natural | xl_controlnet | miou | 13 / 5 / 0 | +0.116 [+0.021, +0.218] | 0.0963 |
| ocsd_xl_natural | xl_controlnet | sketch_chamfer | 0 / 18 / 0 | +0.028 [+0.023, +0.034] | 0.0000 |
| ocsd_xl_natural | xl_controlnet | naturalness | 12 / 2 / 4 | +0.556 [+0.222, +0.861] | 0.0129 |
| ocsd_xl_natural | xl_controlnet | pickscore | 16 / 2 / 0 | +0.936 [+0.636, +1.249] | 0.0013 |
| ocsd_xl_faithful | collage | opr | 9 / 6 / 3 | +0.016 [-0.085, +0.115] | 0.6072 |
| ocsd_xl_faithful | collage | miou | 10 / 8 / 0 | +0.015 [-0.070, +0.100] | 0.8145 |
| ocsd_xl_faithful | collage | sketch_chamfer | 16 / 2 / 0 | -0.010 [-0.014, -0.005] | 0.0013 |
| ocsd_xl_faithful | collage | naturalness | 8 / 3 / 7 | +0.306 [+0.056, +0.583] | 0.2266 |
| ocsd_xl_faithful | collage | pickscore | 11 / 7 / 0 | +0.025 [-0.237, +0.263] | 0.4807 |
| ocsd_xl_faithful | controlnet | opr | 12 / 2 / 4 | +0.210 [+0.110, +0.312] | 0.0129 |
| ocsd_xl_faithful | controlnet | miou | 14 / 4 / 0 | +0.171 [+0.084, +0.257] | 0.0309 |
| ocsd_xl_faithful | controlnet | sketch_chamfer | 9 / 9 / 0 | +0.003 [-0.003, +0.008] | 1.0000 |
| ocsd_xl_faithful | controlnet | naturalness | 6 / 9 / 3 | -0.111 [-0.389, +0.194] | 0.6072 |
| ocsd_xl_faithful | controlnet | pickscore | 9 / 9 / 0 | -0.053 [-0.315, +0.210] | 1.0000 |
| ocsd_xl_faithful | direct_box | opr | 10 / 4 / 4 | +0.042 [-0.100, +0.171] | 0.1796 |
| ocsd_xl_faithful | direct_box | miou | 9 / 9 / 0 | -0.050 [-0.181, +0.070] | 1.0000 |
| ocsd_xl_faithful | direct_box | sketch_chamfer | 17 / 1 / 0 | -0.019 [-0.025, -0.014] | 0.0001 |
| ocsd_xl_faithful | direct_box | naturalness | 0 / 14 / 4 | -0.806 [-1.056, -0.556] | 0.0001 |
| ocsd_xl_faithful | direct_box | pickscore | 6 / 12 / 0 | -0.230 [-0.612, +0.103] | 0.2379 |
| ocsd_xl_faithful | gligen | opr | 11 / 4 / 3 | +0.109 [-0.029, +0.244] | 0.1185 |
| ocsd_xl_faithful | gligen | miou | 11 / 7 / 0 | +0.064 [-0.067, +0.185] | 0.4807 |
| ocsd_xl_faithful | gligen | sketch_chamfer | 18 / 0 / 0 | -0.027 [-0.036, -0.020] | 0.0000 |
| ocsd_xl_faithful | gligen | naturalness | 2 / 14 / 2 | -0.889 [-1.278, -0.472] | 0.0042 |
| ocsd_xl_faithful | gligen | pickscore | 10 / 8 / 0 | +0.100 [-0.273, +0.481] | 0.8145 |
| ocsd_xl_faithful | ocsd | opr | 11 / 4 / 3 | +0.145 [+0.038, +0.256] | 0.1185 |
| ocsd_xl_faithful | ocsd | miou | 13 / 5 / 0 | +0.154 [+0.056, +0.255] | 0.0963 |
| ocsd_xl_faithful | ocsd | sketch_chamfer | 18 / 0 / 0 | -0.017 [-0.022, -0.011] | 0.0000 |
| ocsd_xl_faithful | ocsd | naturalness | 11 / 1 / 6 | +0.472 [+0.167, +0.750] | 0.0063 |
| ocsd_xl_faithful | ocsd | pickscore | 16 / 2 / 0 | +0.437 [+0.117, +0.728] | 0.0013 |
| ocsd_xl_faithful | ocsd_v2 | opr | 10 / 5 / 3 | +0.093 [-0.028, +0.219] | 0.3018 |
| ocsd_xl_faithful | ocsd_v2 | miou | 16 / 2 / 0 | +0.233 [+0.132, +0.331] | 0.0013 |
| ocsd_xl_faithful | ocsd_v2 | sketch_chamfer | 18 / 0 / 0 | -0.030 [-0.035, -0.025] | 0.0000 |
| ocsd_xl_faithful | ocsd_v2 | naturalness | 3 / 10 / 5 | -0.389 [-0.694, -0.083] | 0.0923 |
| ocsd_xl_faithful | ocsd_v2 | pickscore | 6 / 12 / 0 | -0.292 [-0.658, +0.057] | 0.2379 |
| ocsd_xl_faithful | ocsd_v3 | opr | 6 / 6 / 6 | +0.017 [-0.104, +0.137] | 1.0000 |
| ocsd_xl_faithful | ocsd_v3 | miou | 8 / 10 / 0 | -0.041 [-0.159, +0.072] | 0.8145 |
| ocsd_xl_faithful | ocsd_v3 | sketch_chamfer | 16 / 2 / 0 | -0.012 [-0.016, -0.006] | 0.0013 |
| ocsd_xl_faithful | ocsd_v3 | naturalness | 2 / 13 / 3 | -0.750 [-1.056, -0.417] | 0.0074 |
| ocsd_xl_faithful | ocsd_v3 | pickscore | 7 / 11 / 0 | -0.375 [-0.760, -0.028] | 0.4807 |
| ocsd_xl_faithful | ocsd_xl_natural | opr | 7 / 6 / 5 | +0.017 [-0.060, +0.100] | 1.0000 |
| ocsd_xl_faithful | ocsd_xl_natural | miou | 15 / 3 / 0 | +0.118 [+0.059, +0.177] | 0.0075 |
| ocsd_xl_faithful | ocsd_xl_natural | sketch_chamfer | 17 / 1 / 0 | -0.019 [-0.024, -0.013] | 0.0001 |
| ocsd_xl_faithful | ocsd_xl_natural | naturalness | 3 / 9 / 6 | -0.306 [-0.583, -0.056] | 0.1460 |
| ocsd_xl_faithful | ocsd_xl_natural | pickscore | 1 / 17 / 0 | -0.707 [-0.986, -0.453] | 0.0001 |
| ocsd_xl_faithful | xl_collage | opr | 6 / 6 / 6 | -0.004 [-0.061, +0.046] | 1.0000 |
| ocsd_xl_faithful | xl_collage | miou | 12 / 6 / 0 | +0.010 [-0.034, +0.050] | 0.2379 |
| ocsd_xl_faithful | xl_collage | sketch_chamfer | 17 / 1 / 0 | -0.014 [-0.018, -0.009] | 0.0001 |
| ocsd_xl_faithful | xl_collage | naturalness | 4 / 2 / 12 | +0.167 [-0.056, +0.444] | 0.6875 |
| ocsd_xl_faithful | xl_collage | pickscore | 11 / 7 / 0 | +0.049 [-0.041, +0.141] | 0.4807 |
| ocsd_xl_faithful | xl_controlnet | opr | 17 / 0 / 1 | +0.311 [+0.228, +0.400] | 0.0000 |
| ocsd_xl_faithful | xl_controlnet | miou | 17 / 1 / 0 | +0.235 [+0.155, +0.317] | 0.0001 |
| ocsd_xl_faithful | xl_controlnet | sketch_chamfer | 2 / 16 / 0 | +0.010 [+0.004, +0.015] | 0.0013 |
| ocsd_xl_faithful | xl_controlnet | naturalness | 8 / 3 / 7 | +0.250 [-0.083, +0.583] | 0.2266 |
| ocsd_xl_faithful | xl_controlnet | pickscore | 11 / 7 / 0 | +0.229 [+0.017, +0.450] | 0.4807 |
