# OCSD-XL development round 3 (M1-M5 on SDXL, 1024 px)

Protocol: assets/xl_dev_v3/protocol.json. 9 scenes x seeds [0, 1]; means over scenes (seeds averaged) with 95% bootstrap CIs over scenes.
Images are generated at 1024 px and scored at 512 px. sketch_chamfer and time_s: lower is better.

| arm | images | opr | miou | count_acc | ra | joint_layout_success | sketch_chamfer | facing_acc | clip | obj_clip | pickscore | time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xl_controlnet | 18/18 | 0.533 [0.421, 0.674] | 0.433 [0.336, 0.547] | 0.111 [0.000, 0.333] | 0.273 [0.124, 0.480] | 0.111 [0.000, 0.333] | 0.028 [0.022, 0.034] | 1.000 [1.000, 1.000] | 29.082 [27.167, 30.925] | 21.759 [20.435, 23.391] | 21.037 [20.513, 21.587] | 6.369 [6.321, 6.423] |
| xl_collage | 18/18 | 0.760 [0.667, 0.862] | 0.586 [0.499, 0.678] | 0.333 [0.111, 0.611] | 0.598 [0.446, 0.765] | 0.333 [0.111, 0.611] | 0.041 [0.035, 0.047] | 0.954 [0.889, 1.000] | 30.295 [28.128, 32.749] | 23.633 [22.574, 24.918] | 21.023 [20.671, 21.402] | 10.977 [10.339, 11.643] |
| ocsd_xl | 18/18 | 0.777 [0.693, 0.869] | 0.566 [0.479, 0.653] | 0.111 [0.000, 0.278] | 0.600 [0.460, 0.763] | 0.111 [0.000, 0.278] | 0.045 [0.040, 0.051] | 0.981 [0.944, 1.000] | 30.245 [27.658, 32.886] | 23.936 [22.927, 25.016] | 21.556 [20.977, 22.219] | 25.169 [23.581, 26.094] |
| ocsd_xl_noenergy | 18/18 | 0.804 [0.722, 0.890] | 0.579 [0.510, 0.645] | 0.167 [0.000, 0.333] | 0.661 [0.522, 0.808] | 0.111 [0.000, 0.278] | 0.045 [0.040, 0.052] | 0.944 [0.833, 1.000] | 30.571 [28.722, 32.634] | 24.012 [22.991, 25.091] | 21.532 [21.063, 22.017] | 15.579 [14.587, 16.326] |
| ocsd_xl_e05 | 18/18 | 0.727 [0.630, 0.822] | 0.474 [0.378, 0.572] | 0.167 [0.000, 0.389] | 0.520 [0.379, 0.674] | 0.167 [0.000, 0.389] | 0.049 [0.041, 0.057] | 0.926 [0.815, 1.000] | 29.989 [27.918, 32.000] | 23.744 [22.515, 25.084] | 21.318 [20.890, 21.829] | 24.294 [21.022, 26.088] |
| ocsd_xl_w04 | 18/18 | 0.760 [0.681, 0.846] | 0.470 [0.405, 0.527] | 0.056 [0.000, 0.167] | 0.566 [0.436, 0.713] | 0.000 [0.000, 0.000] | 0.055 [0.050, 0.061] | 0.898 [0.778, 0.981] | 30.470 [28.927, 32.372] | 23.825 [22.677, 24.816] | 21.945 [21.458, 22.499] | 25.128 [23.545, 26.083] |
| ocsd_xl_w04_noenergy | 18/18 | 0.769 [0.670, 0.870] | 0.488 [0.413, 0.563] | 0.111 [0.000, 0.333] | 0.597 [0.437, 0.769] | 0.000 [0.000, 0.000] | 0.053 [0.047, 0.059] | 0.870 [0.722, 1.000] | 30.078 [28.269, 31.901] | 23.878 [23.028, 24.820] | 21.929 [21.387, 22.519] | 15.235 [13.362, 16.305] |
| ocsd_xl_w08_noenergy | 18/18 | 0.832 [0.731, 0.931] | 0.644 [0.566, 0.721] | 0.333 [0.111, 0.556] | 0.704 [0.528, 0.877] | 0.278 [0.056, 0.500] | 0.036 [0.030, 0.042] | 0.963 [0.889, 1.000] | 29.869 [27.306, 32.512] | 24.177 [23.377, 25.056] | 21.250 [20.785, 21.759] | 14.362 [12.310, 16.275] |
| ocsd_xl_a02 | 18/18 | 0.770 [0.697, 0.863] | 0.553 [0.479, 0.641] | 0.056 [0.000, 0.167] | 0.584 [0.442, 0.747] | 0.056 [0.000, 0.167] | 0.046 [0.041, 0.051] | 0.963 [0.907, 1.000] | 30.080 [27.803, 32.398] | 23.861 [22.782, 24.994] | 21.415 [20.874, 22.036] | 25.176 [23.531, 26.179] |
| ocsd_xl_a06 | 18/18 | 0.726 [0.637, 0.822] | 0.491 [0.422, 0.580] | 0.111 [0.000, 0.278] | 0.513 [0.364, 0.678] | 0.056 [0.000, 0.167] | 0.046 [0.041, 0.051] | 0.944 [0.833, 1.000] | 30.756 [28.550, 32.956] | 23.990 [22.902, 25.216] | 21.816 [21.212, 22.436] | 25.074 [23.461, 26.054] |
| ocsd_xl_norepair | 18/18 | 0.708 [0.616, 0.811] | 0.496 [0.423, 0.582] | 0.111 [0.000, 0.278] | 0.491 [0.342, 0.659] | 0.056 [0.000, 0.167] | 0.044 [0.040, 0.049] | 0.944 [0.833, 1.000] | 30.209 [27.993, 32.629] | 23.917 [22.940, 25.018] | 21.792 [21.232, 22.419] | 10.863 [10.696, 10.999] |
| ocsd_xl_region | 18/18 | 0.703 [0.602, 0.813] | 0.459 [0.400, 0.519] | 0.111 [0.000, 0.278] | 0.485 [0.324, 0.669] | 0.056 [0.000, 0.167] | 0.046 [0.038, 0.055] | 0.907 [0.796, 1.000] | 29.430 [27.741, 31.026] | 23.584 [22.327, 25.000] | 21.962 [21.490, 22.465] | 23.575 [20.265, 25.728] |
| ocsd_xl_id | 0/18 | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] | nan [nan, nan] |

## By object count

| method               | count_bin   |   opr |   miou |   count_acc |   sketch_chamfer |
|:---------------------|:------------|------:|-------:|------------:|-----------------:|
| ocsd_xl              | 3           | 0.833 |  0.652 |       0.333 |            0.042 |
| ocsd_xl              | 5           | 0.8   |  0.571 |       0     |            0.04  |
| ocsd_xl              | 8+          | 0.697 |  0.476 |       0     |            0.054 |
| ocsd_xl_a02          | 3           | 0.778 |  0.591 |       0.167 |            0.041 |
| ocsd_xl_a02          | 5           | 0.8   |  0.553 |       0     |            0.042 |
| ocsd_xl_a02          | 8+          | 0.731 |  0.516 |       0     |            0.055 |
| ocsd_xl_a06          | 3           | 0.722 |  0.555 |       0.333 |            0.041 |
| ocsd_xl_a06          | 5           | 0.8   |  0.475 |       0     |            0.044 |
| ocsd_xl_a06          | 8+          | 0.655 |  0.443 |       0     |            0.052 |
| ocsd_xl_e05          | 3           | 0.833 |  0.561 |       0.5   |            0.038 |
| ocsd_xl_e05          | 5           | 0.733 |  0.446 |       0     |            0.044 |
| ocsd_xl_e05          | 8+          | 0.616 |  0.415 |       0     |            0.065 |
| ocsd_xl_noenergy     | 3           | 0.778 |  0.574 |       0.167 |            0.038 |
| ocsd_xl_noenergy     | 5           | 0.9   |  0.627 |       0.333 |            0.043 |
| ocsd_xl_noenergy     | 8+          | 0.734 |  0.537 |       0     |            0.053 |
| ocsd_xl_norepair     | 3           | 0.722 |  0.553 |       0.333 |            0.042 |
| ocsd_xl_norepair     | 5           | 0.767 |  0.492 |       0     |            0.041 |
| ocsd_xl_norepair     | 8+          | 0.634 |  0.443 |       0     |            0.05  |
| ocsd_xl_region       | 3           | 0.722 |  0.492 |       0.333 |            0.034 |
| ocsd_xl_region       | 5           | 0.733 |  0.438 |       0     |            0.048 |
| ocsd_xl_region       | 8+          | 0.653 |  0.445 |       0     |            0.057 |
| ocsd_xl_w04          | 3           | 0.778 |  0.512 |       0.167 |            0.053 |
| ocsd_xl_w04          | 5           | 0.833 |  0.502 |       0     |            0.051 |
| ocsd_xl_w04          | 8+          | 0.669 |  0.395 |       0     |            0.061 |
| ocsd_xl_w04_noenergy | 3           | 0.778 |  0.489 |       0.333 |            0.046 |
| ocsd_xl_w04_noenergy | 5           | 0.833 |  0.518 |       0     |            0.052 |
| ocsd_xl_w04_noenergy | 8+          | 0.694 |  0.457 |       0     |            0.061 |
| ocsd_xl_w08_noenergy | 3           | 0.889 |  0.696 |       0.5   |            0.032 |
| ocsd_xl_w08_noenergy | 5           | 0.767 |  0.579 |       0.167 |            0.03  |
| ocsd_xl_w08_noenergy | 8+          | 0.84  |  0.657 |       0.333 |            0.045 |
| xl_collage           | 3           | 0.889 |  0.697 |       0.667 |            0.034 |
| xl_collage           | 5           | 0.667 |  0.485 |       0.167 |            0.039 |
| xl_collage           | 8+          | 0.725 |  0.575 |       0.167 |            0.05  |
| xl_controlnet        | 3           | 0.667 |  0.572 |       0.333 |            0.017 |
| xl_controlnet        | 5           | 0.467 |  0.344 |       0     |            0.032 |
| xl_controlnet        | 8+          | 0.465 |  0.382 |       0     |            0.035 |

## Paired differences against xl_controlnet (a - b)

| a | metric | diff [95% CI] |
|---|---|---|
| xl_collage | opr | 0.227 [0.083, 0.379] |
| xl_collage | miou | 0.153 [0.029, 0.283] |
| xl_collage | count_acc | 0.222 [0.000, 0.444] |
| xl_collage | ra | 0.325 [0.127, 0.547] |
| xl_collage | joint_layout_success | 0.222 [0.000, 0.444] |
| xl_collage | sketch_chamfer | 0.013 [0.009, 0.017] |
| xl_collage | facing_acc | -0.046 [-0.111, 0.000] |
| xl_collage | clip | 1.213 [-0.409, 3.033] |
| xl_collage | obj_clip | 1.874 [1.028, 2.814] |
| xl_collage | pickscore | -0.014 [-0.317, 0.299] |
| xl_collage | time_s | 4.608 [3.961, 5.295] |
| ocsd_xl | opr | 0.244 [0.124, 0.375] |
| ocsd_xl | miou | 0.134 [0.034, 0.240] |
| ocsd_xl | count_acc | 0.000 [-0.167, 0.167] |
| ocsd_xl | ra | 0.327 [0.164, 0.510] |
| ocsd_xl | joint_layout_success | 0.000 [-0.167, 0.167] |
| ocsd_xl | sketch_chamfer | 0.018 [0.011, 0.024] |
| ocsd_xl | facing_acc | -0.019 [-0.056, 0.000] |
| ocsd_xl | clip | 1.162 [0.063, 2.372] |
| ocsd_xl | obj_clip | 2.177 [1.250, 3.239] |
| ocsd_xl | pickscore | 0.519 [-0.108, 1.150] |
| ocsd_xl | time_s | 18.801 [17.178, 19.728] |
| ocsd_xl_noenergy | opr | 0.271 [0.153, 0.396] |
| ocsd_xl_noenergy | miou | 0.147 [0.039, 0.256] |
| ocsd_xl_noenergy | count_acc | 0.056 [-0.111, 0.222] |
| ocsd_xl_noenergy | ra | 0.387 [0.215, 0.569] |
| ocsd_xl_noenergy | joint_layout_success | 0.000 [-0.167, 0.167] |
| ocsd_xl_noenergy | sketch_chamfer | 0.017 [0.012, 0.022] |
| ocsd_xl_noenergy | facing_acc | -0.056 [-0.167, 0.000] |
| ocsd_xl_noenergy | clip | 1.488 [0.553, 2.453] |
| ocsd_xl_noenergy | obj_clip | 2.253 [1.364, 3.239] |
| ocsd_xl_noenergy | pickscore | 0.496 [0.016, 0.949] |
| ocsd_xl_noenergy | time_s | 9.210 [8.196, 9.963] |
| ocsd_xl_e05 | opr | 0.195 [0.089, 0.306] |
| ocsd_xl_e05 | miou | 0.041 [-0.053, 0.129] |
| ocsd_xl_e05 | count_acc | 0.056 [0.000, 0.167] |
| ocsd_xl_e05 | ra | 0.247 [0.113, 0.380] |
| ocsd_xl_e05 | joint_layout_success | 0.056 [0.000, 0.167] |
| ocsd_xl_e05 | sketch_chamfer | 0.021 [0.015, 0.027] |
| ocsd_xl_e05 | facing_acc | -0.074 [-0.185, 0.000] |
| ocsd_xl_e05 | clip | 0.907 [-0.216, 2.104] |
| ocsd_xl_e05 | obj_clip | 1.985 [1.183, 2.926] |
| ocsd_xl_e05 | pickscore | 0.282 [-0.252, 0.892] |
| ocsd_xl_e05 | time_s | 17.926 [14.622, 19.715] |
| ocsd_xl_w04 | opr | 0.227 [0.112, 0.342] |
| ocsd_xl_w04 | miou | 0.037 [-0.070, 0.141] |
| ocsd_xl_w04 | count_acc | -0.056 [-0.167, 0.000] |
| ocsd_xl_w04 | ra | 0.293 [0.148, 0.435] |
| ocsd_xl_w04 | joint_layout_success | -0.111 [-0.333, 0.000] |
| ocsd_xl_w04 | sketch_chamfer | 0.027 [0.022, 0.033] |
| ocsd_xl_w04 | facing_acc | -0.102 [-0.222, -0.019] |
| ocsd_xl_w04 | clip | 1.388 [0.068, 2.854] |
| ocsd_xl_w04 | obj_clip | 2.066 [0.909, 3.300] |
| ocsd_xl_w04 | pickscore | 0.908 [0.280, 1.552] |
| ocsd_xl_w04 | time_s | 18.760 [17.145, 19.711] |
| ocsd_xl_w04_noenergy | opr | 0.236 [0.112, 0.367] |
| ocsd_xl_w04_noenergy | miou | 0.055 [-0.072, 0.179] |
| ocsd_xl_w04_noenergy | count_acc | 0.000 [0.000, 0.000] |
| ocsd_xl_w04_noenergy | ra | 0.324 [0.154, 0.512] |
| ocsd_xl_w04_noenergy | joint_layout_success | -0.111 [-0.333, 0.000] |
| ocsd_xl_w04_noenergy | sketch_chamfer | 0.025 [0.019, 0.031] |
| ocsd_xl_w04_noenergy | facing_acc | -0.130 [-0.278, 0.000] |
| ocsd_xl_w04_noenergy | clip | 0.996 [-0.244, 2.257] |
| ocsd_xl_w04_noenergy | obj_clip | 2.119 [1.163, 3.171] |
| ocsd_xl_w04_noenergy | pickscore | 0.892 [0.270, 1.462] |
| ocsd_xl_w04_noenergy | time_s | 8.867 [6.961, 9.937] |
| ocsd_xl_w08_noenergy | opr | 0.299 [0.141, 0.465] |
| ocsd_xl_w08_noenergy | miou | 0.211 [0.088, 0.333] |
| ocsd_xl_w08_noenergy | count_acc | 0.222 [-0.056, 0.500] |
| ocsd_xl_w08_noenergy | ra | 0.430 [0.198, 0.674] |
| ocsd_xl_w08_noenergy | joint_layout_success | 0.167 [-0.111, 0.444] |
| ocsd_xl_w08_noenergy | sketch_chamfer | 0.008 [0.003, 0.013] |
| ocsd_xl_w08_noenergy | facing_acc | -0.037 [-0.111, 0.000] |
| ocsd_xl_w08_noenergy | clip | 0.786 [-0.099, 1.857] |
| ocsd_xl_w08_noenergy | obj_clip | 2.418 [1.429, 3.371] |
| ocsd_xl_w08_noenergy | pickscore | 0.213 [-0.212, 0.629] |
| ocsd_xl_w08_noenergy | time_s | 7.993 [5.950, 9.926] |
| ocsd_xl_a02 | opr | 0.237 [0.126, 0.359] |
| ocsd_xl_a02 | miou | 0.121 [0.038, 0.213] |
| ocsd_xl_a02 | count_acc | -0.056 [-0.167, 0.000] |
| ocsd_xl_a02 | ra | 0.310 [0.165, 0.486] |
| ocsd_xl_a02 | joint_layout_success | -0.056 [-0.167, 0.000] |
| ocsd_xl_a02 | sketch_chamfer | 0.018 [0.013, 0.023] |
| ocsd_xl_a02 | facing_acc | -0.037 [-0.093, 0.000] |
| ocsd_xl_a02 | clip | 0.998 [-0.126, 2.085] |
| ocsd_xl_a02 | obj_clip | 2.102 [1.153, 3.245] |
| ocsd_xl_a02 | pickscore | 0.379 [-0.237, 0.961] |
| ocsd_xl_a02 | time_s | 18.807 [17.134, 19.813] |
| ocsd_xl_a06 | opr | 0.193 [0.105, 0.294] |
| ocsd_xl_a06 | miou | 0.058 [-0.011, 0.127] |
| ocsd_xl_a06 | count_acc | 0.000 [-0.167, 0.167] |
| ocsd_xl_a06 | ra | 0.240 [0.124, 0.367] |
| ocsd_xl_a06 | joint_layout_success | -0.056 [-0.167, 0.000] |
| ocsd_xl_a06 | sketch_chamfer | 0.018 [0.012, 0.023] |
| ocsd_xl_a06 | facing_acc | -0.056 [-0.167, 0.000] |
| ocsd_xl_a06 | clip | 1.673 [0.654, 2.595] |
| ocsd_xl_a06 | obj_clip | 2.231 [1.406, 3.290] |
| ocsd_xl_a06 | pickscore | 0.780 [0.194, 1.475] |
| ocsd_xl_a06 | time_s | 18.705 [17.058, 19.687] |
| ocsd_xl_norepair | opr | 0.175 [0.081, 0.283] |
| ocsd_xl_norepair | miou | 0.063 [-0.017, 0.142] |
| ocsd_xl_norepair | count_acc | 0.000 [-0.167, 0.167] |
| ocsd_xl_norepair | ra | 0.218 [0.100, 0.354] |
| ocsd_xl_norepair | joint_layout_success | -0.056 [-0.167, 0.000] |
| ocsd_xl_norepair | sketch_chamfer | 0.016 [0.010, 0.022] |
| ocsd_xl_norepair | facing_acc | -0.056 [-0.167, 0.000] |
| ocsd_xl_norepair | clip | 1.127 [-0.248, 2.541] |
| ocsd_xl_norepair | obj_clip | 2.158 [1.231, 3.191] |
| ocsd_xl_norepair | pickscore | 0.755 [0.118, 1.455] |
| ocsd_xl_norepair | time_s | 4.494 [4.359, 4.588] |
| ocsd_xl_region | opr | 0.170 [0.086, 0.258] |
| ocsd_xl_region | miou | 0.026 [-0.041, 0.085] |
| ocsd_xl_region | count_acc | 0.000 [-0.167, 0.167] |
| ocsd_xl_region | ra | 0.212 [0.098, 0.337] |
| ocsd_xl_region | joint_layout_success | -0.056 [-0.167, 0.000] |
| ocsd_xl_region | sketch_chamfer | 0.018 [0.012, 0.025] |
| ocsd_xl_region | facing_acc | -0.093 [-0.204, 0.000] |
| ocsd_xl_region | clip | 0.348 [-0.733, 1.718] |
| ocsd_xl_region | obj_clip | 1.825 [0.953, 2.943] |
| ocsd_xl_region | pickscore | 0.926 [0.443, 1.447] |
| ocsd_xl_region | time_s | 17.206 [13.871, 19.355] |
