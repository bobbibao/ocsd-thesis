# OCSD with harmonised M4 (SD 1.5, M1-M5, M3 on): one-run development, freeze and paper benchmark - paper benchmark

Protocol: assets/m4_study_v1/protocol.json (docs/M4_STUDY_PROTOCOL.md). 72 scenes x seeds [0, 1]; per-scene values averaged over seeds; means with 95% bootstrap CIs over scenes. SD 1.5 at 512 px for every arm. sketch_chamfer: lower is better.

`ocsd_final` = development arm `ocsd_m4_combo` frozen on 2026-11-24 23:32:04 (config 45674144fd85).

| arm | images | opr | miou | count_acc | ra | joint_layout_success | sketch_chamfer | facing_acc | clip | obj_clip | pickscore |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ocsd_final | 144/144 | 0.785 [0.733, 0.834] | 0.484 [0.443, 0.527] | 0.333 [0.236, 0.438] | 0.508 [0.429, 0.585] | 0.201 [0.118, 0.285] | 0.061 [0.057, 0.066] | 0.785 [0.700, 0.860] | 28.687 [27.891, 29.488] | 24.939 [24.506, 25.384] | 21.889 [21.679, 22.097] |
| t2i_adapter | 144/144 | 0.624 [0.554, 0.690] | 0.437 [0.380, 0.494] | 0.312 [0.215, 0.410] | 0.282 [0.213, 0.356] | 0.208 [0.132, 0.292] | 0.035 [0.032, 0.038] | 0.963 [0.930, 0.988] | 28.236 [27.488, 28.974] | 22.870 [22.176, 23.596] | 21.531 [21.299, 21.769] |
| zhang2025 | 144/144 | 0.633 [0.566, 0.698] | 0.435 [0.381, 0.487] | 0.278 [0.188, 0.375] | 0.337 [0.269, 0.411] | 0.167 [0.090, 0.250] | 0.059 [0.054, 0.064] | 0.893 [0.835, 0.945] | 25.550 [24.597, 26.482] | 21.793 [21.194, 22.330] | 20.730 [20.572, 20.883] |
| ocsd | 144/144 | 0.711 [0.648, 0.771] | 0.529 [0.474, 0.582] | 0.333 [0.229, 0.438] | 0.445 [0.366, 0.528] | 0.271 [0.174, 0.375] | 0.055 [0.050, 0.059] | 0.916 [0.862, 0.963] | 28.113 [27.248, 28.960] | 24.009 [23.616, 24.398] | 21.164 [20.967, 21.356] |
| gligen | 144/144 | 0.801 [0.750, 0.851] | 0.653 [0.605, 0.699] | 0.465 [0.361, 0.569] | 0.563 [0.479, 0.648] | 0.444 [0.347, 0.549] | 0.066 [0.063, 0.069] | 0.509 [0.418, 0.599] | 28.230 [27.392, 29.094] | 24.597 [23.941, 25.261] | 21.409 [21.155, 21.660] |
| collage | 144/144 | 0.824 [0.775, 0.869] | 0.607 [0.566, 0.648] | 0.451 [0.354, 0.549] | 0.616 [0.532, 0.699] | 0.319 [0.229, 0.417] | 0.049 [0.045, 0.053] | 0.902 [0.845, 0.953] | 28.800 [27.926, 29.647] | 24.777 [24.293, 25.270] | 21.518 [21.301, 21.732] |
| controlnet | 144/144 | 0.655 [0.588, 0.719] | 0.492 [0.435, 0.549] | 0.257 [0.174, 0.347] | 0.321 [0.249, 0.393] | 0.222 [0.146, 0.306] | 0.038 [0.035, 0.041] | 0.977 [0.948, 0.997] | 27.843 [27.106, 28.570] | 23.248 [22.596, 23.905] | 21.671 [21.452, 21.894] |
| ocsd_v2 | 144/144 | 0.744 [0.688, 0.796] | 0.408 [0.367, 0.449] | 0.361 [0.264, 0.458] | 0.455 [0.388, 0.524] | 0.174 [0.097, 0.257] | 0.066 [0.062, 0.071] | 0.777 [0.691, 0.853] | 28.865 [28.111, 29.602] | 24.576 [24.067, 25.098] | 22.182 [21.958, 22.399] |

Arms: `ocsd_final` OCSD (M4 harmonised); `t2i_adapter` SD + T2I-Adapter; `zhang2025` Zhang et al. (2025) - cài đặt lại; `ocsd` OCSD (đề xuất); `gligen` GLIGEN (hộp + văn bản); `collage` Collage: M2 objects pasted + SDEdit; `controlnet` SD + ControlNet (mô hình nền); `ocsd_v2` OCSD-v2 (joint trajectory, training-free)

## By object count

| method      | count_bin   |   opr |   miou |      ra |   count_acc |   pickscore |
|:------------|:------------|------:|-------:|--------:|------------:|------------:|
| collage     | 1           | 0.972 |  0.718 | nan     |       0.833 |      21.157 |
| collage     | 3           | 0.833 |  0.634 |   0.707 |       0.583 |      21.87  |
| collage     | 5           | 0.806 |  0.577 |   0.663 |       0.306 |      21.69  |
| collage     | 8+          | 0.686 |  0.501 |   0.478 |       0.083 |      21.354 |
| controlnet  | 1           | 0.972 |  0.742 | nan     |       0.833 |      21.594 |
| controlnet  | 3           | 0.639 |  0.497 |   0.392 |       0.139 |      21.936 |
| controlnet  | 5           | 0.617 |  0.454 |   0.393 |       0.056 |      21.859 |
| controlnet  | 8+          | 0.392 |  0.274 |   0.176 |       0     |      21.295 |
| gligen      | 1           | 1     |  0.863 | nan     |       0.917 |      21.582 |
| gligen      | 3           | 0.898 |  0.735 |   0.797 |       0.639 |      21.788 |
| gligen      | 5           | 0.756 |  0.601 |   0.574 |       0.306 |      21.402 |
| gligen      | 8+          | 0.552 |  0.411 |   0.318 |       0     |      20.863 |
| ocsd        | 1           | 0.889 |  0.708 | nan     |       0.806 |      21.013 |
| ocsd        | 3           | 0.796 |  0.611 |   0.615 |       0.417 |      21.602 |
| ocsd        | 5           | 0.644 |  0.445 |   0.413 |       0.083 |      21.354 |
| ocsd        | 8+          | 0.515 |  0.351 |   0.307 |       0.028 |      20.688 |
| ocsd_final  | 1           | 1     |  0.645 | nan     |       0.889 |      21.582 |
| ocsd_final  | 3           | 0.815 |  0.498 |   0.63  |       0.306 |      22.348 |
| ocsd_final  | 5           | 0.7   |  0.416 |   0.498 |       0.111 |      22.196 |
| ocsd_final  | 8+          | 0.625 |  0.378 |   0.396 |       0.028 |      21.429 |
| ocsd_v2     | 1           | 0.944 |  0.573 | nan     |       0.861 |      22.117 |
| ocsd_v2     | 3           | 0.778 |  0.399 |   0.592 |       0.333 |      22.651 |
| ocsd_v2     | 5           | 0.7   |  0.374 |   0.467 |       0.25  |      22.383 |
| ocsd_v2     | 8+          | 0.554 |  0.285 |   0.307 |       0     |      21.574 |
| t2i_adapter | 1           | 0.972 |  0.628 | nan     |       0.944 |      21.795 |
| t2i_adapter | 3           | 0.556 |  0.415 |   0.33  |       0.25  |      21.888 |
| t2i_adapter | 5           | 0.6   |  0.435 |   0.344 |       0.056 |      21.498 |
| t2i_adapter | 8+          | 0.368 |  0.271 |   0.172 |       0     |      20.945 |
| zhang2025   | 1           | 0.778 |  0.557 | nan     |       0.778 |      20.507 |
| zhang2025   | 3           | 0.694 |  0.487 |   0.449 |       0.222 |      20.941 |
| zhang2025   | 5           | 0.606 |  0.408 |   0.335 |       0.083 |      20.993 |
| zhang2025   | 8+          | 0.455 |  0.285 |   0.228 |       0.028 |      20.477 |

## Paired tests

| family | id | a | b | metric | mean a | mean b | diff [95% CI] | W/L/T | p sign | p Holm | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | P1 | ocsd_final | ocsd | opr | 0.785 | 0.711 | +0.074 [+0.023, +0.129] | 32/12/28 | 0.0037 | 0.0110 | 0.0076 |
| primary | P2 | ocsd_final | ocsd | miou | 0.484 | 0.529 | -0.044 [-0.091, +0.002] | 30/42/0 | 0.1945 | 0.2703 | 0.0849 |
| primary | P3 | ocsd_final | ocsd | ra | 0.508 | 0.445 | +0.063 [-0.023, +0.145] | 28/17/9 | 0.1352 | 0.2703 | 0.0700 |
| primary | P4 | ocsd_final | ocsd | pickscore | 21.889 | 21.164 | +0.725 [+0.592, +0.861] | 65/7/0 | 0.0000 | 0.0000 | 0.0000 |
| secondary |  | ocsd_final | ocsd_v2 | opr | 0.785 | 0.744 | +0.041 [+0.003, +0.086] | 27/15/30 | 0.0884 | 1.0000 | 0.0946 |
| secondary |  | ocsd_final | ocsd_v2 | miou | 0.484 | 0.408 | +0.077 [+0.048, +0.107] | 55/17/0 | 0.0000 | 0.0003 | 0.0000 |
| secondary |  | ocsd_final | ocsd_v2 | ra | 0.508 | 0.455 | +0.053 [-0.009, +0.114] | 27/20/7 | 0.3817 | 1.0000 | 0.1054 |
| secondary |  | ocsd_final | ocsd_v2 | count_acc | 0.333 | 0.361 | -0.028 [-0.097, +0.049] | 7/11/54 | 0.4807 | 1.0000 | 0.4886 |
| secondary |  | ocsd_final | ocsd_v2 | sketch_chamfer | 0.061 | 0.066 | -0.005 [-0.009, -0.001] | 46/26/0 | 0.0245 | 0.4892 | 0.0286 |
| secondary |  | ocsd_final | ocsd_v2 | pickscore | 21.889 | 22.182 | -0.293 [-0.395, -0.191] | 21/51/0 | 0.0005 | 0.0139 | 0.0000 |
| secondary |  | ocsd_final | ocsd_v2 | clip | 28.687 | 28.865 | -0.177 [-0.687, +0.319] | 36/36/0 | 1.0000 | 1.0000 | 0.8751 |
| secondary |  | ocsd_final | gligen | opr | 0.785 | 0.801 | -0.017 [-0.060, +0.028] | 18/28/26 | 0.1839 | 1.0000 | 0.3275 |
| secondary |  | ocsd_final | gligen | miou | 0.484 | 0.653 | -0.168 [-0.211, -0.125] | 10/62/0 | 0.0000 | 0.0000 | 0.0000 |
| secondary |  | ocsd_final | gligen | ra | 0.508 | 0.563 | -0.055 [-0.145, +0.035] | 19/34/1 | 0.0534 | 0.8016 | 0.1245 |
| secondary |  | ocsd_final | gligen | count_acc | 0.333 | 0.465 | -0.132 [-0.229, -0.035] | 6/21/45 | 0.0059 | 0.1422 | 0.0103 |
| secondary |  | ocsd_final | gligen | sketch_chamfer | 0.061 | 0.066 | -0.005 [-0.010, +0.000] | 42/30/0 | 0.1945 | 1.0000 | 0.1228 |
| secondary |  | ocsd_final | gligen | pickscore | 21.889 | 21.409 | +0.480 [+0.301, +0.658] | 55/17/0 | 0.0000 | 0.0003 | 0.0000 |
| secondary |  | ocsd_final | gligen | clip | 28.687 | 28.230 | +0.457 [-0.326, +1.284] | 40/32/0 | 0.4096 | 1.0000 | 0.3999 |
| secondary |  | ocsd_final | collage | opr | 0.785 | 0.824 | -0.039 [-0.083, +0.006] | 12/30/30 | 0.0079 | 0.1741 | 0.0190 |
| secondary |  | ocsd_final | collage | miou | 0.484 | 0.607 | -0.123 [-0.161, -0.085] | 12/60/0 | 0.0000 | 0.0000 | 0.0000 |
| secondary |  | ocsd_final | collage | ra | 0.508 | 0.616 | -0.108 [-0.192, -0.023] | 14/36/4 | 0.0026 | 0.0651 | 0.0035 |
| secondary |  | ocsd_final | collage | count_acc | 0.333 | 0.451 | -0.118 [-0.215, -0.028] | 8/20/44 | 0.0357 | 0.6426 | 0.0151 |
| secondary |  | ocsd_final | collage | sketch_chamfer | 0.061 | 0.049 | +0.012 [+0.009, +0.015] | 12/60/0 | 0.0000 | 0.0000 | 0.0000 |
| secondary |  | ocsd_final | collage | pickscore | 21.889 | 21.518 | +0.371 [+0.237, +0.501] | 55/17/0 | 0.0000 | 0.0003 | 0.0000 |
| secondary |  | ocsd_final | collage | clip | 28.687 | 28.800 | -0.113 [-0.641, +0.413] | 33/39/0 | 0.5560 | 1.0000 | 0.8355 |
| secondary |  | ocsd_final | controlnet | opr | 0.785 | 0.655 | +0.130 [+0.080, +0.181] | 41/10/21 | 0.0000 | 0.0004 | 0.0000 |
| secondary |  | ocsd_final | controlnet | miou | 0.484 | 0.492 | -0.007 [-0.050, +0.036] | 35/37/0 | 0.9063 | 1.0000 | 0.7028 |
| secondary |  | ocsd_final | controlnet | ra | 0.508 | 0.321 | +0.188 [+0.104, +0.270] | 40/12/2 | 0.0001 | 0.0034 | 0.0003 |
| secondary |  | ocsd_final | controlnet | count_acc | 0.333 | 0.257 | +0.076 [+0.007, +0.146] | 13/4/55 | 0.0490 | 0.7847 | 0.0454 |
| secondary |  | ocsd_final | controlnet | sketch_chamfer | 0.061 | 0.038 | +0.024 [+0.020, +0.027] | 5/67/0 | 0.0000 | 0.0000 | 0.0000 |
| secondary |  | ocsd_final | controlnet | pickscore | 21.889 | 21.671 | +0.217 [+0.078, +0.358] | 48/24/0 | 0.0063 | 0.1451 | 0.0039 |
| secondary |  | ocsd_final | controlnet | clip | 28.687 | 27.843 | +0.844 [+0.235, +1.466] | 41/31/0 | 0.2888 | 1.0000 | 0.0168 |
| secondary |  | ocsd_final | t2i_adapter | opr | 0.785 | 0.624 | +0.161 [+0.108, +0.216] | 39/7/26 | 0.0000 | 0.0001 | 0.0000 |
| secondary |  | ocsd_final | t2i_adapter | miou | 0.484 | 0.437 | +0.047 [+0.006, +0.091] | 47/25/0 | 0.0128 | 0.2683 | 0.0502 |
| secondary |  | ocsd_final | t2i_adapter | ra | 0.508 | 0.282 | +0.226 [+0.139, +0.311] | 40/10/4 | 0.0000 | 0.0007 | 0.0000 |
| secondary |  | ocsd_final | t2i_adapter | count_acc | 0.333 | 0.312 | +0.021 [-0.056, +0.097] | 9/7/56 | 0.8036 | 1.0000 | 0.5898 |
| secondary |  | ocsd_final | t2i_adapter | sketch_chamfer | 0.061 | 0.035 | +0.026 [+0.022, +0.031] | 7/65/0 | 0.0000 | 0.0000 | 0.0000 |
| secondary |  | ocsd_final | t2i_adapter | pickscore | 21.889 | 21.531 | +0.357 [+0.179, +0.536] | 46/26/0 | 0.0245 | 0.4892 | 0.0005 |
| secondary |  | ocsd_final | t2i_adapter | clip | 28.687 | 28.236 | +0.451 [-0.238, +1.128] | 43/29/0 | 0.1249 | 1.0000 | 0.1340 |
| secondary |  | ocsd_final | zhang2025 | opr | 0.785 | 0.633 | +0.152 [+0.090, +0.219] | 42/7/23 | 0.0000 | 0.0000 | 0.0000 |
| secondary |  | ocsd_final | zhang2025 | miou | 0.484 | 0.435 | +0.050 [-0.003, +0.105] | 45/27/0 | 0.0444 | 0.7543 | 0.0564 |
| secondary |  | ocsd_final | zhang2025 | ra | 0.508 | 0.337 | +0.171 [+0.083, +0.255] | 41/9/4 | 0.0000 | 0.0002 | 0.0001 |
| secondary |  | ocsd_final | zhang2025 | count_acc | 0.333 | 0.278 | +0.056 [-0.028, +0.146] | 10/8/54 | 0.8145 | 1.0000 | 0.1445 |
| secondary |  | ocsd_final | zhang2025 | sketch_chamfer | 0.061 | 0.059 | +0.002 [-0.002, +0.007] | 30/42/0 | 0.1945 | 1.0000 | 0.1557 |
| secondary |  | ocsd_final | zhang2025 | pickscore | 21.889 | 20.730 | +1.159 [+0.989, +1.337] | 67/5/0 | 0.0000 | 0.0000 | 0.0000 |
| secondary |  | ocsd_final | zhang2025 | clip | 28.687 | 25.550 | +3.138 [+2.194, +4.172] | 57/15/0 | 0.0000 | 0.0000 | 0.0000 |

## Pre-registered claims

- Unit: scene (72); per-scene value = mean over seeds 0 and 1. Exact two-sided sign tests, Holm over the four primary tests P1-P4 (ocsd_final vs the original OCSD on OPR, mIoU, RA, PickScore); paired bootstrap 95% CIs; Wilcoxon p reported alongside.
- C-structure holds if P1, P2 or P3 is significant (Holm p < 0.05) in favour of ocsd_final and none of P1-P4 is significantly worse.
- C-natural holds if P4 is significant in favour of ocsd_final and neither P1 nor P2 is significantly worse.
- Comparisons with OCSD-v2, GLIGEN, Collage, ControlNet, T2I-Adapter and Zhang et al. (2025) are secondary: Holm over the secondary family is reported, and no claim of beating a baseline is made unless that adjusted p is below 0.05.
- Blind naturalness ratings (0 broken or pasted / 1 fake but readable / 2 natural) of the 36 sheets (scenes 003-005, seed 0) are made after the run by the analyst, before review_key.json is opened; they are secondary.

## Verdict

- **C-structure** (OCSD with harmonised M4 keeps objects, layout and relations better than the original OCSD): HOLDS (significant in favour: ['P1']; significantly worse: [])
- **C-natural** (OCSD with harmonised M4 is preferred by the learned human-preference model (PickScore) over the original OCSD without losing objects): HOLDS (significant in favour: ['P4']; significantly worse: [])
