# Held-out v2: silhouette confirmation

Locked protocol: docs/HELDOUT2_PROTOCOL.md.

## Primary (direct_silhouette - direct_box, per scene; exact sign test, Holm over two)

| metric | scenes | wins / losses / ties | diff [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|
| facing_acc | 16 | 14 / 0 / 2 | 0.568 [0.406, 0.724] | 0.0001 | 0.0001 |
| sketch_chamfer | 18 | 18 / 0 / 0 | -0.019 [-0.024, -0.015] | 0.0000 | 0.0000 |

facing_acc: higher is better; sketch_chamfer: lower is better (a win = silhouette better).

## All arms

18 scenes x seeds 0/1. Means over scenes (seeds averaged) with 95% bootstrap CIs over scenes.
Sketch fidelity (SAM outline vs raw sketch ink) is primary for silhouette vs box; OWLv2/CLIP and the review of
review/*.png are secondary.

| method | sketch_chamfer | facing_acc | found_rate | count_acc | opr | oce_c | miou | ra | joint_layout_success | clip | obj_clip | time_s | sampling_work |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direct_silhouette | 0.041 [0.038, 0.045] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.194 [0.056, 0.361] | 0.833 [0.766, 0.898] | 2.083 [1.278, 2.972] | 0.747 [0.685, 0.809] | 0.700 [0.587, 0.812] | 0.194 [0.056, 0.361] | 30.292 [29.028, 31.531] | 25.257 [24.596, 25.975] | 26.411 [23.063, 29.934] | 313.111 [273.000, 355.333] |
| direct_box | 0.060 [0.056, 0.066] | 0.432 [0.276, 0.594] | 1.000 [1.000, 1.000] | 0.139 [0.028, 0.278] | 0.723 [0.647, 0.802] | 2.472 [1.639, 3.361] | 0.616 [0.547, 0.689] | 0.529 [0.407, 0.658] | 0.139 [0.028, 0.278] | 29.427 [27.898, 30.955] | 25.687 [24.816, 26.538] | 26.534 [23.201, 30.039] | 313.111 [273.000, 355.333] |
| controlnet | 0.037 [0.031, 0.045] | 0.969 [0.906, 1.000] | 1.000 [1.000, 1.000] | 0.056 [0.000, 0.139] | 0.485 [0.398, 0.560] | 4.139 [3.167, 5.111] | 0.342 [0.275, 0.401] | 0.201 [0.146, 0.253] | 0.000 [0.000, 0.000] | 28.946 [27.363, 30.455] | 21.737 [20.718, 22.768] | 2.779 [2.756, 2.820] | 120.000 [120.000, 120.000] |
| instance_repair_count_matte_soft | 0.065 [0.060, 0.072] | 0.552 [0.380, 0.724] | 1.000 [1.000, 1.000] | 0.167 [0.056, 0.306] | 0.644 [0.514, 0.761] | 2.972 [1.806, 4.278] | 0.487 [0.378, 0.588] | 0.456 [0.312, 0.604] | 0.139 [0.028, 0.278] | 29.865 [28.626, 31.070] | 23.111 [22.171, 24.070] | 5.158 [3.977, 7.141] | 231.111 [188.333, 267.778] |

## Paired differences (a - b)

| a | b | metric | diff [95% CI] | role |
|---|---|---|---|---|
| direct_silhouette | direct_box | sketch_chamfer | -0.019 [-0.024, -0.015] | secondary (automatic) |
| direct_silhouette | direct_box | facing_acc | 0.568 [0.406, 0.724] | secondary (automatic) |
| direct_silhouette | direct_box | count_acc | 0.056 [-0.056, 0.194] | secondary (automatic) |
| direct_silhouette | direct_box | opr | 0.110 [0.055, 0.171] | secondary (automatic) |
| direct_silhouette | direct_box | oce_c | -0.389 [-0.944, 0.194] | secondary (automatic) |
| direct_silhouette | direct_box | miou | 0.131 [0.077, 0.190] | secondary (automatic) |
| direct_silhouette | direct_box | ra | 0.171 [0.087, 0.261] | secondary (automatic) |
| direct_silhouette | direct_box | joint_layout_success | 0.056 [-0.056, 0.194] | secondary (automatic) |
| direct_silhouette | direct_box | clip | 0.865 [-0.235, 1.916] | secondary (automatic) |
| direct_silhouette | controlnet | sketch_chamfer | 0.004 [-0.004, 0.011] | secondary (automatic) |
| direct_silhouette | controlnet | facing_acc | 0.031 [0.000, 0.094] | secondary (automatic) |
| direct_silhouette | controlnet | count_acc | 0.139 [0.000, 0.306] | secondary (automatic) |
| direct_silhouette | controlnet | opr | 0.348 [0.249, 0.463] | secondary (automatic) |
| direct_silhouette | controlnet | oce_c | -2.056 [-2.778, -1.361] | secondary (automatic) |
| direct_silhouette | controlnet | miou | 0.405 [0.321, 0.499] | secondary (automatic) |
| direct_silhouette | controlnet | ra | 0.498 [0.369, 0.631] | secondary (automatic) |
| direct_silhouette | controlnet | joint_layout_success | 0.194 [0.056, 0.361] | secondary (automatic) |
| direct_silhouette | controlnet | clip | 1.346 [-0.178, 2.951] | secondary (automatic) |
| direct_silhouette | instance_repair_count_matte_soft | sketch_chamfer | -0.024 [-0.030, -0.019] | secondary (automatic) |
| direct_silhouette | instance_repair_count_matte_soft | facing_acc | 0.448 [0.276, 0.620] | secondary (automatic) |
| direct_silhouette | instance_repair_count_matte_soft | count_acc | 0.028 [-0.111, 0.167] | secondary (automatic) |
| direct_silhouette | instance_repair_count_matte_soft | opr | 0.190 [0.082, 0.309] | secondary (automatic) |
| direct_silhouette | instance_repair_count_matte_soft | oce_c | -0.889 [-1.833, -0.028] | secondary (automatic) |
| direct_silhouette | instance_repair_count_matte_soft | miou | 0.260 [0.170, 0.357] | secondary (automatic) |
| direct_silhouette | instance_repair_count_matte_soft | ra | 0.244 [0.107, 0.384] | secondary (automatic) |
| direct_silhouette | instance_repair_count_matte_soft | joint_layout_success | 0.056 [-0.111, 0.222] | secondary (automatic) |
| direct_silhouette | instance_repair_count_matte_soft | clip | 0.427 [-0.421, 1.256] | secondary (automatic) |
| direct_box | controlnet | sketch_chamfer | 0.023 [0.014, 0.032] | secondary (automatic) |
| direct_box | controlnet | facing_acc | -0.536 [-0.719, -0.328] | secondary (automatic) |
| direct_box | controlnet | count_acc | 0.083 [-0.028, 0.194] | secondary (automatic) |
| direct_box | controlnet | opr | 0.238 [0.136, 0.348] | secondary (automatic) |
| direct_box | controlnet | oce_c | -1.667 [-2.389, -1.027] | secondary (automatic) |
| direct_box | controlnet | miou | 0.273 [0.194, 0.360] | secondary (automatic) |
| direct_box | controlnet | ra | 0.327 [0.199, 0.459] | secondary (automatic) |
| direct_box | controlnet | joint_layout_success | 0.139 [0.028, 0.278] | secondary (automatic) |
| direct_box | controlnet | clip | 0.481 [-1.251, 2.390] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | sketch_chamfer | -0.005 [-0.008, -0.002] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | facing_acc | -0.120 [-0.339, 0.109] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | count_acc | -0.028 [-0.139, 0.083] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | opr | 0.080 [-0.024, 0.190] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | oce_c | -0.500 [-1.444, 0.444] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | miou | 0.128 [0.046, 0.218] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | ra | 0.073 [-0.064, 0.220] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | joint_layout_success | 0.000 [-0.111, 0.111] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | clip | -0.438 [-1.396, 0.471] | secondary (automatic) |
