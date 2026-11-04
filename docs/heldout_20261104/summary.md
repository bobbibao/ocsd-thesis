# Held-out comparison (automatic metrics; human review pending)

Locked protocol: docs/HELDOUT_PROTOCOL.md. 18 scenes x seeds 0/1. Means over scenes (seeds averaged) with
95% bootstrap CIs over scenes. These OWLv2/CLIP numbers are secondary; the primary endpoint is the blind
human review of review/*.png. direct_scribble was added after stage 1 and is exploratory.

| method | count_acc | opr | oce_c | miou | ra | joint_layout_success | clip | obj_clip | time_s | sampling_work |
|---|---|---|---|---|---|---|---|---|---|---|
| direct_box | 0.361 [0.194, 0.556] | 0.814 [0.740, 0.885] | 2.361 [1.333, 3.500] | 0.704 [0.633, 0.774] | 0.673 [0.553, 0.792] | 0.306 [0.139, 0.500] | 30.009 [28.619, 31.296] | 25.787 [24.911, 26.659] | 27.238 [23.734, 30.920] | 315.222 [273.000, 359.556] |
| direct_scribble | 0.389 [0.194, 0.583] | 0.833 [0.763, 0.902] | 1.944 [1.056, 2.917] | 0.749 [0.681, 0.817] | 0.698 [0.578, 0.817] | 0.361 [0.194, 0.556] | 30.284 [29.268, 31.281] | 25.433 [24.494, 26.446] | 27.130 [23.538, 30.921] | 315.222 [273.000, 359.556] |
| controlnet | 0.056 [0.000, 0.139] | 0.547 [0.431, 0.667] | 4.222 [3.000, 5.583] | 0.399 [0.308, 0.499] | 0.322 [0.177, 0.489] | 0.028 [0.000, 0.083] | 28.409 [27.174, 29.595] | 22.159 [21.241, 23.212] | 2.834 [2.806, 2.878] | 120.000 [120.000, 120.000] |
| instance_repair_count_matte_soft | 0.250 [0.083, 0.444] | 0.641 [0.510, 0.769] | 3.750 [2.250, 5.361] | 0.487 [0.378, 0.597] | 0.469 [0.309, 0.637] | 0.167 [0.028, 0.333] | 29.415 [27.648, 31.068] | 23.367 [22.062, 24.710] | 5.063 [3.821, 7.076] | 212.778 [170.000, 249.444] |

## Paired differences (a - b)

| a | b | metric | diff [95% CI] | role |
|---|---|---|---|---|
| direct_box | controlnet | count_acc | 0.306 [0.111, 0.500] | secondary (automatic) |
| direct_box | controlnet | opr | 0.267 [0.136, 0.390] | secondary (automatic) |
| direct_box | controlnet | oce_c | -1.861 [-3.000, -0.583] | secondary (automatic) |
| direct_box | controlnet | miou | 0.305 [0.191, 0.411] | secondary (automatic) |
| direct_box | controlnet | ra | 0.351 [0.144, 0.541] | secondary (automatic) |
| direct_box | controlnet | joint_layout_success | 0.278 [0.083, 0.472] | secondary (automatic) |
| direct_box | controlnet | clip | 1.600 [-0.226, 3.327] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | count_acc | 0.111 [-0.028, 0.278] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | opr | 0.173 [0.087, 0.264] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | oce_c | -1.389 [-2.528, -0.194] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | miou | 0.218 [0.148, 0.289] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | ra | 0.204 [0.096, 0.313] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | joint_layout_success | 0.139 [0.000, 0.306] | secondary (automatic) |
| direct_box | instance_repair_count_matte_soft | clip | 0.594 [-1.472, 2.654] | secondary (automatic) |
| direct_scribble | controlnet | count_acc | 0.333 [0.111, 0.556] | post-hoc exploratory |
| direct_scribble | controlnet | opr | 0.287 [0.164, 0.402] | post-hoc exploratory |
| direct_scribble | controlnet | oce_c | -2.278 [-3.417, -1.139] | post-hoc exploratory |
| direct_scribble | controlnet | miou | 0.350 [0.250, 0.442] | post-hoc exploratory |
| direct_scribble | controlnet | ra | 0.376 [0.181, 0.553] | post-hoc exploratory |
| direct_scribble | controlnet | joint_layout_success | 0.333 [0.139, 0.528] | post-hoc exploratory |
| direct_scribble | controlnet | clip | 1.874 [0.316, 3.446] | post-hoc exploratory |
| direct_scribble | instance_repair_count_matte_soft | count_acc | 0.139 [-0.056, 0.333] | post-hoc exploratory |
| direct_scribble | instance_repair_count_matte_soft | opr | 0.193 [0.110, 0.283] | post-hoc exploratory |
| direct_scribble | instance_repair_count_matte_soft | oce_c | -1.806 [-2.889, -0.694] | post-hoc exploratory |
| direct_scribble | instance_repair_count_matte_soft | miou | 0.262 [0.196, 0.333] | post-hoc exploratory |
| direct_scribble | instance_repair_count_matte_soft | ra | 0.229 [0.129, 0.343] | post-hoc exploratory |
| direct_scribble | instance_repair_count_matte_soft | joint_layout_success | 0.194 [0.056, 0.361] | post-hoc exploratory |
| direct_scribble | instance_repair_count_matte_soft | clip | 0.869 [-0.794, 2.477] | post-hoc exploratory |
| direct_scribble | direct_box | count_acc | 0.028 [-0.111, 0.167] | post-hoc exploratory |
| direct_scribble | direct_box | opr | 0.020 [-0.026, 0.065] | post-hoc exploratory |
| direct_scribble | direct_box | oce_c | -0.417 [-0.861, 0.056] | post-hoc exploratory |
| direct_scribble | direct_box | miou | 0.045 [0.001, 0.088] | post-hoc exploratory |
| direct_scribble | direct_box | ra | 0.025 [-0.053, 0.100] | post-hoc exploratory |
| direct_scribble | direct_box | joint_layout_success | 0.056 [-0.083, 0.194] | post-hoc exploratory |
| direct_scribble | direct_box | clip | 0.274 [-0.788, 1.669] | post-hoc exploratory |
