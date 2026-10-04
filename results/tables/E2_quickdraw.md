**OPR (%) / OCE-lớp theo complexity trên quickdraw**

| Phương pháp                                           | simple      | medium      | complex     |
|:------------------------------------------------------|:------------|:------------|:------------|
| SD + ControlNet (mô hình nền)                         | 65.6 / 2.33 | 74.4 / 2.42 | 62.3 / 2.62 |
| SD + T2I-Adapter                                      | 57.1 / 3.04 | 75.8 / 1.92 | 62.9 / 2.92 |
| GLIGEN (hộp + văn bản)                                | 81.1 / 1.71 | 83.9 / 2.17 | 81.5 / 1.67 |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         | 71.7 / 2.79 | 73.9 / 2.00 | 58.6 / 3.62 |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      | 72.2 / 4.33 | 80.4 / 1.71 | 66.0 / 2.75 |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) | 70.8 / 2.62 | 71.8 / 2.29 | 67.8 / 2.50 |
| ControlNet, best of N (same check, compute-matched)   | 71.0 / 2.08 | 75.8 / 2.29 | 68.7 / 2.33 |
| GLIGEN, best of N (same check, compute-matched)       | 79.4 / 1.83 | 87.2 / 0.96 | 89.2 / 1.00 |
| Collage: M2 objects pasted + SDEdit                   | 85.8 / 1.00 | 84.0 / 1.42 | 82.9 / 1.96 |
| Collage, best of 3 (same check)                       | 84.8 / 1.33 | 90.1 / 1.25 | 83.6 / 1.62 |
| Zhang et al. (2025) - cài đặt lại                     | 65.5 / 1.96 | 60.2 / 2.62 | 59.0 / 2.67 |
| OCSD-lite (không học định danh)                       | 76.8 / 2.12 | 73.7 / 2.17 | 66.2 / 2.29 |
| OCSD (đề xuất)                                        | 75.0 / 1.75 | 80.2 / 2.17 | 66.8 / 2.25 |
| OCSD-v2 (joint trajectory, training-free)             | 74.4 / 4.29 | 77.4 / 2.46 | 67.9 / 3.08 |