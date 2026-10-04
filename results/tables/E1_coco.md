**OPR (%) / OCE-lớp theo count_bin trên coco**

| Phương pháp                                           | 1           | 2-3         | 4-5         | 6-8         |
|:------------------------------------------------------|:------------|:------------|:------------|:------------|
| SD + ControlNet (mô hình nền)                         | 50.0 / 0.50 | 37.5 / 1.25 | 12.5 / 3.75 | 41.7 / 5.00 |
| SD + T2I-Adapter                                      | 50.0 / 0.50 | 37.5 / 1.25 | 12.5 / 3.50 | 45.8 / 4.50 |
| GLIGEN (hộp + văn bản)                                | 75.0 / 0.25 | 50.0 / 0.75 | 31.2 / 3.25 | 57.7 / 3.25 |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         | 50.0 / 0.50 | 37.5 / 1.25 | 12.5 / 3.50 | 50.0 / 3.75 |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      | 50.0 / 0.50 | 37.5 / 1.00 | 18.8 / 3.25 | 45.8 / 4.50 |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) | 50.0 / 0.50 | 37.5 / 1.25 | 6.2 / 3.75  | 41.7 / 5.00 |
| ControlNet, best of N (same check, compute-matched)   | 50.0 / 0.50 | 37.5 / 1.25 | 6.2 / 3.75  | 50.0 / 4.00 |
| GLIGEN, best of N (same check, compute-matched)       | 75.0 / 0.25 | 50.0 / 0.75 | 31.2 / 3.25 | 78.6 / 1.50 |
| Collage: M2 objects pasted + SDEdit                   | 75.0 / 0.75 | 50.0 / 1.00 | 18.8 / 3.25 | 66.1 / 3.25 |
| Collage, best of 3 (same check)                       | 75.0 / 0.75 | 62.5 / 0.75 | 18.8 / 3.25 | 70.2 / 3.00 |
| Zhang et al. (2025) - cài đặt lại                     | 25.0 / 0.75 | 62.5 / 0.75 | 18.8 / 3.25 | 53.6 / 3.50 |
| OCSD-lite (không học định danh)                       | 75.0 / 0.25 | 62.5 / 0.75 | 18.8 / 3.25 | 53.6 / 3.25 |
| OCSD (đề xuất)                                        | 50.0 / 0.50 | 62.5 / 1.00 | 18.8 / 3.25 | 53.6 / 3.25 |
| OCSD-v2 (joint trajectory, training-free)             | 75.0 / 0.25 | 50.0 / 1.25 | 18.8 / 3.25 | 57.7 / 3.75 |