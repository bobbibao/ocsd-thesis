**OPR (%) / OCE-lớp theo complexity trên quickdraw**

| Phương pháp                                           | simple      | medium      | complex     |
|:------------------------------------------------------|:------------|:------------|:------------|
| SD + ControlNet (mô hình nền)                         | 65.6 / 2.33 | 74.4 / 2.42 | 62.3 / 2.62 |
| SD + T2I-Adapter                                      | 56.8 / 3.12 | 68.4 / 2.38 | 56.2 / 3.17 |
| GLIGEN (hộp + văn bản)                                | 67.2 / 2.46 | 79.2 / 2.50 | 72.2 / 2.46 |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         | 71.7 / 2.96 | 73.4 / 2.25 | 58.9 / 3.38 |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      | 71.3 / 3.62 | 72.6 / 2.54 | 67.7 / 3.00 |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) | 70.8 / 2.62 | 71.8 / 2.29 | 67.8 / 2.50 |
| Zhang et al. (2025) - cài đặt lại                     | 46.5 / 3.29 | 46.5 / 3.00 | 39.8 / 3.96 |
| OCSD-lite (không học định danh)                       | 68.5 / 1.79 | 69.5 / 2.25 | 61.0 / 2.71 |
| OCSD (đề xuất)                                        | 66.2 / 2.04 | 66.5 / 2.38 | 54.7 / 2.79 |