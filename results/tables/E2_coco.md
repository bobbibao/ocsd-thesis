**OPR (%) / OCE-lớp theo complexity trên coco**

| Phương pháp                                           | real        |
|:------------------------------------------------------|:------------|
| SD + ControlNet (mô hình nền)                         | 35.4 / 2.62 |
| SD + T2I-Adapter                                      | 35.4 / 2.62 |
| GLIGEN (hộp + văn bản)                                | 51.4 / 1.88 |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         | 43.8 / 2.12 |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      | 40.6 / 2.50 |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) | 33.9 / 2.62 |
| Zhang et al. (2025) - cài đặt lại                     | 49.5 / 2.50 |
| OCSD-lite (không học định danh)                       | 47.8 / 2.06 |
| OCSD (đề xuất)                                        | 51.4 / 2.00 |