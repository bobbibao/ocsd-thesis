**Thời gian và bộ nhớ trung bình mỗi ảnh**

| Phương pháp                                           |   Thời gian (s) |   M2 (s) |   M3 (s) |   VRAM (GB) |   Số lần sinh |
|:------------------------------------------------------|----------------:|---------:|---------:|------------:|--------------:|
| SD + ControlNet (mô hình nền)                         |            2.22 |   nan    |   nan    |       12.02 |          1    |
| SD + T2I-Adapter                                      |            1.55 |   nan    |   nan    |       12.04 |          1    |
| GLIGEN (hộp + văn bản)                                |            1.92 |   nan    |   nan    |       12.03 |          1    |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         |            2.26 |   nan    |   nan    |       12.03 |          1    |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      |            3.22 |   nan    |   nan    |       12.22 |          1    |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) |            5.34 |   nan    |   nan    |       12.23 |          2.25 |
| ControlNet, best of N (same check, compute-matched)   |           11.72 |   nan    |   nan    |       12.24 |          5.06 |
| GLIGEN, best of N (same check, compute-matched)       |            3.36 |   nan    |   nan    |       12.23 |          1.62 |
| Collage: M2 objects pasted + SDEdit                   |            3.02 |     9.43 |   nan    |       12.03 |          1    |
| Collage, best of 3 (same check)                       |            5.82 |     9.43 |   nan    |       12.24 |          1.94 |
| Zhang et al. (2025) - cài đặt lại                     |            1.87 |     6.92 |    35.79 |       12.03 |          1    |
| OCSD-lite (không học định danh)                       |            4.15 |     9.43 |     0    |       12.23 |          1.75 |
| OCSD (đề xuất)                                        |            4.83 |     9.43 |    36.36 |       12.51 |          1.88 |
| OCSD-v2 (joint trajectory, training-free)             |            6.83 |     9.43 |     0    |       12.23 |          2.19 |