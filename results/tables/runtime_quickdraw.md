**Thời gian và bộ nhớ trung bình mỗi ảnh**

| Phương pháp                                           |   Thời gian (s) |   M2 (s) |   M3 (s) |   VRAM (GB) |   Số lần sinh |
|:------------------------------------------------------|----------------:|---------:|---------:|------------:|--------------:|
| SD + ControlNet (mô hình nền)                         |            2.16 |   nan    |   nan    |       11.99 |          1    |
| SD + T2I-Adapter                                      |            1.51 |   nan    |   nan    |       12.04 |          1    |
| GLIGEN (hộp + văn bản)                                |            1.89 |   nan    |   nan    |       12.03 |          1    |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         |            2.22 |   nan    |   nan    |       12.03 |          1    |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      |            3.18 |   nan    |   nan    |       12.22 |          1    |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) |            5.21 |   nan    |   nan    |       12.23 |          2.26 |
| ControlNet, best of N (same check, compute-matched)   |           11.4  |   nan    |   nan    |       12.24 |          5.01 |
| GLIGEN, best of N (same check, compute-matched)       |            7.53 |   nan    |   nan    |       12.24 |          3.72 |
| Collage: M2 objects pasted + SDEdit                   |            2.95 |    12.32 |   nan    |       12.03 |          1    |
| Collage, best of 3 (same check)                       |            5.95 |    12.32 |   nan    |       12.24 |          2.03 |
| Zhang et al. (2025) - cài đặt lại                     |            1.84 |     9.04 |    35.57 |       12.03 |          1    |
| OCSD-lite (không học định danh)                       |            4.4  |    12.41 |     0    |        8.22 |          1.92 |
| OCSD (đề xuất)                                        |            4.87 |    12.41 |    35.59 |        8.5  |          1.9  |
| OCSD-v2 (joint trajectory, training-free)             |            6.86 |    12.32 |     0    |       12.24 |          2.24 |