**Thời gian và bộ nhớ trung bình mỗi ảnh**

| Phương pháp                                           |   Thời gian (s) |   M2 (s) |   M3 (s) |   VRAM (GB) |   Số lần sinh |
|:------------------------------------------------------|----------------:|---------:|---------:|------------:|--------------:|
| SD + ControlNet (mô hình nền)                         |           11.41 |   nan    |   nan    |       10.74 |          1    |
| SD + T2I-Adapter                                      |            8.29 |   nan    |   nan    |       10.74 |          1    |
| GLIGEN (hộp + văn bản)                                |           11.26 |   nan    |   nan    |       10.81 |          1    |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         |           11.42 |   nan    |   nan    |       10.81 |          1    |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      |           13.67 |   nan    |   nan    |       10.97 |          1    |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) |           29.35 |   nan    |   nan    |       11.03 |          2.42 |
| Zhang et al. (2025) - cài đặt lại                     |            2.54 |    10.6  |    37.23 |        8.13 |          1    |
| OCSD-lite (không học định danh)                       |           27.58 |   173.02 |     0    |       11.14 |          2.17 |
| OCSD (đề xuất)                                        |           10.27 |    22.62 |    37.64 |        8.27 |          1.75 |