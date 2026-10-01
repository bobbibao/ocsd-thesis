**Thời gian và bộ nhớ trung bình mỗi ảnh**

| Phương pháp                                           |   Thời gian (s) |   M2 (s) |   M3 (s) |   VRAM (GB) |   Số lần sinh |
|:------------------------------------------------------|----------------:|---------:|---------:|------------:|--------------:|
| SD + ControlNet (mô hình nền)                         |            2.22 |   nan    |   nan    |       12.02 |          1    |
| SD + T2I-Adapter                                      |            1.58 |   nan    |   nan    |       12.03 |          1    |
| GLIGEN (hộp + văn bản)                                |            1.63 |   nan    |   nan    |       12.02 |          1    |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         |            2.32 |   nan    |   nan    |       12.02 |          1    |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      |            3.33 |   nan    |   nan    |       12.19 |          1    |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) |            5.33 |   nan    |   nan    |       12.23 |          2.25 |
| Zhang et al. (2025) - cài đặt lại                     |            2.28 |     6.92 |    38.19 |       12.02 |          1    |
| OCSD-lite (không học định danh)                       |            4.53 |     9.96 |     0    |       12.23 |          1.88 |
| OCSD (đề xuất)                                        |            5.56 |     9.43 |    38.26 |       12.49 |          2.06 |