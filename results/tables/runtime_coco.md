**Thời gian và bộ nhớ trung bình mỗi ảnh**

| Phương pháp                                           |   Thời gian (s) |   M2 (s) |   M3 (s) |   VRAM (GB) |   Số lần sinh |
|:------------------------------------------------------|----------------:|---------:|---------:|------------:|--------------:|
| SD + ControlNet (mô hình nền)                         |            2.5  |   nan    |   nan    |       11.87 |          1    |
| SD + T2I-Adapter                                      |            1.79 |   nan    |   nan    |       11.88 |          1    |
| GLIGEN (hộp + văn bản)                                |            4.44 |   nan    |   nan    |       11.94 |          1    |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         |            2.59 |   nan    |   nan    |       11.94 |          1    |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      |            3.34 |   nan    |   nan    |       12.04 |          1    |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) |            5.27 |   nan    |   nan    |       12.08 |          2.2  |
| Zhang et al. (2025) - cài đặt lại                     |            2.26 |     7.29 |    37.77 |       11.73 |          1    |
| OCSD-lite (không học định danh)                       |            4.95 |    10.7  |     0    |       12.08 |          1.94 |
| OCSD (đề xuất)                                        |            6.18 |    10.48 |    37.87 |       12.19 |          2.05 |