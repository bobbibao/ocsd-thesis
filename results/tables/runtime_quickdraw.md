**Thời gian và bộ nhớ trung bình mỗi ảnh**

| Phương pháp                                           |   Thời gian (s) |   M2 (s) |   M3 (s) |   VRAM (GB) |   Số lần sinh |
|:------------------------------------------------------|----------------:|---------:|---------:|------------:|--------------:|
| SD + ControlNet (mô hình nền)                         |            3.29 |   nan    |   nan    |       11.88 |          1    |
| SD + T2I-Adapter                                      |            2.37 |   nan    |   nan    |       11.88 |          1    |
| GLIGEN (hộp + văn bản)                                |            2.63 |   nan    |   nan    |       11.89 |          1    |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         |            3.37 |   nan    |   nan    |       11.89 |          1    |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      |            4.5  |   nan    |   nan    |       12.06 |          1    |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) |            7.84 |   nan    |   nan    |       12.1  |          2.34 |
| Zhang et al. (2025) - cài đặt lại                     |            2.22 |     8.65 |    36.86 |       11.48 |          1    |
| OCSD-lite (không học định danh)                       |            4.83 |    12.57 |     0    |        7.42 |          2.1  |
| OCSD (đề xuất)                                        |            4.88 |    12.41 |    35.62 |        7.7  |          1.9  |