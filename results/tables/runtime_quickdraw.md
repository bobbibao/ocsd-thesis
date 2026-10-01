**Thời gian và bộ nhớ trung bình mỗi ảnh**

| Phương pháp                                           |   Thời gian (s) |   M2 (s) |   M3 (s) |   VRAM (GB) |   Số lần sinh |
|:------------------------------------------------------|----------------:|---------:|---------:|------------:|--------------:|
| SD + ControlNet (mô hình nền)                         |            2.17 |   nan    |   nan    |       12.01 |          1    |
| SD + T2I-Adapter                                      |            1.54 |   nan    |   nan    |       12.01 |          1    |
| GLIGEN (hộp + văn bản)                                |            1.72 |   nan    |   nan    |       12.02 |          1    |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         |            2.25 |   nan    |   nan    |       12.02 |          1    |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      |            3.24 |   nan    |   nan    |       12.19 |          1    |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) |            5.47 |   nan    |   nan    |       12.23 |          2.37 |
| Zhang et al. (2025) - cài đặt lại                     |            2.2  |     9.04 |    36.95 |       12.02 |          1    |
| OCSD-lite (không học định danh)                       |            5.11 |    12.57 |     0    |       12.23 |          2.17 |
| OCSD (đề xuất)                                        |            5.7  |    12.41 |    36.97 |       12.49 |          2.18 |