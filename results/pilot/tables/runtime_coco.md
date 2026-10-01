**Thời gian và bộ nhớ trung bình mỗi ảnh**

| Phương pháp                                           |   Thời gian (s) |   M2 (s) |   M3 (s) |   VRAM (GB) |   Số lần sinh |
|:------------------------------------------------------|----------------:|---------:|---------:|------------:|--------------:|
| SD + ControlNet (mô hình nền)                         |            6.04 |   nan    |   nan    |       10.77 |          1    |
| SD + T2I-Adapter                                      |            4.45 |   nan    |   nan    |       10.77 |          1    |
| GLIGEN (hộp + văn bản)                                |            7.4  |   nan    |   nan    |       11.07 |          1    |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         |            6.09 |   nan    |   nan    |       11.07 |          1    |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      |            7.52 |   nan    |   nan    |       11.23 |          1    |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) |           12.89 |   nan    |   nan    |       11.27 |          2.38 |
| Zhang et al. (2025) - cài đặt lại                     |            2.54 |    13.62 |    37.73 |       10.54 |          1    |
| OCSD-lite (không học định danh)                       |            9.03 |    20.59 |     0    |       11.04 |          2.25 |
| OCSD (đề xuất)                                        |           12.89 |    19.88 |    37.93 |       11.01 |          2.75 |