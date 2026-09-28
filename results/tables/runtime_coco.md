**Thời gian và bộ nhớ trung bình mỗi ảnh**

| Phương pháp                                           |   Thời gian (s) |   M2 (s) |   M3 (s) |   VRAM (GB) |   Số lần sinh |
|:------------------------------------------------------|----------------:|---------:|---------:|------------:|--------------:|
| SD + ControlNet (mô hình nền)                         |           11.38 |      nan |      nan |       10.94 |          1    |
| SD + T2I-Adapter                                      |            8.37 |      nan |      nan |       10.95 |          1    |
| GLIGEN (hộp + văn bản)                                |            9.24 |      nan |      nan |       10.94 |          1    |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         |           11.46 |      nan |      nan |       10.94 |          1    |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      |           13.57 |      nan |      nan |       11.1  |          1    |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) |           19.71 |      nan |      nan |       11.14 |          1.67 |