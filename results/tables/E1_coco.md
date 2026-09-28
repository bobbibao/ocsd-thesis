**OPR (%) / OCE-lớp theo count_bin trên coco**

| Phương pháp                                           | 1            | 2-3         | 4-5        |
|:------------------------------------------------------|:-------------|:------------|:-----------|
| SD + ControlNet (mô hình nền)                         | 100.0 / 0.00 | 0.0 / 2.00  | 0.0 / 5.00 |
| SD + T2I-Adapter                                      | 100.0 / 0.00 | 0.0 / 2.00  | 0.0 / 5.00 |
| GLIGEN (hộp + văn bản)                                | 100.0 / 0.00 | 50.0 / 1.00 | 0.0 / 5.00 |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         | 100.0 / 1.00 | 50.0 / 1.00 | 0.0 / 5.00 |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      | 100.0 / 0.00 | 50.0 / 1.00 | 0.0 / 5.00 |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) | 100.0 / 0.00 | 0.0 / 2.00  | 0.0 / 5.00 |