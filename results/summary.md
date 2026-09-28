# Kết quả thực nghiệm OCSD (2026-09-28 20:12:59)

Tier: pilot; backbone: sd15; seeds: [0]; bộ phát hiện đánh giá: owlv2

Phần cứng: {"python": "3.13.15", "torch": "2.11.0+cu128", "gpu": "NVIDIA A100-SXM4-80GB", "vram_gb": 79.3, "diffusers": "0.40.0", "transformers": "5.17.0", "cpu": "Intel(R) Xeon(R) CPU @ 2.20GHz", "ram_gb": 167.1}

## E1_coco

**OPR (%) / OCE-lớp theo count_bin trên coco**

| Phương pháp                                           | 1            | 2-3         | 4-5        |
|:------------------------------------------------------|:-------------|:------------|:-----------|
| SD + ControlNet (mô hình nền)                         | 100.0 / 0.00 | 0.0 / 2.00  | 0.0 / 5.00 |
| SD + T2I-Adapter                                      | 100.0 / 0.00 | 0.0 / 2.00  | 0.0 / 5.00 |
| GLIGEN (hộp + văn bản)                                | 100.0 / 0.00 | 50.0 / 1.00 | 0.0 / 5.00 |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         | 100.0 / 1.00 | 50.0 / 1.00 | 0.0 / 5.00 |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      | 100.0 / 0.00 | 50.0 / 1.00 | 0.0 / 5.00 |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) | 100.0 / 0.00 | 0.0 / 2.00  | 0.0 / 5.00 |

## E1_quickdraw

**OPR (%) / OCE-lớp theo count_bin trên quickdraw**

| Phương pháp                                           | 1            | 3           | 5           | 8+          |
|:------------------------------------------------------|:-------------|:------------|:------------|:------------|
| SD + ControlNet (mô hình nền)                         | 88.9 / 0.33  | 55.6 / 0.89 | 55.6 / 2.44 | 47.6 / 5.89 |
| SD + T2I-Adapter                                      | 100.0 / 0.00 | 55.6 / 1.78 | 53.3 / 2.78 | 44.1 / 6.44 |
| GLIGEN (hộp + văn bản)                                | 88.9 / 0.11  | 63.0 / 1.89 | 51.1 / 2.89 | 45.9 / 6.11 |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         | 88.9 / 0.22  | 55.6 / 1.89 | 51.1 / 2.44 | 45.5 / 5.33 |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      | 100.0 / 0.11 | 74.1 / 1.22 | 53.3 / 2.89 | 47.3 / 5.11 |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) | 88.9 / 0.11  | 66.7 / 1.22 | 60.0 / 2.11 | 56.9 / 5.33 |
| OCSD-lite (không học định danh)                       | 100.0 / 0.00 | 81.5 / 1.00 | 75.6 / 1.44 | 53.0 / 4.89 |

## E2_coco

**OPR (%) / OCE-lớp theo complexity trên coco**

| Phương pháp                                           | real        |
|:------------------------------------------------------|:------------|
| SD + ControlNet (mô hình nền)                         | 33.3 / 2.33 |
| SD + T2I-Adapter                                      | 33.3 / 2.33 |
| GLIGEN (hộp + văn bản)                                | 50.0 / 2.00 |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         | 50.0 / 2.33 |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      | 50.0 / 2.00 |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) | 33.3 / 2.33 |

## E2_quickdraw

**OPR (%) / OCE-lớp theo complexity trên quickdraw**

| Phương pháp                                           | simple      | medium      | complex     |
|:------------------------------------------------------|:------------|:------------|:------------|
| SD + ControlNet (mô hình nền)                         | 68.4 / 1.75 | 59.8 / 2.92 | 57.5 / 2.50 |
| SD + T2I-Adapter                                      | 63.3 / 2.58 | 67.9 / 2.92 | 58.6 / 2.75 |
| GLIGEN (hộp + văn bản)                                | 61.4 / 2.83 | 56.1 / 2.83 | 69.2 / 2.58 |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         | 69.6 / 1.83 | 58.4 / 2.58 | 52.8 / 3.00 |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      | 73.3 / 2.17 | 61.7 / 2.42 | 71.1 / 2.42 |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) | 73.1 / 1.42 | 65.1 / 3.00 | 66.1 / 2.17 |
| OCSD-lite (không học định danh)                       | 78.8 / 1.67 | 73.1 / 2.00 | 80.6 / 1.83 |

## E3_coco

**So sánh định lượng trên coco (3 cảnh, trung bình ± nửa KTC 95%)**

| Phương pháp                                           | OPR (%) ↑       | OCE-lớp ↓       | Đếm đúng (%) ↑   | mIoU ↑            | RA (%) ↑      | CLIP ↑           | CLIP-đt ↑        | ID-Sim ↑   | FID ↓   | KID×10³ ↓   |
|:------------------------------------------------------|:----------------|:----------------|:-----------------|:------------------|:--------------|:-----------------|:-----------------|:-----------|:--------|:------------|
| SD + ControlNet (mô hình nền)                         | 33.3 ± 50.0     | 2.33 ± 2.50     | **33.3 ± 50.0**  | 0.321 ± 0.482     | **0.0 ± 0.0** | 23.88 ± 2.66     | 20.26 ± 2.68     | –          | –       | –           |
| SD + T2I-Adapter                                      | 33.3 ± 50.0     | 2.33 ± 2.50     | **33.3 ± 50.0**  | 0.308 ± 0.462     | **0.0 ± 0.0** | **24.75 ± 4.30** | 18.78 ± 2.58     | –          | –       | –           |
| GLIGEN (hộp + văn bản)                                | **50.0 ± 50.0** | **2.00 ± 2.50** | **33.3 ± 50.0**  | 0.452 ± 0.484     | **0.0 ± 0.0** | 21.67 ± 3.15     | 19.45 ± 4.75     | –          | –       | –           |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         | **50.0 ± 50.0** | 2.33 ± 2.00     | 0.0 ± 0.0        | **0.480 ± 0.483** | **0.0 ± 0.0** | 21.63 ± 2.62     | 20.22 ± 4.09     | –          | –       | –           |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      | **50.0 ± 50.0** | **2.00 ± 2.50** | **33.3 ± 50.0**  | 0.470 ± 0.484     | **0.0 ± 0.0** | 22.16 ± 1.69     | **20.37 ± 3.50** | –          | –       | –           |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) | 33.3 ± 50.0     | 2.33 ± 2.50     | **33.3 ± 50.0**  | 0.321 ± 0.482     | **0.0 ± 0.0** | 23.78 ± 2.82     | 19.79 ± 3.39     | –          | –       | –           |

## E3_quickdraw

**So sánh định lượng trên quickdraw (36 cảnh, trung bình ± nửa KTC 95%)**

| Phương pháp                                           | OPR (%) ↑      | OCE-lớp ↓       | Đếm đúng (%) ↑   | mIoU ↑            | RA (%) ↑        | CLIP ↑           | CLIP-đt ↑        | ID-Sim ↑          | FID ↓      | KID×10³ ↓   |
|:------------------------------------------------------|:---------------|:----------------|:-----------------|:------------------|:----------------|:-----------------|:-----------------|:------------------|:-----------|:------------|
| SD + ControlNet (mô hình nền)                         | 61.9 ± 10.4    | 2.39 ± 0.89     | **36.1 ± 16.7**  | 0.463 ± 0.091     | 30.8 ± 11.3     | 28.21 ± 0.96     | 22.54 ± 1.03     | –                 | 310.52     | 43.76       |
| SD + T2I-Adapter                                      | 63.2 ± 10.3    | 2.75 ± 0.94     | 33.3 ± 13.9      | 0.533 ± 0.096     | 27.8 ± 11.8     | 27.52 ± 0.99     | 22.27 ± 0.95     | –                 | 322.64     | 69.59       |
| GLIGEN (hộp + văn bản)                                | 62.2 ± 9.7     | 2.75 ± 0.90     | 27.8 ± 13.9      | 0.487 ± 0.086     | 28.1 ± 10.5     | 28.74 ± 1.00     | 23.04 ± 0.96     | –                 | 294.18     | **35.43**   |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         | 60.3 ± 10.5    | 2.47 ± 0.82     | 27.8 ± 13.9      | 0.463 ± 0.091     | 28.2 ± 10.7     | 27.82 ± 1.06     | 23.06 ± 0.98     | –                 | 305.99     | 50.34       |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      | 68.7 ± 8.9     | 2.33 ± 0.69     | 30.6 ± 15.3      | 0.507 ± 0.084     | 33.8 ± 11.2     | **29.44 ± 0.85** | 23.59 ± 0.98     | –                 | **286.77** | 35.65       |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) | 68.1 ± 9.4     | 2.19 ± 0.76     | 33.3 ± 15.3      | 0.537 ± 0.083     | 39.3 ± 11.5     | 28.06 ± 1.00     | 22.79 ± 0.87     | –                 | 297.84     | 42.49       |
| OCSD-lite (không học định danh)                       | **77.5 ± 7.2** | **1.83 ± 0.65** | **36.1 ± 16.7**  | **0.540 ± 0.073** | **46.9 ± 10.0** | 29.38 ± 0.99     | **23.95 ± 0.84** | **0.372 ± 0.053** | 312.71     | 62.87       |

## detector_ceiling_coco

**Độ đo trên chính ảnh thật (trần của bộ phát hiện)**

| Phương pháp                  | OPR (%) ↑       | OCE-lớp ↓       | mIoU ↑            | RA (%) ↑       | CLIP ↑           |
|:-----------------------------|:----------------|:----------------|:------------------|:---------------|:-----------------|
| Ảnh thật (trần bộ phát hiện) | **53.1 ± 22.4** | **7.38 ± 8.94** | **0.450 ± 0.234** | **8.0 ± 10.8** | **26.20 ± 4.77** |

## runtime_coco

**Thời gian và bộ nhớ trung bình mỗi ảnh**

| Phương pháp                                           |   Thời gian (s) |   M2 (s) |   M3 (s) |   VRAM (GB) |   Số lần sinh |
|:------------------------------------------------------|----------------:|---------:|---------:|------------:|--------------:|
| SD + ControlNet (mô hình nền)                         |           11.38 |      nan |      nan |       10.94 |          1    |
| SD + T2I-Adapter                                      |            8.37 |      nan |      nan |       10.95 |          1    |
| GLIGEN (hộp + văn bản)                                |            9.24 |      nan |      nan |       10.94 |          1    |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         |           11.46 |      nan |      nan |       10.94 |          1    |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      |           13.57 |      nan |      nan |       11.1  |          1    |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) |           19.71 |      nan |      nan |       11.14 |          1.67 |

## runtime_quickdraw

**Thời gian và bộ nhớ trung bình mỗi ảnh**

| Phương pháp                                           |   Thời gian (s) |   M2 (s) |   M3 (s) |   VRAM (GB) |   Số lần sinh |
|:------------------------------------------------------|----------------:|---------:|---------:|------------:|--------------:|
| SD + ControlNet (mô hình nền)                         |           11.41 |   nan    |      nan |       10.74 |          1    |
| SD + T2I-Adapter                                      |            8.29 |   nan    |      nan |       10.74 |          1    |
| GLIGEN (hộp + văn bản)                                |           11.26 |   nan    |      nan |       10.81 |          1    |
| ControlNet + chú ý vùng (kiểu DenseDiffusion)         |           11.42 |   nan    |      nan |       10.81 |          1    |
| ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)      |           13.67 |   nan    |      nan |       10.97 |          1    |
| ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra) |           29.35 |   nan    |      nan |       11.03 |          2.42 |
| OCSD-lite (không học định danh)                       |           27.58 |   173.02 |        0 |       11.14 |          2.17 |

