# OCSD: sinh ảnh cảnh nhất quán đối tượng từ phác thảo và văn bản

Mã nguồn khóa luận tốt nghiệp *Object-Consistent Sketch-and-Text Guided Scene Image Generation with Diffusion Models*.
Sinh viên: Lê Hoàng Bảo (21090331), Thái Bá Cường (21050681). GVHD: TS. Nguyễn Thanh Chuyên.

Mục tiêu: từ **phác thảo cảnh + câu mô tả**, sinh ảnh có **đúng đối tượng, đúng số lượng, đúng vị trí/hình dạng và đúng quan hệ**.

## Chạy trên Google Colab (Run all)

1. Mở `notebooks/RUN_ALL.ipynb` trên Colab (File → Open notebook → GitHub → `bobbibao/ocsd-thesis`).
2. Runtime → Change runtime type → **GPU L4** (hoặc A100).
3. Ở ô 1 chọn `TIER` (`pilot` cho lần đầu), rồi Runtime → **Run all**.
4. Kết quả nằm trong Google Drive: `MyDrive/KLTN_OCSD/results/` (`summary.md`, `summary.json`, `progress.json`, `tables/`, `figures/`, `user_study/`).

Colab ngắt kết nối hoặc hết giờ: chỉ cần **Run all lại**; ảnh và độ đo đã có trên Drive được bỏ qua.
Không cần khóa Hugging Face: mọi mô hình dùng đều công khai.

Thiết kế cho **Google Colab Pro** (khoảng 100 compute unit/tháng; T4 ≈ 1,8 unit/giờ, L4 ≈ 4,8 unit/giờ).
Mọi phương pháp dùng cùng siêu tham số của tier nên so sánh vẫn công bằng.

| Tier | QuickDraw-Scenes | Học định danh (OCSD, Zhang) | COCO-Sketch | Seed | DDIM / K / M3 | Ước tính |
|---|---|---|---|---|---|---|
| `pilot` | 24 cảnh | 8 | 8 (4) | 1 | 30 / 2 / 100+100 | ~1 giờ L4, ~5 unit |
| `paper` | 96 cảnh (8/ô) | 48 | 48 (24) | 2 | 30 / 2 / 150+150 | ~12 giờ L4 (~60 unit) hoặc ~30 giờ T4 (~55 unit) |
| `full` | 300 cảnh | 180 | 200 (100) | 3 | 50 / 4 / 200+200 | cần GPU ngoài Colab Pro |

Ước tính trên là tính tay, chưa đo. Sau mỗi lượt chạy, notebook ghi `results/budget_estimate.json`
với số giờ GPU dự kiến của từng tier, tính từ thời gian đo thật trên GPU đang dùng.
Colab Pro không chạy nền: giữ tab trình duyệt mở; khi phiên hết giờ, mở lại và Run all để chạy tiếp.

Các notebook khác: `01_demo_modules.ipynb` (hình minh họa từng mô-đun cho Chương 3–4), `02_app_gradio.ipynb` (ứng dụng minh họa).

## Phương pháp (thư mục `ocsd/`)

| Mô-đun | Nội dung | Mã |
|---|---|---|
| M1 | chuẩn hóa phác thảo, tách đối tượng, mặt nạ, hộp, quan hệ, câu lệnh có nhóm token | `sketch.py` |
| M2 | K ứng viên/đối tượng bằng SD + ControlNet Scribble, Grounding DINO + SAM, chọn theo CLIP + IoU + độ tin cậy | `method.py: object_branch` |
| M3 | token định danh `<o_i>` (2 giai đoạn: nhúng → nhúng + LoRA), mất mát có mặt nạ + tách biệt chú ý L_att | `method.py: learn_identity` |
| M4 | đặt đối tượng (tìm tỉ lệ/độ dịch tối đa IoU), trộn tiềm ẩn khi t > αT, nhánh nền có câu lệnh âm theo lớp | `method.py: compose_foreground, build_scene` |
| M5 | (a) chú ý chéo giới hạn theo vùng, (b) dẫn hướng năng lượng chú ý, (c) ControlNet cấp cảnh ω, (d) kiểm tra hậu sinh + sinh lại | `attention.py`, `engine.py: energy_update`, `method.py: generate` |

Phương pháp so sánh (`methods.py`), cùng backbone SD 1.5, cùng DDIM 50 bước, CFG 7,5, câu lệnh âm và seed:
ControlNet Scribble, T2I-Adapter Sketch, GLIGEN, ControlNet + chú ý vùng (kiểu DenseDiffusion),
ControlNet + năng lượng chú ý (kiểu BoxDiff/Attend-and-Excite), ControlNet chọn tốt nhất trong 3 bằng cùng bộ kiểm tra
(đối chứng công bằng cho M5(d)), Zhang et al. (2025) cài đặt lại, OCSD-lite (không huấn luyện), OCSD.
Cắt bỏ: 12 cấu hình (`abl_*`) và khảo sát α ∈ {0,4 … 1,0}.

## Đánh giá (`metrics.py`, `report.py`)

- OPR, OCE, OCE theo lớp, tỉ lệ đếm đúng, mIoU bố cục, độ chính xác quan hệ RA. Phát hiện bằng **OWLv2**, khác Grounding DINO
  mà phương pháp dùng bên trong, để phương pháp không được lợi khi tối ưu theo chính bộ phát hiện dùng chấm điểm.
- CLIP score toàn ảnh và cấp đối tượng (CLIP ViT-L/14), ID-Sim (DINOv2), FID/KID (Inception), LPIPS đa dạng.
- Trung bình ± khoảng tin cậy 95% (bootstrap theo cảnh, lấy trung bình các seed trước), kiểm định Wilcoxon có ghép cặp
  với hiệu chỉnh Holm. Hàng "ảnh thật" trên COCO cho biết trần của bộ phát hiện.

## Kiểm thử không cần GPU

```bash
pip install -r requirements-colab.txt torch torchvision open_clip_torch
python tests/smoke_test.py          # mọi phương pháp + đánh giá + báo cáo với mô hình thu nhỏ ngẫu nhiên
python tests/test_vision_tiny.py    # lớp bao OWLv2/CLIP/SAM/DINOv2
python tests/test_coco_builder.py   # dựng COCO-Sketch trên bộ COCO giả
```
(`smoke_test.py` cần biến môi trường `SMOKE_BENCH` trỏ tới một QuickDraw-Scenes dựng ở 128 px và `CLIP_BPE` trỏ tới
`open_clip/bpe_simple_vocab_16e6.txt.gz`; xem đầu tệp.)
