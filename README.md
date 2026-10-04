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
| `paper` | 72 cảnh (6/ô, seed thứ 2 trên tập học định danh) | 36 | 32 (16) | 2 | 30 / 2 / 100+100 | ~8,4 giờ A100 đo thực + ~0,9 giờ tinh chỉnh |
| `full` | 300 cảnh | 180 | 200 (100) | 3 | 50 / 4 / 200+200 | cần GPU ngoài Colab Pro |

Tier `paper`/`full` chạy thêm giai đoạn **T (tinh chỉnh)**: chọn α và cường độ LoRA trên các cảnh của pilot (`results/tuning/tuned.json`), rồi loại các cảnh này khỏi tập đánh giá; kết quả pilot được chuyển vào `results/pilot/`.

Ước tính của `paper` lấy từ thời gian đo trên A100 trong lượt pilot. Sau mỗi lượt chạy, notebook ghi `results/budget_estimate.json`
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
Cắt bỏ: 14 cấu hình (`abl_*`) và khảo sát α ∈ {0,4 … 1,0}.

### Final configuration (frozen code defaults, `config.py: OCSDConfig`)

| Setting | Value | Where it comes from |
|---|---|---|
| α (latent blending while t > αT) | 0.1 | tuning phase 1 (`results/tuning/tuned.json`) |
| LoRA strength at inference | 0.5 | tuning phase 1 |
| M5(b) energy guidance | on object-phrase tokens, η = 20, τ = 10 | tuning phase 2 |
| Caption in the global prompt P_g | on (only changes COCO-Sketch prompts) | plan item 2.6 |
| M5(a) region attention | λ0 = 8, γ = 1 | thesis Table 3.x |
| M5(c) scene ControlNet / M5(d) verification | ω = 0.4 / R = 2, threshold 0.35, λ0 × 1.5 | thesis Table 3.x |

Tiers `paper` and `full` still read `results/tuning/tuned.json` on top of these defaults. Tuning phase 3 (M5(a) / M5(b)
x α) may switch off one of M5(a) / M5(b) and change α; its result is in `tuned.json` (`method_m5`). Zhang et al. (2025)
keeps its own setup (α 0.5, LoRA 1.0, no M5, no caption) apart from α, which the baseline phase tunes. The tests run on
the `paper` tier's power job are fixed in advance in `docs/PREREGISTRATION.md`.

### OCSD-v2 (`methods.V2`, `blend_mode="joint"`)

At α = 0.1 the thesis sampler hands the latent to the scene prompt, the scene ControlNet and M5 only for the last
4 of 30 U-Net calls (t ≤ 100): the background is denoised alone and the objects are pasted back until then. OCSD-v2
keeps the same ingredients but changes how they are combined:

| Change | Code |
|---|---|
| One trajectory: the U-Net sees the composite at every step, with the scene prompt (+ caption) and the scene ControlNet; the M2 objects are re-imposed inside their masks while t > αT | `method._denoise`, `blend_mode="joint"` |
| Anchor masks shrink from each object's border (a fraction of its own inner radius), so the model redraws contact and edges | `method.anchor_masks`, `anchor_shrink` |
| M5(a)/(b) act from the first step; energy averaged over objects (scale independent of the object count) | `energy_reduce="mean"` |
| Small objects keep their strongest cell at coarse attention resolutions instead of being suppressed there | `region_min_cell` |
| Caption words that name a class only attend inside that class's masks | `caption_class_masks` |
| M5(d) re-denoises only the regions of missing objects and extra detections | `method.repair_scene`, `repair` |
| No per-scene training (M3 off); M2 objects drawn per seed, so seeds are independent samples | `use_identity=False`, `m2_per_seed` |

Stage T tunes α x anchor_shrink for OCSD-v2 on the tuning split (`chosen_v2`). Ablations: `v2abl_*` (E4v2).

### Baselines added for a fair comparison

- `collage`, `collage_bo3`: the M2 objects pasted onto a ControlNet image and harmonised by SDEdit (± the same
  best-of-3 check): what a plain cut-and-paste pipeline achieves with OCSD's objects.
- `controlnet_bon`, `gligen_bon`: best of up to N = 8 samples with the same Grounding DINO check, roughly matching
  OCSD's compute.
- Stage T also tunes one knob per baseline with OCSD's rule (ControlNet scale, T2I-Adapter scale, GLIGEN β, λ0 of
  the region baseline, η of the energy baseline, collage strength, Zhang et al.'s α).

### Evaluation additions

- Robustness table (`tables/robustness_*`): OPR of the same images at IoU 0.5 as well as 0.1, under OWLv2 with every
  benchmark class as a competing query and under a COCO-trained closed-set detector (DETR; tree, house and rabbit have
  no COCO class), plus relation accuracy and IoU over detected objects only, and Kendall's τ between method rankings.
- `power_per_cell` (`config.TIERS`) enlarges the power set without changing the E3 / E4 scenes.
- Control-vs-quality sweeps (job `pareto`, figure `pareto_quickdraw.png`): OPR against KID and global CLIP along α
  (OCSD-v2, thesis sampler), ControlNet scale and GLIGEN β.
- Statistics in separate Holm families (`stats_*.csv`, column `family`); pre-registered tests in `prereg_*`.
- Realism A/B kit on randomly drawn scenes (`results/user_study_realism/`).
- `SKIP_JOBS` (notebook cell 1) leaves jobs out by name prefix, e.g. `pareto` or `pareto,E3_quickdraw_power`.

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
python tests/test_tune.py           # pilot -> paper flow: tuning phases 1-3, stale outputs, power job, report
python tests/test_prompt.py         # caption in the global prompt P_g
python tests/test_v2.py             # OCSD-v2, repair, collage / best-of-N baselines, new report functions
python tests/test_v2_ablation.py    # the 10 OCSD-v2 ablation rows end to end on a 3- and a 5-object scene
```
(`smoke_test.py` cần biến môi trường `SMOKE_BENCH` trỏ tới một QuickDraw-Scenes dựng ở 128 px và `CLIP_BPE` trỏ tới
`open_clip/bpe_simple_vocab_16e6.txt.gz`; xem đầu tệp.)
