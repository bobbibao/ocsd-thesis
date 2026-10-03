# Những điểm mã nguồn khác với bản thảo khóa luận v1 (cần cập nhật Chương 3–4)

Tệp này dành cho người viết khóa luận: mỗi mục nêu bản thảo đang viết gì, mã làm gì và lý do.

## Chương 3 – Phương pháp

1. **Backbone: SD 1.5 thay cho SD 2.1.** ControlNet Scribble v1.1, T2I-Adapter Sketch và GLIGEN chỉ có bản chính thức cho SD 1.x,
   nên dùng SD 1.5 để mọi phương pháp chạy chung một backbone. SD 2.1 vẫn chọn được (`BACKBONE="sd21"`), khi đó không có
   T2I-Adapter và GLIGEN.
2. **M3: S1 = S2 = 200 bước ở cấu hình gốc, 150 + 150 ở tier `paper`** (bản thảo ghi 400/400), để vừa ngân sách GPU của Colab; LoRA hạng 16 trên to_q/k/v/out.
   Các tham số khác giữ như Bảng 3.x: tốc độ học nhúng 5e-3 ở giai đoạn 1, 5e-5 ở giai đoạn 2; LoRA 1e-4; λ_att = 0,01.
   Ảnh ghép để tính L_att: 2–3 đối tượng đặt cạnh nhau trên nền trắng, tối đa 6 ảnh ghép mỗi cảnh, lấy mẫu với xác suất 0,35.
3. **M2: 30 bước khử nhiễu cho ảnh đối tượng** (ảnh cảnh vẫn 50 bước); câu lệnh "a photo of a {cụm từ}, simple white background".
4. **M4: nhánh nền dùng câu lệnh âm chứa tên các lớp tiền cảnh**, để nền không tự sinh thêm đối tượng cùng lớp (tránh đếm thừa).
   Áp dụng cho cả OCSD và bản cài đặt lại Zhang et al. để so sánh công bằng.
5. **M4: cách tính tương đương.** Ở giai đoạn trộn, z_t bị ghi đè bằng phép trộn ở mỗi bước, nên chỉ cần khử nhiễu nhánh nền;
   U-Net của cảnh chỉ chạy ở bước trộn cuối cùng. Kết quả giống hệt Thuật toán M4 nhưng nhanh hơn.
6. **M5(b): dùng bản đồ chú ý theo Attend-and-Excite** (bỏ token <sot>, nhân 100 rồi softmax trên token văn bản, làm mịn Gauss 3×3),
   η_t giảm tuyến tính từ η xuống η/2 trong τ bước.
7. **M5(d): kiểm tra bằng Grounding DINO** (ngưỡng 0,35), sinh lại tối đa R = 2 lần với seed mới và λ0 × 1,5; trả về ảnh có điểm
   `OPR + 0,5·mIoU + 0,25·RA − 0,25·OCE_lớp/N` cao nhất.
8. M2, M3 được làm **một lần cho mỗi cảnh** (seed chuẩn bị = 0) và dùng lại cho mọi seed và mọi biến thể cắt bỏ cùng khóa chuẩn bị.

## Chương 4 – Thực nghiệm

1. **Dữ liệu.** Thay Sketchy/SketchyCOCO/FS-COCO bằng hai bộ tự dựng, tải tự động:
   - *QuickDraw-Scenes*: phác thảo vẽ tay thật từ Google QuickDraw (28 lớp trùng với COCO), đặt theo bố cục có phối cảnh.
     Ma trận 4 mức số đối tượng (1, 3, 5, 8–10) × 3 mức độ phức tạp. Độ phức tạp được định nghĩa cụ thể:
     *đơn giản* = nét chi tiết (tercile trên theo số điểm + 10 × số nét), được QuickDraw nhận dạng, không chồng lấn;
     *trung bình* = tercile giữa, chồng lấn IoU ≤ 0,15; *phức tạp* = tercile dưới (trừu tượng nhất), chồng lấn ≤ 0,30,
     nhiễu nét σ = 2,5 px, nét mảnh hơn. 40% đối tượng là bản sao cùng lớp và cùng cụm từ ("three white sheep") để kiểm tra lỗi đếm.
     Câu mô tả gộp số đếm ("a brown horse and two white sheep in a green meadow") để baseline cũng có thông tin số lượng.
   - *COCO-Sketch*: ảnh COCO val2017 có 1–8 đối tượng nổi bật (≥ 1,5% diện tích, không bị cắt), phác thảo PiDiNet
     (scribble), mặt nạ phân đoạn thật, câu mô tả do người viết. Có ảnh thật để tính FID/KID.
   FS-COCO có thể bổ sung sau nếu nhóm tải về Drive.
2. **Phương pháp so sánh.** FineControlNet không có mã công khai cho SD 1.5 nên được thay bằng
   "ControlNet + chú ý vùng" (cơ chế kiểu DenseDiffusion, cùng mặt nạ phác thảo). Thêm GLIGEN (hộp + văn bản),
   "ControlNet + năng lượng chú ý" (kiểu BoxDiff / Attend-and-Excite) và
   **"ControlNet chọn tốt nhất trong 3 bằng cùng bộ kiểm tra"**: đối chứng công bằng cho M5(d), vì OCSD được sinh lại tối đa 3 lần.
3. **Độ đo.** Bộ phát hiện đánh giá là **OWLv2** (ngưỡng 0,30), khác Grounding DINO mà phương pháp dùng bên trong
   (bản thảo ghi Grounding DINO 0,35). Thêm OCE theo lớp (Σ_c |N_gen,c − N_in,c|, chặt hơn OCE tổng) và tỉ lệ đếm đúng.
   CLIP cấp đối tượng cắt theo hộp phác thảo (không phụ thuộc bộ phát hiện). Thêm hàng "ảnh thật" trên COCO làm trần bộ phát hiện.
4. **Thống kê.** Mỗi cảnh chạy 2 seed (tier `paper`); số liệu là trung bình ± nửa khoảng tin cậy 95% (bootstrap theo cảnh);
   so sánh OCSD với từng phương pháp bằng kiểm định Wilcoxon có ghép cặp, hiệu chỉnh Holm (`tables/stats_*.csv`).
5. **Quy mô (tier `paper`, vừa ngân sách Colab Pro).** Mọi phương pháp dùng DDIM 30 bước (bản thảo ghi 50), M2 K = 2 ứng viên
   với 20 bước, M3 150 + 150 bước; 2 seed. QuickDraw-Scenes 96 cảnh (8/ô); OCSD và Zhang et al. (cần học định danh) chạy
   trên tập con phân tầng 48 cảnh, các phương pháp không huấn luyện chạy trên toàn bộ 96 cảnh (bảng `E3all`); COCO-Sketch
   48 cảnh (24 cảnh cho OCSD/Zhang, 1 seed); cắt bỏ 24 cảnh và khảo sát α 16 cảnh (3 và 5 đối tượng, 1 seed), lấy trong
   tập cảnh đã học định danh. Tier `full` giữ cấu hình Bảng 3.x (50 bước, K = 4, 200 + 200) nếu có thêm GPU.
6. **Bảng/hình sinh tự động** (trong `results/tables` và `results/figures`): `E3_*` (Bảng so sánh), `E3all_*`, `E1_*`, `E2_*`
   (Bảng E1–E2), `E4_*` (cắt bỏ), `alpha_*`, `runtime_*`, `detector_ceiling_coco`, `opr_curves_quickdraw.png`,
   `alpha_quickdraw.png`, `qualitative_*.png`, `benchmark_examples.png`; bộ khảo sát người dùng ẩn danh trong `results/user_study`.
7. **Tinh chỉnh sau pilot (28/09/2026).** Pilot cho thấy α = 0,5 quá cao (α càng nhỏ OPR càng tăng) và OCSD đầy đủ
   kém OCSD-lite về OPR trên QuickDraw. Tier `paper` thêm giai đoạn T: lưới α ∈ {0; 0,1; 0,2; 0,3; 0,5} × cường độ LoRA
   ∈ {0,5; 1,0}, 2 seed, trên 12 cảnh QuickDraw + 4 cảnh COCO của pilot; chọn cấu hình có OPR + mIoU + RA lớn nhất
   với CLIP không giảm quá 1 điểm so với mặc định. Các cảnh pilot bị loại khỏi tập đánh giá (tập tinh chỉnh tách
   biệt tập kiểm tra). Mục 3.x (siêu tham số) và 4.x (thiết lập) cần ghi rõ quy trình này và giá trị được chọn
   (`results/tuning/tuned.json`). Quy mô `paper` mới: 72 cảnh QuickDraw (6/ô), 36 cảnh học định danh, 32 cảnh COCO
   (16 học định danh), cắt bỏ 18 cảnh, α 12 cảnh; M3 100 + 100 bước; seed thứ 2 chạy trên tập học định danh.
8. **Tinh chỉnh dẫn hướng năng lượng M5(b) (29/09/2026).** Ở lượt `paper` đầu, bảng cắt bỏ cho thấy M5(b) áp lên token
   định danh `<o_i>` làm giảm mạnh OPR (59,6% → 78,1% khi bỏ). Giai đoạn T có thêm pha 2 trên cùng tập tinh chỉnh:
   {tắt; token định danh η = 20 (cũ), η = 10; token cụm từ đối tượng η = 20, η = 10}, cùng quy tắc chọn. Chỉ các
   dòng họ OCSD (OCSD, OCSD-lite, cắt bỏ, α) được sinh lại; baseline giữ nguyên. Tham số `energy_tokens` mới
   ("id" | "phrase") cần mô tả ở mục M5(b); giá trị chọn nằm trong `results/tuning/tuned.json`.
9. **Kết quả cuối và hai sửa lỗi đánh giá (02/10/2026).** Pha 2 chọn token cụm từ đối tượng, η = 20 (cấu hình cũ
   xếp cuối trong năm lựa chọn). (a) Bảng chỉ dùng các cặp (cảnh, seed) của kế hoạch `paper`
   (`results/<split>/plan.json`): một lượt pilot chạy sau khi tách tập tinh chỉnh đã thêm dòng baseline cho các cảnh
   tinh chỉnh, làm lệch trần bộ phát hiện và bảng thời gian. (b) FID/KID ở bảng E3 tính trên cùng một tập ảnh cho mọi
   phương pháp (tập học định danh × các seed), vì FID phụ thuộc số ảnh; trước đó baseline 108 ảnh, OCSD 72 ảnh.
   Mục 4.x cần nêu rõ: FID/KID so sánh trên cùng số ảnh; FID tuyệt đối cao do số ảnh nhỏ (72 trên QuickDraw, 16 trên COCO).
10. **Model improvements after the final run (04/10/2026; not run yet).** (a) Stage T has a phase 3 on the same tuning
    split: {M5(a) + M5(b), M5(a) only, M5(b) only} × α ∈ {0.1, 0.3, 0.4, 0.5, 0.6}, same selection rule, with the
    phase-2 setting as reference (plan 2.3). (b) On COCO-Sketch the global prompt P_g now ends with the human caption
    (`use_caption`, group of free tokens: no region bias, no energy), because P_g built from class names and the
    background phrase lost the caption's meaning and global CLIP dropped to 19–20 (plan 2.6). It is a no-op on
    QuickDraw-Scenes, whose caption is built from the phrases. It also applies to the "ControlNet + region attention" and
    "ControlNet + attention energy" baselines, not to Zhang et al. (kept as re-implemented). (c) The final configuration
    is frozen as the code defaults: α 0.1, LoRA 0.5, energy on phrase tokens with η = 20, caption in P_g (plan 3.1).
    (d) Job `E3_quickdraw_power`: OCSD, OCSD-lite, GLIGEN and ControlNet on all 72 QuickDraw scenes with both seeds,
    with the tests fixed in advance in `docs/PREREGISTRATION.md` (tables `E3power_*`, `E3power8_*`, `prereg_*`;
    plan 2.4). (e) Ablation rows "+ both M5(a) and M5(b)" and "+ M3 with the full 200 + 200 steps" (plan 2.7); an
    ablation row that tuning makes identical to OCSD is not generated. The E4 table and `E3all` now average the same
    seeds for every row (before, OCSD rows averaged two seeds and ablation rows one).
    Only the (split, method) folders whose settings changed are regenerated (`outputs/.gen_configs.json`).
