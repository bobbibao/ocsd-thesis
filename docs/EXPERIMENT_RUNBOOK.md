# Chạy và xác minh phiên bản tối ưu trên Colab

## Mở từ GitHub như trước

Notebook chính: [RUN_ALL.ipynb trên Colab](https://colab.research.google.com/github/bobbibao/ocsd-thesis/blob/main/notebooks/RUN_ALL.ipynb).
Không cần chép ZIP lên Drive. Giữ REPO_URL của bobbibao/ocsd-thesis và BRANCH="main".

1. Mở link Colab, chọn GPU L4/A100 và giữ DRIVE_ROOT đang dùng.
2. Mặc định `TIER="repair_screen"`, `BACKBONE="sd15"`, `RUN_ID=""`, `MAX_MINUTES=20`.
   Đọc [QUALITY_SCREEN.md](QUALITY_SCREEN.md) trước khi tiêu GPU: đây là giả thuyết chưa có kết quả chất lượng.
3. Run all chỉ chạy screening 48 ảnh và đánh giá, không tự chạy pilot/paper/full.
4. `PUSH_RESULTS=True` và Colab secret GH_TOKEN gửi log/kết quả lên colab-results như luồng cũ.
   Thiếu token thì kết quả vẫn ở Drive; không yêu cầu đưa token vào code.
5. Đọc summary.md, screen_means.csv, screen_paired.csv và ảnh. `do_not_scale_up` nghĩa là chưa nên
   chạy lớn. `review_for_heldout_validation` chỉ là tín hiệu cần xem xét, không phải kết luận thắng.

RUN_ID trống lấy code mới bằng fast-forward khi checkout sạch; checkout có chỉnh sửa sẽ dừng để
bảo toàn dữ liệu. Dùng runtime mới để lấy main sạch. RUN_ID cụ thể giữ code checkout khi resume;
nếu main đã đổi từ lần chạy cũ, dùng BRANCH bằng tên branch/tag giữ đúng commit cũ trên runtime mới.
Không dùng hai kernel đồng thời ghi cùng RUN_ID. Tiếp tục hỗ trợ REPO_URL="" và DRIVE_ROOT/code
khi chủ động dùng code local, nhưng đó không phải luồng mặc định.

## Nơi lưu và resume

- `results/runs/<run_id>/run_manifest.json`: code, notebook, thư viện, model SHA, dữ liệu, thời gian,
  trạng thái từng giai đoạn, tiến độ job và checksum artifact.
- `outputs/runs/<run_id>/<split>/<method>/`: ảnh và JSON cùng signature/checksum, cấu hình, chi phí.
- `results/runs/<run_id>/`: `summary.md`, `summary.json`, `tables/`, `figures/`, `tuning/`, `logs/`.
- `results/latest_run.json`: con trỏ tới kết quả mới nhất. Các đường dẫn kết quả cũ được giữ nguyên;
  dùng con trỏ này khi chuyển các công cụ ngoài sang định dạng mới.
- `data/benchmarks_v2/`: benchmark mới. `data/benchmarks/` cũ không bị ghi đè. Raw QuickDraw/COCO và
  Hugging Face cache được tái sử dụng khi còn hợp lệ. Cache mô hình nằm trên đĩa Colab, không trên Drive.

Ghi lại RUN_ID. Sau ngắt kết nối, dùng lại đúng code/cấu hình/thư viện và RUN_ID, rồi Run all.
RUN_ID trống tự tìm lượt tương thích gần nhất. Nếu code, dữ liệu hoặc thư viện thay đổi, RUN_ID tường
minh bị từ chối; bắt đầu lượt mới. Mỗi lượt có model revision riêng; resume dùng lại SHA đã khóa.
Fingerprint notebook bỏ qua output Colab tự lưu và giá trị ô cấu hình vận hành; cấu hình khoa học
vẫn được đối chiếu qua `contract.experiment`. Đổi RUN_ID/MAX_MINUTES không tự làm hỏng resume.
Ảnh/JSON hỏng hoặc sai signature được lưu sang `_stale` và tạo lại, không xóa cả cache.

Stage thất bại dừng notebook; xem `logs/LATEST_ERROR.txt` và manifest rồi sửa nguyên nhân và resume.
Hết `MAX_MINUTES` cũng giữ ảnh đã hoàn tất. Giới hạn được kiểm tra giữa các scene, nên một scene có
nhiều phương pháp có thể vượt giới hạn. Tuning dùng giới hạn theo grid, không phải tổng cả notebook.

`00_push_logs.ipynb` dùng RUN_ID cụ thể hoặc `latest_run.json` khi để trống; chỉ gửi khi đặt
`PUSH_RESULTS=True` và có Colab secret `GH_TOKEN`. Manifest đi cùng kết quả được gửi. Gói bàn giao
không tự gửi dữ liệu lên GitHub.

## Kiểm tra trước khi dùng số liệu

- Manifest phải thuộc RUN_ID đang xét; `status == "complete"`; mọi job có `todo == 0`.
- Tất cả bảng/hình lấy từ cùng `results/runs/<run_id>`; kiểm tra SHA-256 với `artifacts` trong manifest.
- Dùng `contract.code`, `contract.notebook_sha256`, `git.commit/status` và `model_revisions` để xác định
  phiên bản thực thi. Không dùng timestamp file như bằng chứng duy nhất.
- Ghi rõ bộ dữ liệu `qd-v2`, số scene/seed thực sự, seed tuning, các job bỏ qua và backbone.
- Mọi kết luận từ lượt mới là exploratory trừ khi có protocol mới được chốt trước khi sinh ảnh.
- So sánh ngân sách bằng `compute`, `prep_compute`, `prep_times`, `scene_time` và `tries`. N lần sinh
  không tự động đồng nghĩa cùng FLOPs/thời gian; GLIGEN dùng U-Net grounding riêng.

## Ngân sách thử đầu tiên

Mặc định chọn `repair_screen`; chưa có thời gian GPU đo thực cho biến thể mới. Giới hạn 20 phút chỉ áp dụng
cho stage sinh ảnh giữa các scene, không phải trần compute units. Nếu sau screening chủ động chạy
`pilot`, kết quả lịch sử trên A100 80 GB tại nhánh `colab-results` ước tính **1,3 giờ GPU sinh ảnh**
cho pilot, chưa gồm khoảng 10–15% đánh giá và tải dữ liệu/mô hình. Dành khoảng **1,5–2 giờ trên A100**
cho lượt thử, đây chỉ là ước lượng từ code cũ; L4 và bộ dữ liệu mới phải đo lại. Không suy ra chi phí tiền
hoặc compute units cố định từ CPU smoke tests. Dùng `budget_estimate.json` của lượt mới sau khi chạy.

## Xác minh đã làm / còn thiếu

Đã chạy trên CPU: prompt, vision wrappers nhỏ, COCO giả, 15 hồi quy provenance/cache/FID/tuning/manifest,
M3 v-prediction và thay rank LoRA, smoke đại diện 4 phương pháp × 2 seed, test OCSD-v2 52 ảnh với
resume và báo cáo. Fixture nét vẽ tổng hợp dùng định dạng QuickDraw; không phải dữ liệu benchmark thật.
Các bài kiểm thử mới dùng mock FID để kiểm tra cache/chọn mẫu, không xác minh Inception/FID thật.

Còn cần Colab: tải model SHA thật, pretrained inference/GLIGEN, fp16/CUDA và OOM, PiDiNet thật,
FID/KID/LPIPS với trọng số thật, mount Drive và resume qua hai phiên. Toàn bộ `test_tune.py`,
`test_v2_ablation.py` và smoke 110 ảnh chưa được chạy đầy đủ trong lượt sửa này; các đường liên quan
được kiểm bằng test chuyên biệt và test OCSD-v2. Không chạy thực nghiệm dài thay cho người dùng.

CPU: cài torch/torchvision phù hợp, `pip install -r requirements-dev.txt`, rồi chạy
`python tests/test_reproducibility.py`, `python tests/test_training_regression.py` và các test README.
`CLIP_BPE` trỏ tới file BPE trong package open_clip; test tải QuickDraw cần truy cập storage.googleapis.com.
Thêm `python tests/test_screening.py`: 9 kiểm thử mới, gồm sửa vùng bằng model nhỏ và Runner/resume.
Sinh lại notebook bằng `python tools/make_notebooks.py`.

Missing-only M2 and CPU ZIP recovery: see [SPARSE_REPAIR_SCREEN.md](SPARSE_REPAIR_SCREEN.md). Use a fresh RUN_ID for this code version.

Current default and independent review: [SEMANTIC_SCREEN.md](SEMANTIC_SCREEN.md). GLIGEN candidates are shared but charged per method; generation count exceeds final output count.

2026-10-29: semantic screen failed the development comparison; default restored to repair_screen. Do not rerun completed screens or scale up. See SEMANTIC_20261029_RESULTS.md.
