# Canary GPU: 4 ảnh trước khi mở rộng

CPU preflight `20261103T142406Z_013a5f0e` đã [PASS với trọng số thật](DIRECT_PREFLIGHT_20261103_PASS.md). Chưa có ảnh để kết luận mô hình tốt hơn. Lượt này kiểm tra đường sinh ảnh 512×512 trên GPU và tác động của nét phác thảo trong một phép thử nhỏ; không phải so sánh SOTA hay kết quả paper.

## Chạy như trước bằng RUN_ALL

1. Mở lại [RUN_ALL.ipynb từ main](https://colab.research.google.com/github/bobbibao/ocsd-thesis/blob/main/notebooks/RUN_ALL.ipynb) để nhận ô canary mới.
2. Chọn **GPU L4 hoặc A100**, tối thiểu 20 GiB VRAM. Giữ `RUN_MODE="direct_canary"`, `RUN_ID=""`, `BRANCH="main"`, đường dẫn Drive cũ. Không chạy lại CPU preflight.
3. **Run all**. Chỉ chạy canary; các stage tuning/sinh ảnh/metrics/upload cũ tự bỏ qua. Không cần GH_TOKEN hay export screen.
4. Gửi ZIP **`ocsd_direct_canary_….zip`** tải về, kể cả khi lỗi hoặc chưa đủ ảnh. File cũng nằm trong `MyDrive/KLTN_OCSD/results/direct_canary/attempt_…/`.
5. Nếu trạng thái `complete`, xem hai PNG `run/SO_SANH_seed0.png`, `run/SO_SANH_seed1.png`. Điền `run/REVIEW_TEMPLATE.txt` hoặc trả lời theo mẫu đó trong chat. Hướng dẫn cụ thể ở `run/HUONG_DAN.txt`.

Tiến trình canary bị dừng sau **600 giây**, gồm tải/nạp trọng số và sinh ảnh, **chưa gồm cài thư viện và thời gian GPU nhàn rỗi**. Kiểm tra thời hạn diễn ra trước từng UNet call; dành 30 giây cuối cho đóng gói. Notebook có timeout bên ngoài để dừng cả trường hợp kẹt khi tải/nạp. Không có cam kết rằng 4 ảnh sẽ xong trong ngân sách này. Khi thiếu ảnh, gửi log để đánh giá trước khi chạy tiếp; không tự tăng giới hạn.

Tải mới khoảng **13,6 GB checkpoint + 1,71 GB CLIP**, chưa gồm cache/tệp tạm. Đổi từ CPU sang GPU có thể làm mất cache cục bộ; Drive giữ kết quả đã có. Tải ZIP xong thì ngắt runtime GPU khi không dùng.

## Protocol cố định trước khi xem ảnh

Một cảnh phát triển mới: **xe đạp vàng bên trái, xe đạp xanh dương bên phải, một ghế băng gỗ phía sau trong công viên**. Hai seed `0, 1`; mỗi seed có hai nhánh:

| Nhánh | Điều kiện | Sampling |
|---|---|---|
| `direct_box` | Caption, phrase/box/tâm riêng cho từng instance | PLMS 50 bước, MIS 0.36, alpha 0.8, CFG 7.5 |
| `direct_scribble` | Như trên, thêm 20 điểm mực cho từng instance | Giống nhánh box |

Cùng caption, negative prompt, box, trọng số và noise ban đầu trong mỗi cặp seed. Model/source revisions và checksum được khóa từ CPU preflight đã đạt; không tự lấy model revision mới. Không SDXL refiner, không tìm nhiều ứng viên rồi chọn ảnh đẹp, không đổi prompt/seed theo kết quả. UNet dùng autocast FP16, VAE decode FP32 cho cả hai nhánh để tránh tràn FP16; độ ổn định trên GPU vẫn cần lượt canary này xác minh. Dừng ngay khi một UNet output không hữu hạn, OOM hoặc lỗi khác; không tự thử cấu hình nhẹ hơn.

Mỗi ảnh cần **216** UNet sample calls, cả bốn ảnh là **864**; công thức gồm CFG và bước improved Euler đầu mỗi trajectory. Đây không phải FLOPs hoặc ngân sách ngang ControlNet/OCSD. Chi phí các lượt lỗi được giữ trong lịch sử attempt; một call OOM có thể chỉ chạy một phần, và công việc đang chạy khi bị hệ điều hành kill được ghi là không xác định. Thời gian, peak CUDA memory và số call từng ảnh nằm trong JSON đi cùng ảnh.

Chỉ tái sử dụng ảnh khi contract code/thư viện/GPU/protocol giống nhau và checksum khớp. Run ID được suy ra từ contract, không dùng RUN_ID của thí nghiệm cũ. Kết quả đầy đủ đã có được đóng gói lại mà không nạp trọng số. Nếu đổi môi trường/code, bắt đầu thư mục riêng. Ảnh hoàn tất bị thay đổi làm dừng để kiểm tra; không tự ghi đè. Ảnh viết dở chưa có completion record sẽ được sinh lại khi người dùng chủ động chạy tiếp.

## Đầu vào và review

[Scene và nguồn nét vẽ](../assets/direct_canary_v1/README.md) đã khóa trước khi sinh ảnh: các bản QuickDraw nhận dạng được đầu tiên từ dòng 3000 trở đi, hai nét xe đạp khác nhau và một nét ghế. Layout/caption được thiết kế cho ca phát triển này. Không dùng mask chuẩn, chỉ dùng phrase/box có sẵn và các điểm mực. Đây chưa phải hệ thống tự tách instance từ sketch thô; không tuyên bố dữ liệu chưa từng nằm trong training của model.

PNG so sánh hiển thị phác thảo và hai ảnh A/B nguyên vẹn ở độ phân giải gốc. A/B đổi thứ tự theo seed; khóa đối chiếu nằm riêng trong gói phục vụ phân tích. Không cần mở HTML. Với mỗi ảnh, chấm:

- **Số lượng:** đúng 2 xe đạp + 1 ghế; thừa cũng là lỗi.
- **Hình dáng:** bánh, khung xe, tư thế và đường bao so với phác thảo.
- **Vị trí, quan hệ, màu sắc:** vàng trái, xanh phải, ghế phía trên/phía sau; ghi “không rõ” nếu không xác định được chiều sâu.
- **Tự nhiên:** 0 = méo/ghép/dính rõ; 1 = còn giả nhưng nhận ra được; 2 = tự nhiên, hợp ánh sáng/phối cảnh; hoặc “không rõ”.

Không suy ra cải thiện tổng thể từ bốn ảnh. Nếu hỏng count/shape hoặc không có tín hiệu tăng tự nhiên, dừng mở rộng và phân tích lỗi. Nếu đáng khảo sát tiếp, chốt một protocol nhiều cảnh/seed có cả ControlNet và OCSD soft-matte trước khi chạy. Lượt này **chưa có các đối chứng đó**, nên không thể kết luận thắng phương pháp cũ hoặc phương pháp khác.

## Kiểm tra tại môi trường phát triển

Kiểm thử CPU dùng pixel tổng hợp để kiểm tra orchestration: bốn job cố định, cache/resume, dữ liệu hỏng, đóng gói khi ngắt và checksum; các vùng ảnh trong PNG so sánh trùng ảnh gốc từng pixel. Lõi chính thức thu nhỏ kiểm tra timeout/nonfinite dừng sớm và hoàn nguyên state. Kiểm tra code/fixture không thay thế đo GPU thật hay chấm chất lượng ảnh.
