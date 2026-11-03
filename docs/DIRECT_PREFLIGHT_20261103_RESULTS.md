# CPU preflight 03/11/2026: lỗi bộ ghi log khi nạp CLIP

> Lịch sử lỗi. Lượt chạy lại `20261103T142406Z_013a5f0e` đã [PASS](DIRECT_PREFLIGHT_20261103_PASS.md); không cần chạy lại CPU theo hướng dẫn phục hồi bên dưới.

Run `20261103T140807Z_e0a5503f`, code `2076784bd6dd4fdb8631e2def8f77365a83ffdf1`, trạng thái **failed**. Chưa chạy bước probe UniFusion/UNet và chưa sinh ảnh. Không dùng lượt này làm bằng chứng chất lượng mô hình hoặc kết luận tương thích hoàn tất.

## Đối chiếu gói đã nhận

- File: `ocsd_direct_preflight_20261103T140807Z_e0a5503f.zip`.
- SHA-256 ZIP: `73a2b6b80802d5839aa3565bf94309a9f74752d11899c4c96c0e3f8b80e5f2e0`.
- ZIP gồm `preflight.json`, `preflight.log`, `setup.log`. Ba hash mã nguồn/requirements trong JSON khớp chính xác nội dung tại commit được ghi.
- Python 3.13.18; torch 2.11.0+cpu; torchvision 0.26.0+cpu; Transformers 5.17.0; HF Hub 1.33.0; timm 1.0.30; einops 0.8.2; NumPy 2.4.6; OmegaConf 2.3.0.
- Cài thư viện trả mã 0. Script dừng sau 89,05 giây; đây là thời gian tới lỗi, không phải benchmark tốc độ suy luận.

## Điểm dừng thực tế

Log cho thấy file InstanceDiffusion khoảng **13,6 GB** và file CLIP khoảng **1,71 GB** đã tải xong. Trước bước lỗi, luồng mã đã qua các lời gọi strict load của UNet, VAE, diffusion và global text encoder. Tuy nhiên JSON chưa ghi model provenance, và chưa có kết quả forward của probe.

Lỗi tại `CLIPTextModel.from_pretrained` → Transformers `log_state_dict_report` → `_style`:

```text
if sys.stdout is not None and sys.stdout.isatty():
AttributeError: 'Tee' object has no attribute 'isatty'
```

CLIP nguồn chứa cả vision và text. Khi chỉ nạp text, Transformers tạo báo cáo về các key vision/projection không dùng và kiểm tra khả năng hiển thị màu của stdout. Wrapper ghi log của preflight chỉ có `write`/`flush`, thiếu giao diện stream mà báo cáo này cần. Đây là lỗi wrapper, không phải bằng chứng checkpoint thiếu learned weights hay mô hình tạo ảnh lỗi.

## Sửa và kiểm thử

`Tee` chuyển tiếp các thuộc tính/phương thức stream như `isatty`, `fileno`, `encoding`, `errors` tới terminal gốc; `write` và `writelines` vẫn chép nội dung vào log. Không bỏ báo cáo nạp trọng số và không nới lỏng kiểm tra thiếu learned weights.

Hai kiểm thử mới tái hiện lỗi trước khi sửa:

1. Giao diện stream và việc chép đầy đủ text Unicode.
2. Nạp một checkpoint CLIP đầy đủ thu nhỏ, lưu cục bộ, thành CLIPTextModel **bên trong chính hàm preflight có chuyển hướng stdout/stderr**. Đường tạo loading report thật gây cùng lỗi `isatty` với bản cũ; bản sửa phải nạp đủ text weights và xuất ZIP trạng thái pass của fixture.

Checkpoint trong kiểm thử là trọng số tổng hợp nhỏ, không phải checkpoint pretrained 13,6 GB. Kết quả kiểm thử này xác nhận sửa lỗi ghi log, không thay thế preflight thật trên Colab. Adapter suy luận, trọng số, tham số và điều kiện GPU không thay đổi.

Sau sửa: 5 kiểm thử preflight và 15 kiểm thử tái lập đều đạt trên môi trường CPU phát triển (Python 3.12, Transformers 5.17.0). Cả 5 notebook hợp lệ và tái sinh nhất quán từ generator. Lỗi nạp CLIP đã được tái hiện trước sửa và hết lỗi sau sửa trong fixture cục bộ; preflight đầy đủ trên phiên Colab Python 3.13 vẫn chờ lượt chạy lại.

## Bước tiếp theo

Giữ phiên Colab CPU hiện tại nếu còn hoạt động, `RUN_MODE="direct_preflight"`, `RUN_ID=""`, rồi **Run all** để ô 2 cập nhật code và ô 3 dùng lại cache. Gửi ZIP mới, kể cả khi có lỗi khác. Nếu runtime đã mất, tải lại trọng số trong phiên CPU mới. Chưa chạy GPU, export screen hoặc yêu cầu người dùng chấm ảnh.
