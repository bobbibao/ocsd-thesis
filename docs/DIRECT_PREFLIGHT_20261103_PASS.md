# CPU preflight đã PASS — 03/11/2026

Gói `ocsd_direct_preflight_20261103T142406Z_013a5f0e.zip` thuộc code `78e12886ff19b685ce60c3a65ab1f4273846f028`. Ba hash mã nguồn/requirements khớp commit; SHA-256 ZIP là `8134c8db3645bbadcacbfa3801f4cb8e4884ff6e3abcc07da508e79c760a96d0`. [JSON kết quả gốc](direct_preflight_20261103_pass.json) được giữ để đối chiếu và khóa trọng số cho thử nghiệm tiếp theo.

Đã xác minh trên Colab CPU, Python 3.13.18, torch 2.11.0+cpu và Transformers 5.17.0:

- Nạp checkpoint EMA InstanceDiffusion và đủ learned weights của CLIP text.
- Phrase features: `2 × 768`; đầu ra UniFusion: `1 × 184 × 768`.
- Cache conditioner trùng từng giá trị với tính lại không cache; 2 yêu cầu, 1 lần tính và 1 cache hit.
- Đổi scribble làm đặc trưng conditioner thay đổi: max absolute difference `29.372278213500977`. Đây là kiểm tra đường điều kiện hoạt động, **không phải điểm chất lượng ảnh**.
- Một forward UNet pretrained trên latent `1 × 4 × 8 × 8` cho kết quả hữu hạn.
- Thời gian script khoảng 123,32 giây, gồm nạp/kiểm tra, không phải thời gian sinh một ảnh.

Model revisions:

| Nguồn | Revision |
|---|---|
| InstanceDiffusion source | `722900ef03e3579fb693e65a1550333a4a162581` |
| `xudongw/InstanceDiffusion` | `fc74d76cde4a3dd0c3894e1764c8cf7299c77595` |
| `openai/clip-vit-large-patch14` | `32bd64288804d66eefd0ccbe215aa642df71cc41` |

Checkpoint SHA-256: `b4de3f7877a1860b52837afe3a767ca92a2ef5084056048b503af80027cde1d1`.

Các dòng `UNEXPECTED` trong báo cáo CLIP là vision tower/projection/logit scale không dùng khi chỉ nạp text tower; kiểm tra learned text weights đã đạt. Lỗi wrapper `Tee.isatty` trước đó đã hết. Cảnh báo deprecation của upstream không làm hỏng lượt kiểm tra này.

**Giới hạn:** chưa sinh ảnh, chưa xác minh FP16/full-resolution GPU, chưa đo số lượng/hình dáng/naturalness hoặc so sánh phương pháp. Không cần chạy lại CPU preflight. Bước tiếp theo là [canary 4 ảnh có giới hạn](DIRECT_CANARY.md), chưa mở rộng paper/full.
