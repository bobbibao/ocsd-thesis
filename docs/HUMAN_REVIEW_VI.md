# Review ảnh OCSD dành cho người mới

Tải [bộ review ZIP](reviews/ocsd-human-review-20261102.zip), giải nén, rồi mở **REVIEW.html** bằng Chrome/Edge/Firefox trên máy tính. Không cần Python, Colab, GPU, đăng nhập hoặc kết nối mạng. Nếu GitHub không xem trước được ZIP, chọn **Download raw file**. Trang chứa sẵn toàn bộ ảnh; không tải dữ liệu lên dịch vụ khác.

## Làm thế nào?

1. Đọc mô tả tiếng Việt, nhìn phác thảo, rồi so với ảnh A–D bên dưới. Bấm ảnh để phóng to. Nhãn không gắn cố định với một phương pháp.
2. Chọn điểm cho từng tiêu chí: **0 = sai rõ; 1 = đúng một phần; 2 = đúng đầy đủ; ? = không chắc; N/A = không áp dụng**. Để trống nếu chưa chấm. Đây là thang điểm mới, không phải thang 1–5 trong CSV cũ.
3. Ghi điều bạn thực sự thấy: “cần 4 ngựa và 1 chó; thấy 5 ngựa, thiếu chó”, hoặc “đủ 3 ghế nhưng ghế bên trái sai màu”. Không cần tính metric.
4. Bấm **Tải kết quả JSON**. Gửi file `ocsd_review_20261102T165132Z_e9ea4c56.json` trong chat. Không gửi lại ảnh, không sửa JSON bằng tay.

Có 12 cảnh, 48 kết quả nhưng chỉ **25 ảnh khác nhau**, tương ứng 200 lựa chọn nếu chấm đủ tám tiêu chí. Các ảnh có file giống hệt trong cùng cảnh được gộp; câu trả lời sẽ tự áp dụng cho các nhãn đó. Không coi những nhãn lặp là các đánh giá độc lập.

**Nếu ít thời gian:** bắt đầu với cảnh **6, 7 và 8** (xe đạp/tàu; nhà/thuyền/mèo; ngựa/chó). Chấm “đúng loại” và “đúng số lượng”, ghi số từng loại thực thấy; sau đó chấm vị trí và hình dáng nếu rõ. Các mục khác được phép để trống. Đây là nhóm ưu tiên chẩn đoán, không đại diện cho toàn bộ chất lượng mô hình. Gửi bản chấm dở cũng hữu ích.

## Phân biệt tiêu chí

| Tiêu chí | Cần nhìn gì? |
|---|---|
| Đúng loại | Có đủ loại chó, mèo, ghế… theo mô tả không? Chưa xét số lượng. |
| Đúng số lượng | Đếm riêng từng loại. Đủ tổng số nhưng sai loại vẫn chưa đạt. Thừa cũng là lỗi. |
| Đúng vị trí | Vị trí và kích thước tương đối có khớp phác thảo không? |
| Đúng hình dáng | Đường bao, tư thế, hướng nhìn có khớp nét vẽ không? Đúng vị trí chưa chắc đúng hình dáng. |
| Đúng quan hệ | Trái/phải, trên/dưới, chồng lấp có đúng không? Một vật thì chọn N/A. Không rõ thì chọn ?. |
| Đúng đặc điểm | Màu, chất liệu, kích thước được yêu cầu. Không có yêu cầu thì N/A. |
| Đúng bối cảnh | Phòng khách, rừng, sân ga… có đúng mô tả không? |
| Ảnh tự nhiên | Vật có méo, dính nhau, thừa bộ phận hoặc lộ viền ghép không? |

Không cần cố đoán hình khó nhận ra. Với chim/vịt: mô tả liệt kê riêng ba chim và một vịt; đừng đếm lại vịt vào ba chim. Nếu không phân biệt được, chọn ? và ghi chú. Không suy màu vật từ nét phác thảo đen.

## Lưu và tiếp tục

Trang thử tự lưu trên trình duyệt nhưng file cục bộ có thể không cho phép. **Luôn tải JSON trước khi đóng trang**. Lần sau mở lại trang, bấm **Nhập JSON** để tiếp tục. Nhập file thay thế trạng thái hiện tại sau xác nhận; tải bản cũ trước nếu muốn giữ. Chỉ nhập được đúng run và phiên bản thang điểm.

Kết quả chứa run ID, mã người chấm tùy chọn, điểm, ghi chú, nhãn ảnh và SHA-256. Giá trị chưa chấm luôn để trống; trang không tự điền đánh giá. Không cần tên thật. File JSON là định dạng trả lời chính thức; phần `rows` có 48 dòng tương thích đối chiếu nhãn, phần `ratings` giữ 25 nhóm để tiếp tục chấm.

Đây là review phát triển mô hình của người dùng đã biết một số kết quả, **không phải nghiên cứu người dùng mù độc lập**. Không dùng điểm này làm test cuối hoặc điều chỉnh ngưỡng đến khi thắng. Chúng ta dùng nó để phân biệt lỗi detector với lỗi sinh ảnh và quyết định thử nghiệm tiếp theo.

## Tạo lại bộ review (dành cho người phát triển)

```bash
python tools/build_human_review.py --bundle-root /path/to/merged-export --out /tmp/review
python tools/audit_instance_failures.py --bundle-root /path/to/merged-export --out /tmp/failures.csv
```

Công cụ kiểm tra hash tất cả đầu vào sử dụng, giữ ảnh PNG gốc, không đưa tên phương pháp/metric/khuyến nghị điểm vào trang. Bản ZIP đã tạo sẵn cho người review; họ không cần chạy các lệnh này.
