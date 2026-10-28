# Kiểm tra chất lượng trước khi tăng ngân sách — 2026-10-29

## Điều đã biết

Các sửa đổi cache, manifest và M3 không chứng minh ảnh tốt hơn. Bảng E3 QuickDraw lịch sử
cho OPR OCSD 74,0%, GLIGEN 82,2%, Collage 84,2%; OCSD-v2 có KID tốt hơn OCSD nhưng mIoU
thấp hơn. Đây là lý do thử một hướng khác, không phải bằng chứng cho hướng mới. Không thể
hứa chất lượng vượt trội hay điểm paper trước khi có kiểm chứng độc lập.

## Giả thuyết mới: grounded_repair

1. GLIGEN sinh cảnh theo hộp/văn bản, cùng seed và tham số với baseline GLIGEN.
2. Grounding DINO kiểm tra đối tượng, đếm và quan hệ. Nếu đạt thì trả nguyên ảnh, không chạy M2/M3.
3. Nếu thiếu đối tượng, lấy bộ M2 của scene (có cache, theo seed). Đây là bộ đầy đủ, chưa tối ưu
   thành chỉ sinh những crop thiếu. Nếu chỉ thừa đối tượng thì không cần M2.
4. Dùng SD 1.5 + ControlNet sửa vùng lỗi trong ảnh GLIGEN SD 1.4. Đối tượng đã khớp được bảo vệ
   bằng mask và hộp phát hiện; pixel ngoài vùng sửa được giữ nguyên. Không huấn luyện M3,
   không dùng energy guidance hay region attention trong biến thể này.
5. Chỉ nhận ảnh sửa nếu verifier cải thiện, không mất đối tượng đã khớp, không tăng lỗi đếm,
   không giảm mIoU/RA, và CLIP toàn ảnh giảm không quá 0,5 điểm so với ảnh ban đầu.
   Nếu bị từ chối, giữ ảnh trước đó và dừng thử lại. Screen cho phép tối đa một lượt sửa.

Verifier có thể nhận sai. Điều kiện trên chỉ đảm bảo quy tắc chấp nhận trên verifier,
không đảm bảo OWLv2 hoặc người xem đánh giá ảnh tốt hơn. CLIP được dùng bên trong phương pháp,
nên CLIP báo cáo không phải bằng chứng độc lập. Hai backbone khác nhau cũng có thể gây lệch phong cách.
Đây là kết hợp pipeline thực nghiệm; chưa chứng minh tính mới so với nghiên cứu liên quan.

## Screen trong RUN_ALL.ipynb

Mặc định: `TIER="screen"`, `BACKBONE="sd15"`, `MAX_MINUTES=20`.
Giữ luồng GitHub → Colab → Drive → nhánh colab-results, không cần chép ZIP.

- 12 scene QuickDraw: mỗi ô số đối tượng × độ phức tạp một scene; seed 0.
- 7 phương pháp, 84 ảnh đầu ra: ControlNet, GLIGEN, GLIGEN best-of-2, Collage, OCSD-v2,
  grounded_repair không M2, grounded_repair có M2.
- Giữ 512 px, DDIM 30 bước, M2 K=2 và 20 bước; không giảm độ phân giải để tạo lợi thế giả.
- Không COCO, M3, FID/KID, evaluator phụ hay tuning grid. Không tự chạy pilot/paper/full tiếp theo.
- 84 là số ảnh đầu ra, không phải số lần gọi diffusion: ứng viên M2 và best-of-2 tốn thêm compute.
  MAX_MINUTES chỉ kiểm tra giữa scene trong giai đoạn sinh ảnh; không phải trần tổng thời gian
  hoặc compute units. Thời gian pretrained trên A100/L4 của biến thể mới **chưa đo**.

Đọc `tables/screen_means.csv`, `screen_paired.csv`, `summary.md` và ảnh ở outputs.
`standalone_time_s` cộng thời gian scene và chi phí M2/M3 từ metadata để tránh che chi phí chuẩn bị;
cache chia sẻ làm thời gian thực của cả lượt khác tổng thời gian standalone. Counter `compute`
của grounded_repair đã gồm M2 thực sự chạy (`compute_includes_m2=True`), không cộng M2 lần nữa.
Best-of-2 không được gọi là cùng ngân sách: so sánh cả chi phí thực đo.

Nếu trung bình OPR không tăng so với cả GLIGEN, GLIGEN best-of-2 và Collage, hoặc mIoU/đếm
suy giảm so với một trong ba, báo cáo ghi `do_not_scale_up`. Nếu đạt điều kiện mô tả này,
chỉ ghi `review_for_heldout_validation`, vẫn `research_claim="unproven"`. Kiểm tra CI ghép cặp,
ảnh ẩn tên phương pháp và chi phí trước khi quyết định tiếp. Không dùng 12 scene/1 seed để kết luận SOTA.

Các scene screening phải ở phía phát triển/tuning. Luồng paper/full đóng băng những scene đã có
vào tuning_split trước khi dựng tập đánh giá mới. Không chỉnh tham số dựa trên tập đánh giá cuối.
Biến thể mới chỉ được đăng ký trong screen; chưa tự thêm vào bảng paper với tư cách phương pháp đã thắng.

## Xác minh và giới hạn

`python tests/test_screening.py`: 9 kiểm thử offline về chấp nhận/từ chối, M2 gọi lười,
đối chứng không M2, mask bảo vệ, kế hoạch 84 ảnh và báo cáo thiếu dữ liệu.
Một test chạy diffusion nhỏ thật để kiểm tra pixel ngoài vùng sửa không thay đổi.
GLIGEN pretrained được mock ở các test điều phối; không có số liệu GPU mới.
Chỉ cân nhắc lượt lớn khi có tín hiệu độc lập đủ tốt, cùng protocol held-out và ngân sách rõ ràng.
