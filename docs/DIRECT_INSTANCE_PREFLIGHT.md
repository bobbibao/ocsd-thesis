# Điều kiện hóa trực tiếp từng đối tượng: kiểm tra CPU trước khi dùng GPU

Ngày 03/11/2026. Trạng thái: đã có adapter và kiểm thử lõi trên CPU; **chưa xác minh trọng số thật, chưa đo cải thiện chất lượng ảnh**.

## Việc cần làm trên Colab

1. Mở lại [RUN_ALL.ipynb từ main](https://colab.research.google.com/github/bobbibao/ocsd-thesis/blob/main/notebooks/RUN_ALL.ipynb), dùng phiên runtime mới.
2. Chọn **Runtime → Change runtime type → Hardware accelerator: None (CPU)**. Nếu đang giữ GPU, đổi về CPU trước khi chạy.
3. Giữ `RUN_MODE="direct_preflight"`, `RUN_ID=""`, `BRANCH="main"` và đường dẫn Drive cũ. Chọn **Runtime → Run all**.
4. Chờ tải thư viện/trọng số công khai, tổng dung lượng có thể vài GB. CPU vẫn cần RAM, dung lượng đĩa và thời gian tải; chưa có thời gian đo trên Colab cho bước này.
5. Gửi **một file `ocsd_direct_preflight_….zip`** notebook tải về vào cuộc trò chuyện, dù kết quả là `pass`, `failed` hay `incomplete`. Không cần chấm ảnh, mở HTML, chạy export screen hoặc nhập GH_TOKEN.

Nếu trình duyệt không tự tải ZIP, lấy file theo đường dẫn in ở cuối ô 3 trên Drive:
`MyDrive/KLTN_OCSD/results/direct_preflight/attempt_…/`.
ZIP chỉ chứa JSON và log cài đặt/kiểm tra. Không chứa trọng số nhiều GB. Nếu thiếu RAM hoặc tiến trình bị hệ điều hành dừng, notebook giữ log vào ZIP `incomplete`; gửi file đó để xử lý, chưa chuyển sang GPU.

Các ô cài thí nghiệm cũ và A–F được bỏ qua trong chế độ này, kể cả khi kiểm tra thất bại. Kết quả ảnh cũ và `latest_run.json` được giữ nguyên. `TIER`, `BACKBONE`, `MAX_MINUTES`, `PUSH_RESULTS` chỉ dùng khi chọn `RUN_MODE="experiment"`; chế độ đó dành cho tái lập thí nghiệm cũ và vẫn cần GPU. **Không chạy lại foreground_screen/paper/full cho bước này.**

## Vì sao chuyển hướng

[Review đã nhận](FOREGROUND_20261103_RESULTS.md): A–D đủ số lượng/hình dáng; A/B được ưu tiên nhưng còn thiếu tự nhiên. Kiểm tra ảnh cho thấy nền lấp bằng Telea bị nhòe và chi tiết mảnh của xe đạp suy giảm qua xử lý latent. Tiếp tục chỉnh mask trên cùng một cảnh không đủ để chứng minh phương pháp tốt hơn.

Giả thuyết mới: sinh cả cảnh với điều kiện theo từng đối tượng có thể giảm dấu vết ghép ảnh, đồng thời giữ liên kết mô tả–vị trí–nét vẽ. Đây là **giả thuyết cần kiểm chứng**. InstanceDiffusion là phương pháp có sẵn của tác giả khác, dùng làm đối chứng mạnh và nền tảng khảo sát; tích hợp nó chưa phải đóng góp kiến trúc OCSD mới.

Nguồn đã kiểm tra:

- [InstanceDiffusion, CVPR 2024](https://arxiv.org/abs/2402.03290): UniFusion nhận điều kiện theo instance, ScaleU và Multi-instance Sampler hỗ trợ liên kết thuộc tính/vị trí.
- [Mã chính thức, commit cố định](https://github.com/frank-xwang/InstanceDiffusion/tree/722900ef03e3579fb693e65a1550333a4a162581). Chính README gọi đây là bản tái hiện của tác giả thứ nhất và lưu ý kết quả có thể khác paper. Không lấy số liệu paper làm kết quả tích hợp này.
- [Checkpoint công khai](https://huggingface.co/xudongw/InstanceDiffusion), `instancediffusion_sd15.pth`; CLIP từ `openai/clip-vit-large-patch14`. Mỗi lượt ghi revision SHA và hash checkpoint đã tải.
- [Bản port diffusers đã kiểm tra](https://github.com/gokyeongryeol/diffusers/tree/d377f574007a2cb0dc31d768bf61a9d56cbfc7ad): pipeline công khai chỉ truyền phrase/box; `INSTDIFFTextBoundingboxProjection.forward` thay scribble bằng null embedding. Chỉ đặt tên “sketch” cho nhánh này sẽ không làm nó dùng nét vẽ. Adapter hiện tại dùng lõi chính thức có đường scribble hoạt động; không thay thư viện diffusers của OCSD bằng fork.

## Hợp đồng đầu vào và cách tích hợp

`ocsd/direct_instances.py` nhận scene đã có **phrase, box và raster nét vẽ riêng cho từng instance**. Hai xe đạp cùng lớp vẫn là hai instance. Tọa độ được chuẩn hóa theo đúng chiều rộng/cao, hỗ trợ 1–30 instance và từ chối sketch trống hoặc box ngoài ảnh.

Mỗi raster được lấy 20 điểm nằm trên nét mực bằng phủ điểm xa nhất, sắp thứ tự theo khoảng cách tới gốc tọa độ; lặp điểm khi nét quá ngắn. Đây là biểu diễn mất thông tin, **không phải giữ toàn bộ đường bao**. Bản này không tự phân đoạn một phác thảo thô thành instance. Nó dùng chú giải phrase/box có sẵn, không dùng mask/polygon chuẩn của benchmark; tâm điểm lấy từ box. Vì vậy chưa được gọi là hệ thống end-to-end chỉ nhận sketch thô và caption.

Lõi UNet, UniFusion, ScaleU, PLMS/MIS và VAE được lấy từ commit chính thức đã khóa. Mọi trọng số học được phải khớp tên/kích thước; không bỏ qua key lỗi hoặc dùng trọng số khởi tạo thay thế. Bộ nạp đổi prefix CLIP cũ sang Transformers 5 và chỉ bỏ buffer position IDs không học sau khi kiểm tra đúng dãy chuẩn. Đặc trưng phrase dùng `pooler_output` trước projection, như upstream, bỏ lượt forward CLIP vision không dùng. ConvNeXt khởi tạo không tải thêm ImageNet vì strict checkpoint load cung cấp toàn bộ trọng số của nó.

Lưu ý tương thích quan trọng: Transformers 5 đổi `CLIPModel.text_model_output.pooler_output` thành đầu ra sau projection; upstream huấn luyện với ý nghĩa trước projection của Transformers 4. Adapter dùng trực tiếp `CLIPTextModel.pooler_output`; kiểm thử bắt đặc trưng ở nhánh text trước projection để tránh lỗi âm thầm này.

Các tối ưu chạy suy luận:

- Lưu đặc trưng UniFusion trong phạm vi **một ảnh**, cho nhánh toàn cảnh, từng đối tượng và unconditional. Trọng số, modality và precision phải cố định trong phạm vi đó. Thay tensor/in-place edit làm mất hiệu lực cache; không dùng cache cho huấn luyện.
- Tái sử dụng tensor null thay vì cấp phát lại mỗi CFG call. Cache được giải phóng sau ảnh/khi lỗi.
- Khôi phục input convolution SD một lần khi điều kiện hết hiệu lực; cuối ảnh khôi phục lại trọng số learned, kể cả khi sampler lỗi. Điều này tránh ảnh sau phụ thuộc trạng thái còn lại của ảnh trước.
- Ghi số UNet forward thực tế và so với kế hoạch. Với CFG tách hai nhánh, 50 bước và MIS 0.36, 3 instance cần **216** lượt, 5 instance cần **292** lượt; không dùng MIS là **102**. Công thức tính cả bước improved Euler đầu mỗi trajectory. Đây là số lượt forward, **không phải FLOPs/thời gian tương đương** với DDIM/ControlNet.

`DirectModel.generate` có sẵn cho tích hợp sau khi xác minh checkpoint. Chưa đưa nhánh này vào danh sách phương pháp của Runner hay tự động sinh ảnh trong RUN_ALL. Không bật SDXL refiner hoặc tìm nhiều ảnh rồi chọn ảnh đẹp.

## Bằng chứng hiện có và giới hạn

Kiểm thử bằng lõi chính thức thu nhỏ, trọng số ngẫu nhiên có kích hoạt đường conditioning, CPU float32:

- Cùng đầu vào/latent, bật và tắt cache cho kết quả **bằng nhau từng giá trị**. 34 lượt UniFusion giảm còn 4; số UNet call giữ nguyên 34. Đây là giảm tính lặp conditioner, chưa phải tăng tốc end-to-end đã đo trên mô hình đầy đủ.
- Đổi scribble làm latent thay đổi; lặp lại ảnh không bị trạng thái convolution/scales từ lần trước ảnh hưởng.
- Khi sampler lỗi, convolution, null input, forward cache và scale được khôi phục.
- Text-only CLIP cho pooler giống nhánh text trong CLIPModel có cả vision; kiểm tra ánh xạ checkpoint cũ và từ chối thiếu learned weights.
- Các kiểm tra bootstrap xác nhận lỗi tải/cài thư viện vẫn xuất ZIP, và chế độ CPU không thực thi bất kỳ stage cũ A–F nào.

Lệnh tái lập kiểm thử lõi, với checkout chính thức đúng commit:

```bash
OCSD_INSTDIFF_SOURCE=/path/to/InstanceDiffusion python -m unittest discover -s tests -p 'test_direct*.py' -v
```

Ở môi trường phát triển hiện tại, tải Hugging Face bị proxy từ chối `403 Forbidden`, nên lần preflight thực **failed trước khi tải checkpoint**; đường xuất log lỗi đã hoạt động. Không có GPU ở môi trường này. Không suy diễn kiểm thử ngẫu nhiên thành chất lượng ảnh hoặc tương thích FP16 trên GPU.

Preflight trên Colab kiểm tra: nạp strict checkpoint, phrase features, UniFusion với scribble thật, cache parity, đổi scribble có tác động, và một forward UNet pretrained trên latent 8×8. Không decode một scene, không đo thẩm mỹ, không xác minh full-resolution GPU memory/FP16. **PASS chỉ là điều kiện kỹ thuật để thiết kế thử nghiệm chất lượng nhỏ tiếp theo.**

## Bước nghiên cứu sau khi kiểm tra đạt

Chốt protocol trước lượt GPU: cảnh phát triển mới chưa dùng để chỉnh tham số, seed ghép cặp, cùng caption/instance annotations, đối chứng direct box-only và direct box+scribble, ControlNet và nhánh soft-matte đã được người dùng ưu tiên. Ghi chi phí đầy đủ; không gọi số lượt UNet bằng nhau là cùng compute. Dừng nếu mất đối tượng, sai đếm, hỏng hình dáng hoặc vượt ngân sách.

Nếu cần người dùng chấm, xuất **PNG side-by-side** và ảnh gốc có nhãn ẩn, kèm mô tả yêu cầu từng cảnh và form count/shape/layout/relations/naturalness. Không yêu cầu chấm lại bộ A–D đã xong. Chỉ sau khi có tín hiệu giữ được cấu trúc và tăng tự nhiên mới xem xét nhiều seed/cảnh và tập đánh giá chưa xem; chưa có cơ sở cho tuyên bố SOTA hoặc paper xuất sắc.

Giấy phép upstream: đa số mã Apache-2.0; CLIP, Stable Diffusion và GLIGEN có điều khoản riêng theo [README](https://github.com/frank-xwang/InstanceDiffusion/tree/722900ef03e3579fb693e65a1550333a4a162581#license-and-acknowledgment). Repo OCSD không sao chép checkpoint hoặc mã upstream; tải checkout đã khóa khi cần.
