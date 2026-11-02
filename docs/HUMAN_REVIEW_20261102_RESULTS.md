# Review người dùng: cải thiện cục bộ có thật, metric chưa phản ánh đầy đủ chất lượng

Run `20261102T165132Z_e9ea4c56`, review xuất lúc `2026-11-02T17:53:09.092Z`. Đây là một người dùng chấm phục vụ phát triển, không phải nghiên cứu nhiều người hoặc test mù độc lập. Người chấm đã được trao đổi một số kết quả trước đó. Không tính p-value hoặc suy luận SOTA từ review này.

## Kiểm tra và phạm vi dữ liệu

Đã đối chiếu đúng run/schema, toàn bộ 48 dòng nhãn với khóa A–D, hash ảnh gốc và 25 nhóm ảnh trùng. Phần `rows` khớp `ratings`; không có nhãn bị gán nhầm. File gốc được giữ tại [source.json](reviews/20261102_user_review/source.json), SHA-256 `bb148244d4b833fba44efbc981ef6369749a100e76c90aeacb4ab58eff34adac`.

Có phản hồi dưới dạng điểm hoặc nhận xét cho cả 12 cảnh: **88/200 mục có điểm số**, trên 11/25 nhóm ảnh khác nhau; 12 nhóm có ghi chú. Mỗi phương pháp có điểm số cho cùng 7/12 cảnh. Nhận xét ở các cảnh khác được giữ như bằng chứng định tính, không tự đổi thành điểm. 48 dòng sau khi giải mã chỉ là cách nối với phương pháp; không phải 48 ý kiến độc lập.

- [Điểm và ghi chú theo phương pháp](reviews/20261102_user_review/ratings_by_method.csv)
- [Thống kê điểm số và số mục chưa chấm](reviews/20261102_user_review/numeric_summary.csv)
- [Thông tin kiểm tra](reviews/20261102_user_review/summary.json)

Thang điểm 0/1/2 được giữ nguyên. Người chấm có cho điểm quan hệ ở cảnh một đối tượng thay vì N/A; thống kê thô không âm thầm sửa những điểm này. Điểm và lời bình cũng không bị ép cho khớp nhau: ví dụ người chấm nói ảnh ngựa B tự nhiên hơn nhưng vẫn cho realism=1 ở cả hai nhóm. Không suy ra chênh lệch điểm từ nhận xét đó.

## Kết quả theo từng cảnh

| Cảnh | Phản hồi của người dùng | Ý nghĩa đối với tối ưu |
|---|---|---|
| 1 — máy bay | Cả bốn bị chấm 0; nhận xét nhìn như mái che, không phải máy bay được yêu cầu. | `joint_layout_success=1` của detector không chứng minh ảnh đúng. Kiểm tra lại ảnh gốc cho thấy mái che chiếm phần lớn và chỉ có một dạng giống bộ phận máy bay; điều này không đáp ứng yêu cầu sinh cả đối tượng. |
| 2 — ngựa | Đúng nội dung/vị trí/hình dạng; ảnh B được nhận xét tự nhiên hơn nhưng điểm realism vẫn bằng nhau. B là GLIGEN best-of-five. | Giữ kết quả tốt; không phóng đại khác biệt chủ quan thành thắng về metric. |
| 3 — cây | Thiếu đồng cỏ xanh. | Đúng box/cây không đủ đúng bối cảnh. |
| 4 — ba ghế | Đủ và đúng loại nhưng sai màu; A còn sai màu ghế thứ ba, B/C/D bị nhận xét sai tất cả màu. A là GLIGEN best-of-five. | Cần gắn thuộc tính đúng từng instance; chỉ tăng presence không giải quyết lỗi. |
| 5 — voi/xe tải | Đủ và đúng đối tượng nhưng voi quay sai hướng so với sketch. | Cần đo tư thế/hướng/hình dáng riêng, không gọi box IoU là shape accuracy. |
| 6 — xe đạp/tàu | Một xe đạp được yêu cầu nhưng sinh nhiều xe, sai màu; tàu đúng vị trí/màu nhưng chưa tự nhiên. | Xác nhận lỗi sinh thừa thật ở trường hợp router bỏ qua vì đã tìm thấy mọi instance cần có. |
| 7 — nhà/thuyền/mèo | A/B/CD đều presence=2, count=2. B và CD bằng nhau cả tám tiêu chí; A thấp hơn layout/shape/relations. B là focus; CD là sparse/global. | Không có lợi ích human-rated của focus so với hai nhánh sửa đối chứng tại cảnh này. Chênh lệch OPR detector không được xác nhận tương ứng. |
| 8 — ngựa/chó | C có chó nhưng chưa tự nhiên; C presence/count=2/2, ABD=0/1. C là focus. | Xác nhận lợi ích sửa sai lớp của focus. Shape, attributes và realism vẫn chỉ 1; đây không phải ảnh hoàn toàn đúng. |
| 9 — ghế/ô/mèo | Loại/số lượng/vị trí=2, shape/relations/attributes/background=1, realism=0. | Một cảnh đạt joint box endpoint vẫn thất bại về tự nhiên và nhiều yêu cầu khác. |
| 10 — đông chó/ghế/ô/mèo | D được nhận xét đúng nhất nhưng sai màu ghế/chó. D là GLIGEN best-of-five. | Không có cơ sở nói nhánh sửa tốt nhất ở cảnh đông đối tượng. Đây là nhận xét, không có điểm số. |
| 11 — xe buýt/tàu/chó | C được nhận xét đúng nhất nhưng chó chưa giống, tàu sai màu, chưa thực tế. C là GLIGEN best-of-five. | Cần cải thiện chất lượng đề xuất và binding; không tự gán điểm hoặc giải thích thành tác động của CLIP guard. |
| 12 — chim/thỏ/vịt | Nhận xét tại D nói rõ cả A–D đều sinh ba vịt, thiếu thỏ, chim hòa trộn. | Lỗi nhiều instance và lớp gần nhau. Phạm vi lời bình là cả bốn; không lan truyền thành điểm số ở các dòng chưa chấm. |

## Kết luận được sửa sau review

**Giữ:** focus có ích rõ trong cảnh ngựa/chó; nhận xét này phù hợp với ảnh và phần lớn tín hiệu tự động.

**Thu hẹp:** trước đó ảnh thuyền được mô tả như một cải thiện về đối tượng. Người dùng chấm focus và sparse/global bằng nhau ở cảnh đó. Vì vậy chỉ được nói detector tăng điểm và hình ảnh thay đổi, không nói review xác nhận tăng độ đúng loại/số lượng.

**Bác bỏ việc dùng một metric làm kết luận toàn cảnh:** cảnh máy bay cho thấy false acceptance theo đánh giá người dùng ngay cả khi box endpoint đạt. Các cảnh ghế, voi, cây và phòng khách cho thấy màu, hướng, nền và tính tự nhiên bị bỏ sót. Không chuyển kết quả detector 6/12 thành “6 ảnh hoàn toàn đúng”.

Trong 7 cảnh có đủ điểm, focus cao hơn global/sparse về presence/count ở **một cảnh** (ngựa/chó); shape, attributes và realism không cải thiện ở bất kỳ cảnh đã chấm số nào so với global/sparse. Không nhóm phương pháp nào được cho realism=2 trên các cảnh có điểm. Điều này là bằng chứng phát triển có giới hạn, không phải tỷ lệ lỗi trên tập kiểm định đại diện.

## Hướng tối ưu sau khi đã có phản hồi

1. **Thêm năng lực xử lý vật thể thừa:** cảnh xe đạp đã được người dùng xác nhận, nên không còn chỉ là nghi ngờ detector. Thiết kế thao tác remove-extra riêng với add-missing/replace-wrong-class; bảo vệ các đối tượng đúng. Tuy nhiên, không xóa mọi box không được ghép cặp vì có thể chỉ là nhiều detection trên cùng một vật.
2. **Nâng chất lượng kiểm tra đối tượng và hình dáng:** đưa cảnh máy bay vào tập lỗi phát triển để kiểm tra whole-object so với bộ phận/cận cảnh. Đo hướng/shape riêng; không nới ngưỡng để tạo thêm “đạt”. Các lỗi ghế/voi cần kiểm soát theo từng đối tượng thay vì thêm global prompt.
3. **Đối chiếu với điều kiện instance mạnh hơn:** ưu tiên baseline có phrase + scribble/mask như InstanceDiffusion, với cùng thông tin đầu vào và kiểm tra chuyển đổi tọa độ. Kết quả này không ủng hộ chỉ tăng số lần lấy mẫu hoặc sửa nối tiếp trên nền hiện tại.
4. **Giữ focus là thành phần thử nghiệm**, không công bố thành model tốt nhất. Thử nghiệm tiếp theo phải tách tác động text/negative text/sketch và kiểm soát ngân sách, rồi đánh giá cảnh mới. Không dùng 12 cảnh đã review làm held-out test.

Không cần người dùng chấm lại hoặc chạy GPU để hoàn tất bước phân tích này. Chưa thay đổi trọng số, tham số sinh ảnh hoặc notebook mặc định trong commit tiếp nhận review. Nếu cần đánh giá một thay đổi cụ thể sau này, phải tạo bộ before/after mới kèm yêu cầu và định dạng đầu ra; không yêu cầu người mới tự mở log hoặc đo metric.

## Tái lập

```bash
python tools/analyze_human_review.py \
  --bundle-root /path/to/merged-export \
  --review docs/reviews/20261102_user_review/source.json \
  --out /tmp/ocsd-human-review-analysis
```

Công cụ từ chối sai run, sai hash ảnh, sai nhóm/nhãn, dòng trùng/thiếu, điểm ngoài thang, hoặc mâu thuẫn giữa hai biểu diễn của cùng câu trả lời. Test kiểm tra các trường hợp này và bảo toàn điểm trống/ảnh trùng.
