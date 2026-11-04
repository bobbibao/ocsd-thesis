Bạn tiếp tục dự án bobbibao/ocsd-thesis từ công việc trước đây của bạn và phần Codex đã làm tới 03/11/2026. Hãy đọc tài liệu, đối chiếu code và dữ liệu thực, rồi tiếp tục nghiên cứu/tối ưu; không chỉ đề xuất kế hoạch.

Tôi gửi kèm CLAUDE_HANDOFF_20261103.md. Nếu file chưa có trên main, dùng bản đính kèm này. Trước sửa code, kiểm tra git status, branch và work chưa commit của bạn; không ghi đè chúng, không merge lại lịch sử cũ sau author rewrite. Kiểm tra AUTHOR_REWRITE_MAP.tsv nếu có; SHA cũ trong manifest là bằng chứng lịch sử, không sửa chúng thành SHA mới.

Mục tiêu: từ phác thảo cảnh + caption sinh ảnh đúng loại, số lượng, vị trí/hình dáng, quan hệ và tự nhiên. Tôi muốn nghiên cứu mạnh nhưng không được tuyên bố tốt nhất/SOTA/đủ paper khi chưa có bằng chứng. Tôi mới, chạy bằng notebooks/RUN_ALL.ipynb trên Colab; cần commit và push main khi hoàn tất code/checks. Compute units đắt: ưu tiên phân tích dữ liệu có sẵn, lượt thử nhỏ có budget/resume; không tự mở paper/full hoặc sweep nhiều lần cùng cảnh.

Đọc theo thứ tự (đường dẫn tính từ repo):
1. docs/CLAUDE_HANDOFF_20261103.md (bản đính kèm nếu chưa trên main).
2. docs/DIRECT_PREFLIGHT_20261103_PASS.md và docs/direct_preflight_20261103_pass.json.
3. docs/DIRECT_CANARY.md; assets/direct_canary_v1/protocol.json, scene.json, README.md.
4. ocsd/direct_instances.py, ocsd/direct_canary.py, tools/run_direct_canary.py, tools/make_notebooks.py, notebooks/RUN_ALL.ipynb; tests/test_direct*.py.
5. docs/FOREGROUND_20261103_RESULTS.md; docs/reviews/foreground_20261103_user_feedback.json.
6. docs/HUMAN_REVIEW_20261102_RESULTS.md, docs/COUNT_20261102_RESULTS.md, docs/CLEAN_20261103_RESULTS.md.
7. docs/SPARSE_REPAIR_20261029_RESULTS.md, docs/SEMANTIC_20261029_RESULTS.md, docs/INSTANCE_20261102_RESULTS.md và CSV liên quan.
8. docs/RESEARCH_20261102.md, docs/RESEARCH_NEXT_INSTANCE_REPAIR.md, docs/OPTIMIZATION_AUDIT.md, docs/EXPERIMENT_CHANGELOG.md, docs/EXPERIMENT_RUNBOOK.md.
9. Đối chiếu docs/IMPROVEMENT_PLAN.md và công việc cũ của bạn với các bằng chứng mới; không coi mọi kế hoạch lịch sử còn là khuyến nghị hiện tại.

Các trích dẫn/bằng chứng quan trọng:
- Review nguyên văn của tôi: “tất cả các ảnh A,B,C,D đều có đủ số lượng, hình dáng nhưng ảnh A,B là chính xác nhất (giống nhau) nhưng còn chưa tự nhiên lắm”. A=soft matte, B=hard matte; không tự chuyển thành điểm số; không yêu cầu chấm lại bộ đó.
- docs/SEMANTIC_20261029_RESULTS.md: “Stop this semantic policy version.” Giữ kết quả âm tính, không chạy lại để tìm kết quả đẹp.
- docs/INSTANCE_20261102_RESULTS.md: “The same six scenes pass the joint endpoint for all four methods; there is no improvement on that endpoint.” Detector/box success không đo toàn bộ silhouette/thuộc tính/thẩm mỹ.
- docs/DIRECT_PREFLIGHT_20261103_PASS.md: “chưa sinh ảnh, chưa xác minh FP16/full-resolution GPU, chưa đo số lượng/hình dáng/naturalness hoặc so sánh phương pháp”. CPU PASS không phải quality PASS.
- docs/DIRECT_CANARY.md: “Canary chưa có ControlNet và OCSD soft-matte”, vì vậy chưa thể kết luận thắng các phương pháp đó.

Trạng thái mới nhất:
CPU preflight 20261103T142406Z_013a5f0e đã PASS với EMA checkpoint thật, đủ text weights, conditioner cache exact, scribble có tác động và UNet trên latent 8×8 hữu hạn. Không cần chạy CPU lại. Lỗi Tee.isatty của lượt trước đã sửa.

Codex đã push canary GPU (SHA lịch sử fb4311d0050ff401ef4d1a175bebe811ddbef4bd, có thể đổi SHA khi đổi author). Chưa nhận ZIP kết quả GPU của canary. RUN_ALL mặc định direct_canary; GPU L4/A100 >=20 GiB; RUN_ID trống. Một cảnh mới hai xe đạp vàng/trái, xanh/phải và một ghế băng; seed 0/1 × box/scribble =4 ảnh; 50 PLMS, MIS .36, alpha .8, CFG7.5. 864 UNet sample calls cho bốn ảnh hoàn tất, không phải FLOPs hay toàn bộ chi phí. Timeout child600s gồm tải/nạp/sinh, chưa gồm cài và nhàn rỗi. UNet autocast FP16, VAE FP32. Đổi runtime có thể tải lại13.6GB checkpoint +1.71GB CLIP. Có checksum/resume/failed attempt history, không retry cấu hình khác tự động.

InstanceDiffusion là external baseline của tác giả khác (https://arxiv.org/abs/2402.03290), không gọi là kiến trúc OCSD mới. Official source pin722900ef03e3579fb693e65a1550333a4a162581. Bản diffusers port được kiểm tra null hóa scribble nên adapter dùng official core. Đầu vào vẫn cần instance phrase/box/sketch có sẵn, 20 điểm mực mất thông tin; chưa phải end-to-end sketch thô. Không dùng oracle masks. Phải ghi rõ các giới hạn này trong paper.

Việc cần làm:
A. Xác nhận trạng thái repo và đọc protocol; bảo toàn work trước đây của bạn.
B. Nếu có ZIP canary mới, audit checksum/code/libs/model revisions, đủ job/seed, actual compute/VRAM/history trước khi chấm ảnh. Nếu chưa có, giữ lượt thử4ảnh này làm bước tiếp theo, không tự thêm sweep.
C. Xem ảnh gốc và review tôi. Khi cần tôi chấm, cung cấp PNG side-by-side, nhãn A/B, hướng dẫn từng tiêu chí và mẫu kết quả TXT hoặc form hoạt động. Không dùng HTML duy nhất. Canary đã xuất SO_SANH_seed0.png, SO_SANH_seed1.png, HUONG_DAN.txt, REVIEW_TEMPLATE.txt.
D. Nếu lỗi kỹ thuật, tái hiện/sửa/test, commit/push; chỉ chạy lại phần cần thiết. Nếu chất lượng kém, phân tích failure và dừng mở rộng. Nếu khả quan, chốt protocol nhiều cảnh/seed, ControlNet + OCSD soft-matte, ablations và chi phí đầy đủ trước output. Tách development/held-out, dùng paired seeds, uncertainty, kiểm tra shape/naturalness chứ không chỉ detector.
E. Nghiên cứu các phương pháp mạnh từ nguồn gốc chính thức/paper; phân biệt tái hiện baseline, engineering optimization và đóng góp nghiên cứu mới. Không bịa citation, metric hay kết quả GPU. Giải thích giả thuyết nào được bằng chứng ủng hộ/bác bỏ và thí nghiệm tiếp theo kiểm tra điều gì.
F. Khi hoàn tất phần có thể làm, kiểm thử thích hợp, commit/push main theo workflow của tôi, báo chính xác phiên bản, giới hạn và bước Colab tiếp theo. Không sử dụng hoặc chép token từ cuộc trò chuyện; token từng bị lộ cần được thu hồi.

Mở đầu bằng tóm tắt trạng thái bạn đã hiểu, các bằng chứng đã đọc và bước tiếp theo cụ thể; sau đó thực hiện phần việc có thể làm ngay. Không hứa paper10điểm hoặc model tốt nhất trước thí nghiệm.
