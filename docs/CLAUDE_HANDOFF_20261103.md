# Bàn giao OCSD cho Claude — 03/11/2026

## Trạng thái quyết định

Mục tiêu của người dùng: sketch cảnh + caption → đúng loại, số lượng, hình dáng, vị trí, quan hệ và ảnh tự nhiên. Muốn paper mạnh nhưng **chưa có bằng chứng SOTA hoặc cải thiện tổng thể trên tập chưa xem**. Không biến mong muốn thành kết luận. Người dùng dùng Colab Pro, nhạy với compute units, muốn commit/push main và chạy `notebooks/RUN_ALL.ipynb` như cũ. Nếu cần review, phải có PNG side-by-side, hướng dẫn dễ hiểu và mẫu trả lời; người dùng mới và HTML từng mở trắng.

**Hiện tại:** CPU preflight của InstanceDiffusion với trọng số thật đã PASS. Đã triển khai canary GPU 4 ảnh, nhưng **chưa nhận kết quả canary GPU**, chưa có ảnh chứng minh nhánh mới tốt hơn. Không yêu cầu chạy lại CPU/foreground sweep. Default RUN_ALL là `RUN_MODE="direct_canary"`; GPU L4/A100, RUN_ID trống. Giới hạn child process 600 giây gồm tải/nạp/sinh ảnh, không gồm cài thư viện/thời gian GPU nhàn rỗi. Chưa nâng paper/full.

## Những gì đã làm và bằng chứng cần đọc

1. **Tính tái lập, giới hạn compute, upload/export:** thêm namespace từng run, source/dependency/model/data identities, checksum, resume từ kết quả hợp lệ, bảo vệ khỏi dùng cache lệch cấu hình; upload GitHub lỗi không bắt chạy lại model, kết quả giữ Drive. Export chia ZIP dưới giới hạn đính kèm. Đọc `docs/OPTIMIZATION_AUDIT.md`, `docs/EXPERIMENT_CHANGELOG.md`, `docs/EXPERIMENT_RUNBOOK.md`, `docs/QUALITY_SCREEN.md`, `ocsd/provenance.py`, `ocsd/runner.py`, `ocsd/sync.py`.
2. **Screen ban đầu và M2 sparse/missing-only repair:** audit ảnh và bottleneck sinh ứng viên đối tượng; thêm sửa riêng phần thiếu, diagnostic, budget, chia sẻ phần chuẩn bị giữa đối chứng. Có bằng chứng tiết kiệm/cải thiện cục bộ; không đồng nghĩa ảnh tốt toàn diện. Đọc `docs/SCREEN_20261028_REVIEW.md`, `docs/SPARSE_REPAIR_20261029_RESULTS.md` và CSV đi kèm. Các số chính xác phải lấy bảng của cùng run; không trộn run hoặc chỉ báo mean tốt nhất.
3. **Semantic routing/GLIGEN:** thử routing có giới hạn, đối chứng chia sẻ, review ẩn nhãn. Semantic screen âm tính; quay về hướng sparse phát triển. Đọc `docs/SEMANTIC_20261029_RESULTS.md`, `docs/SEMANTIC_SCREEN.md`.
4. **Instance-focused repair:** sửa theo instance và siết đánh giá layout. Có cải thiện cục bộ nhưng chưa tăng joint success đủ để mở rộng. Review người dùng cho thấy detector/box success bỏ sót lỗi nhìn thấy. Đọc `docs/INSTANCE_20261102_RESULTS.md`, `docs/HUMAN_REVIEW_20261102_RESULTS.md`, `docs/RESEARCH_20261102.md`, `docs/RESEARCH_NEXT_INSTANCE_REPAIR.md`.
5. **Count-aware replacement:** xe đạp vàng xuất hiện nhưng méo, thiếu tự nhiên; điểm đếm tự động đạt chưa qua visual gate. Đọc `docs/COUNT_20261102_RESULTS.md`, `docs/COUNT_ROUTE_AUDIT.json`, `ocsd/count_repair.py`.
6. **Clean canvas/blending:** kiểm tra init/blend, ảnh trung gian và proposal; phát hiện nền gỗ lọt vào bánh xe qua mask/cutout và nền Telea nhòe. Đọc `docs/CLEAN_20261103_RESULTS.md`, `docs/CLEAN_SCREEN.md`, `ocsd/clean_repair.py`.
7. **Learned foreground:** BiRefNet-lite CPU, mask cứng/mềm và anchor nền cảnh. Audit 81 artifact, tái lập seed trên CPU, input dùng chung và pixel vùng bảo vệ. Đọc `docs/FOREGROUND_20261103_RESULTS.md`, `tools/audit_foreground_run.py`, `ocsd/foreground.py`. Không tiếp tục sweep cùng xe đạp để chọn ảnh đẹp.
8. **Review dễ dùng:** `tools/build_human_review.py` nhúng ảnh HTML tĩnh trước JS, fallback PNG nguyên ảnh, roundtrip form và kiểm tra trình duyệt. Người dùng đã review A–D; không yêu cầu chấm lại. File gốc/hashes/nhãn: `docs/reviews/foreground_20261103_user_feedback.json`, `docs/reviews/foreground_20261103_ABCD.png`.
9. **Hướng direct instance conditioning:** tích hợp lõi chính thức InstanceDiffusion đã pin; không cài fork diffusers vì bản đã kiểm tra null hóa scribble. Đây là phương pháp bên ngoài/đối chứng mạnh, **không gọi là kiến trúc mới của OCSD**. Adapter nhận instance phrase/box/sketch có sẵn, 20 điểm trên nét mực; không dùng oracle mask, không tự tách sketch thô. Đọc `docs/DIRECT_INSTANCE_PREFLIGHT.md`, `ocsd/direct_instances.py`.
10. **Tối ưu và sửa tương thích direct:** cache UniFusion trong từng ảnh, reuse null condition, khôi phục learned input-conv sau mỗi ảnh/khi lỗi, count actual UNet calls; CLIP dùng pooler trước projection (Transformers 5 đổi nghĩa field trong CLIPModel), map strict tên trọng số v4→v5, không chạy vision tower không dùng. Tiny official-core CPU: 34 evaluations conditioner → 4, latent trùng từng giá trị, UNet calls vẫn 34; **không phải benchmark tăng tốc full model**. Các test còn kiểm tra scribble sensitivity, state cleanup, timeout và nonfinite dừng sớm.

## Kết quả foreground cuối cùng (đã review)

Run `20261103T035056Z_dc40e4b5`:

| Nhãn | Method | mIoU | CLIP |
|---|---|---:|---:|
| D | instance_repair_count_clean | 0.73340 | 33.71369 |
| C | instance_repair_count_context | 0.73851 | 33.70676 |
| B | instance_repair_count_matte_hard | 0.73300 | 33.42460 |
| A | instance_repair_count_matte_soft | 0.73172 | 33.29917 |

Cả bốn OPR=1, count error=0, RA=1, joint box success=1 trên cảnh này. Work standalone 280 mỗi nhánh; shared actual 460. Learned foreground CPU forward khoảng 8.825 giây; first hard wall khoảng 14.42 giây, soft reuse. Không so số UNet calls giữa kiến trúc khác như cùng FLOPs.

Review nguyên văn: **“tất cả các ảnh A,B,C,D đều có đủ số lượng, hình dáng nhưng ảnh A,B là chính xác nhất (giống nhau) nhưng còn chưa tự nhiên lắm”**. A/B giống theo cảm nhận, không phải file byte-identical. Không tự suy thành điểm số. Ưu tiên hiện tại: tăng naturalness mà giữ count/shape; các score detector không thay thế review.

## CPU preflight mới nhất

- Run lỗi `20261103T140807Z_e0a5503f`: tải xong, dừng do logger Tee thiếu `isatty()` lúc CLIP loading report. Đã tái hiện lỗi bằng CLIP checkpoint nhỏ cục bộ, sửa giao diện stream, giữ strict learned-weight checks. `docs/DIRECT_PREFLIGHT_20261103_RESULTS.md` là lịch sử lỗi, không phải trạng thái hiện tại.
- Run **PASS** `20261103T142406Z_013a5f0e`: `docs/DIRECT_PREFLIGHT_20261103_PASS.md`, JSON nguyên kết quả `docs/direct_preflight_20261103_pass.json`. Phrase 2×768, conditioner 1×184×768, cache exact, đổi scribble max difference 29.3722782, UNet latent 1×4×8×8 hữu hạn. Wall 123.32s gồm load/check, không phải throughput sinh ảnh. Python 3.13.18, torch 2.11.0+cpu, Transformers 5.17.0. **scene_generation=false, quality_validated=false, gpu_fp16_validated=false**.
- Source commit official: `722900ef03e3579fb693e65a1550333a4a162581`.
- HF InstanceDiffusion: `fc74d76cde4a3dd0c3894e1764c8cf7299c77595`; CLIP: `32bd64288804d66eefd0ccbe215aa642df71cc41`.
- Checkpoint SHA256: `b4de3f7877a1860b52837afe3a767ca92a2ef5084056048b503af80027cde1d1`.
- Tải mới 13.6 GB checkpoint + 1.71 GB CLIP, chưa gồm cache/tệp tạm. Môi trường Codex bị chặn HF và không có GPU; CPU thật được người dùng chạy trên Colab. Không giả vờ đã test GPU tại Codex.

## Đang làm / bước tiếp theo cụ thể

Canary đã code và push, chờ người dùng chạy/gửi `ocsd_direct_canary_….zip`:

- `docs/DIRECT_CANARY.md`, `assets/direct_canary_v1/protocol.json`, `assets/direct_canary_v1/scene.json`, `assets/direct_canary_v1/README.md` là protocol/input đã khóa trước output.
- Một cảnh mới: xe đạp vàng trái, xe đạp xanh phải, ghế băng gỗ phía sau trong công viên. QuickDraw thật, có nguồn/key/license, lấy các record recognized đầu tiên từ index 3000; không tuyên bố held-out/training-data independence.
- Seeds 0/1 × direct_box/direct_scribble = 4 ảnh; cùng noise trong cặp. 50 PLMS steps, MIS .36, alpha .8, CFG 7.5; 216 UNet calls/ảnh, 864 cho bốn ảnh hoàn tất. Chi phí attempt lỗi cần tính riêng. Không refiner/rerank/tuning.
- UNet autocast FP16, VAE FP32; GPU cần >=20 GiB VRAM. Đây là bước kiểm tra GPU/ảnh, chưa có kết quả. Checkpoint SHA/revisions khóa theo CPU PASS.
- `ocsd/direct_canary.py`, `tools/run_direct_canary.py`: completion/checksum, contract theo code/libs/GPU/protocol, resume phần thiếu, lịch sử attempts, timeout, xuất ZIP cả khi fail. Không tự thử tham số khác sau lỗi.
- Review: PNG nguyên pixel `SO_SANH_seed0.png`, `SO_SANH_seed1.png`, A/B đảo theo seed; `review_key.json` cho analyst; `HUONG_DAN.txt`, `REVIEW_TEMPLATE.txt` cho người dùng. Chấm count/shape/layout/relations/color và naturalness 0/1/2/không rõ. Không cần HTML/export screen.
- **Canary chưa có ControlNet và OCSD soft-matte**, nên chỉ khảo sát thêm scribble và tính khả thi; không kết luận thắng các phương pháp đó.

Sau khi nhận ZIP: kiểm tra checksum, contract, đủ bốn job/seed, actual calls, peak VRAM, timeout/failure history; xem PNG gốc, đối chiếu review trước quyết định. Nếu lỗi kỹ thuật: sửa có test, commit/push, chỉ đề nghị chạy lại phần cần thiết. Nếu count/shape/naturalness chưa đạt: phân tích failure, không tự scale. Nếu khả quan: chốt protocol trước output cho nhiều cảnh/seed và đối chứng ControlNet + OCSD soft-matte, ghép cặp dữ liệu/ngân sách, ghi đủ prep/failed work; tách development khỏi tập đánh giá cuối, báo bất định và kết quả âm tính. Bảo toàn credit tác giả InstanceDiffusion; nghiên cứu đóng góp mới cần ablation độc lập.

## Kiểm thử / vận hành / lịch sử Git

Lượt gần nhất: 18 test `test_direct*.py` và 15 `test_reproducibility.py` đạt; 5 notebook validate/AST và tái sinh nhất quán, 9 stage cũ giữ nguyên. Các fixture GPU orchestration dùng pixel tổng hợp, không phải pretrained quality. Kiểm tra CLI CPU từ chối GPU canary và vẫn đóng gói lỗi. Lệnh lõi:

```bash
OCSD_INSTDIFF_SOURCE=/path/to/pinned/InstanceDiffusion python -m unittest discover -s tests -p 'test_direct*.py' -v
python -m unittest discover -s tests -p test_reproducibility.py -v
python tools/make_notebooks.py
```

Người dùng yêu cầu đổi author/committer các commit Bobby Le thành bobbibao. Xem `docs/AUTHOR_REWRITE_MAP.tsv` để ánh xạ SHA cũ→mới. Giữ nguyên SHA lịch sử trong manifest/result như bằng chứng gốc; không sửa để giả rằng thí nghiệm đã chạy trên code mới. Tree của các commit ánh xạ phải giống nhau. Sau rewrite, clone mới hoặc backup work chưa commit rồi đồng bộ có kiểm soát; không merge lịch sử cũ trở lại main. Claude có branch riêng cần kiểm tra trước khi rebase/cherry-pick để tránh lặp commit. Không xóa work của Claude/người dùng.

Không đưa token người dùng vào prompt/log/file. Token từng xuất hiện trong chat cần thu hồi; dùng quyền truy cập đã cấu hình. User cho phép commit và push main để chạy Colab; không tự merge thay đổi khác của Claude hoặc chạy GPU diện rộng.

### Số liệu bổ sung đã đối chiếu tài liệu gốc

Sparse run 20261029T000432Z_6ad81320: M2 candidates 70→24 (giảm65.7%); full/sparse OPR cùng .9023, count error cùng1.6667, RA cùng .7638; mIoU .703370→.704652, CI delta[-.000121,.003525], Holm p=1. Mean standalone12.3968→5.2407s, nhưng không randomized timing; không hứa speedup2.37× tổng quát. So best-of-2 chưa chứng minh tradeoff tốt hơn.

Instance run20261102T165132Z_e9ea4c56: global→focused OPR .8493→.8826, mIoU .6910→.7207, count error2.0833→1.7500, RA .6460→.7314; cả bốn phương pháp vẫn joint success6/12. Focused work151.67/ảnh so GLIGEN best-of-5 là300, nhưng không phải FLOPs hay bằng chứng chất lượng tương đương ở nửa chi phí. A100 ở lượt trước và L4 ở lượt này không được gộp thành so sánh có kiểm soát.
