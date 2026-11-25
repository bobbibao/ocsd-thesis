"""Bobby 2026-11-26: slide 10 'Đóng góp đề tài' - develop each of the 4 cards (no 'mở rộng của Zhang'), fill the
space. Cards keep Bobby's colours/titles; bodies become 3 short bullets. Usage: slide10_contrib.py IN OUT"""
import copy, sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
p = Presentation(sys.argv[1]); s = p.slides[9]
INK = C(0x1F, 0x24, 0x30)
sh = list(s.shapes); byname = lambda n, i=0: [x for x in sh if x.name == n][i]
cards = [  # (card shape, title box, body box, title colour, bullets)
    (byname('Shape 4'), byname('Text 5'), byname('Text 6'), C(0xD9, 0x69, 0x2B), [
        ('Khung OCSD 5 mô-đun: ', 'xử lý “vật thể” xuyên suốt từ phác thảo đến ảnh.'),
        ('Mỗi vật được vẽ và học định danh riêng ', '(M2, M3) → hai vật cùng loại không lẫn vào nhau.'),
        ('Hợp thành cảnh hài hòa ', '(M4, M5): giữ vật ở các bước đầu rồi nhả dần → đủ vật và ảnh tự nhiên.')]),
    (byname('Shape 7'), byname('Text 8'), byname('Text 9'), C(0x2E, 0x5A, 0x9C), [
        ('QuickDraw-Scenes: ', 'phác thảo nét tay, kiểm soát riêng số vật (1, 3, 5, 8+) và độ phức tạp nét.'),
        ('COCO-Sketch: ', 'phác thảo dựng từ ảnh thực, cảnh đời thường.'),
        ('Mỗi cảnh có câu mô tả, mặt nạ và hộp của từng vật; ', 'tách riêng tập phát triển và tập đánh giá.')]),
    (byname('Shape 10'), byname('Text 11'), byname('Text 15', 1), C(0x6A, 0x3D, 0x9A), [
        ('Bộ độ đo theo 4 yêu cầu R1–R4: ', 'OPR, sai số số lượng, mIoU, RA; thêm PickScore và chấm mù cho độ tự nhiên.'),
        ('Chấm bằng OWLv2, ', 'khác bộ phát hiện dùng trong phương pháp → không “học tủ” công cụ chấm.'),
        ('Kiểm định thống kê ', '(dấu cặp, hiệu chỉnh Holm), giả thuyết đăng ký trước; so 5 phương pháp trên cùng SD 1.5.')]),
    (byname('Shape 13'), byname('Text 14'), byname('Text 15', 0), C(0x3F, 0x7F, 0x37), [
        ('Toàn bộ mã nguồn trên GitHub, ', 'chạy thí nghiệm bằng notebook Colab, kết quả lưu Google Drive.'),
        ('Lưu ảnh trung gian của từng mô-đun M1–M5 ', '(các ví dụ thật trong phần phương pháp).'),
        ('Kèm ứng dụng web minh họa ', 'và kiểm thử tự động cho từng mô-đun.')])]
H, Y = 2.5, (1.6, 4.25)
for k, (card, ttl, body, col, bullets) in enumerate(cards):
    top = Y[k // 2]; card.top, card.height = I(top), I(H)
    ttl.top = I(top + 0.15)
    body.top, body.height = I(top + 0.62), I(H - 0.72)
    tf = body.text_frame; tf.word_wrap = True
    for para in list(tf.paragraphs)[1:]: para._p.getparent().remove(para._p)
    first = tf.paragraphs[0]
    for r in list(first.runs): r._r.getparent().remove(r._r)
    for j, (lead, rest) in enumerate(bullets):
        para = first if j == 0 else tf.add_paragraph(); para.space_after = Pt(5)
        for t, b, c in (('•  ', True, col), (lead, True, INK), (rest, False, INK)):
            r = para.add_run(); r.text = t; r.font.size = Pt(13); r.font.bold = b; r.font.color.rgb = c; r.font.name = 'Arial'
s.notes_slide.notes_text_frame.text = (
    'Đề tài có bốn đóng góp. Thứ nhất là phương pháp OCSD gồm năm mô-đun, xử lý vật thể xuyên suốt từ phác thảo đến ảnh: '
    'mỗi vật được vẽ riêng và học định danh riêng, nên hai vật cùng loại không lẫn vào nhau; khi hợp thành cảnh, vật được giữ ở các bước đầu rồi nhả dần, '
    'nên vừa đủ vật vừa tự nhiên. Thứ hai là hai bộ dữ liệu kiểm thử: QuickDraw-Scenes với phác thảo nét tay, kiểm soát riêng số vật và độ phức tạp của nét, '
    'và COCO-Sketch dựng từ ảnh thực; mỗi cảnh có câu mô tả, mặt nạ và hộp của từng vật. Thứ ba là bộ đánh giá theo bốn yêu cầu R1 đến R4, '
    'chấm bằng OWLv2 khác với bộ phát hiện trong phương pháp, có kiểm định thống kê và giả thuyết đăng ký trước, so với năm phương pháp trên cùng Stable Diffusion 1.5. '
    'Thứ tư là toàn bộ mã nguồn, notebook chạy thí nghiệm, ảnh trung gian của từng mô-đun và ứng dụng web minh họa.')
p.save(sys.argv[2]); print('ok')
