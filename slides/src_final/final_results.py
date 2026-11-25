"""Result slides -> FINAL OCSD (SD 1.5, M1-M5, M3 on, harmonised M4), numbers from results/m4_study_v1
(72 QuickDraw-Scenes x seeds 0,1; docs/M4_STUDY_RESULTS.md). No claim that OCSD beats GLIGEN or Collage on structure.
Usage: python final_results.py IN.pptx OUT.pptx (run in this folder; charts in charts/ from charts_final.py)."""
import copy, re, sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

SRC, OUT = sys.argv[1:3]
p = Presentation(SRC)
INK, MUT, ORG, BLU, GRY, PEA = C(0x1F,0x24,0x30), C(0x5B,0x62,0x70), C(0xD9,0x69,0x2B), C(0x2E,0x5A,0x9C), C(0xF3,0xF4,0xF6), C(0xFB,0xE9,0xDD)
GRN_BG, YEL_BG, BLU_BG = C(0xE6,0xF1,0xE6), C(0xFE,0xF4,0xD8), C(0xE6,0xEE,0xFA)
def alltext(s): return ' '.join(t.text or '' for t in s._element.iter(qn('a:t')))
def find(marker): return next(s for s in p.slides if marker in alltext(s))
def sh(s, name): return [x for x in s.shapes if x.name == name][0]
def settext(shape, lines):
    tf = shape.text_frame; r0 = tf.paragraphs[0].runs[0]
    for para in list(tf.paragraphs)[1:]: para._p.getparent().remove(para._p)
    for r in list(tf.paragraphs[0].runs)[1:]: r._r.getparent().remove(r._r)
    r0.text = lines[0]
    for l in lines[1:]:
        para = tf.add_paragraph(); r = para.add_run(); r.text = l
        r.font.size, r.font.bold, r.font.name = r0.font.size, r0.font.bold, r0.font.name
        if r0.font.color.type is not None: r.font.color.rgb = r0.font.color.rgb
def card_text(shape, head, body):
    ps = shape.text_frame.paragraphs
    ps[0].runs[0].text = head
    for r in ps[0].runs[1:]: r.text = ''
    ps[1].runs[0].text = body
    for r in ps[1].runs[1:]: r.text = ''
    for para in ps[2:]: para._p.getparent().remove(para._p)
def repic(s, old, path, w=None, h=None, x=None, y=None):
    new = s.shapes.add_picture(path, x if x is not None else old.left, y if y is not None else old.top,
                               w if w is not None else old.width, h if h is not None else old.height)
    old._element.addprevious(new._element); old._element.getparent().remove(old._element); return new
def rm(x): x._element.getparent().remove(x._element)
def notes(s, text): s.notes_slide.notes_text_frame.text = text
def box(s, x, y, w, h, fill, rounded=True):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, I(x), I(y), I(w), I(h))
    b.fill.solid(); b.fill.fore_color.rgb = fill; b.line.fill.background(); b.shadow.inherit = False
    if rounded: b.adjustments[0] = 0.06
    return b
def tb(s, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=4):
    t = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in 'margin_left', 'margin_right', 'margin_top', 'margin_bottom': setattr(tf, m, 0)
    for k, line in enumerate(lines):
        para = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); para.alignment = align; para.space_after = Pt(space)
        for txt, sz, col, b, *it in line:
            r = para.add_run(); r.text = txt; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b; r.font.name = 'Arial'
            r.font.italic = bool(it and it[0])
    return t
TITLE_PROTO = copy.deepcopy(sh(find('Khi số đối tượng tăng'), 'Text 1')._element)
def set_title(s, text):
    t = [x for x in s.shapes if x.name == 'Text 1']
    if not t:
        s.shapes._spTree.append(copy.deepcopy(TITLE_PROTO)); t = [x for x in s.shapes if x.name == 'Text 1']
    settext(t[0], [text])
def clear(s, keep=('Text 0', 'Text 1', 'Text 2', 'Text 3')):
    tree = s.shapes._spTree
    for el in list(tree)[2:]:
        nm = el.find('.//' + qn('p:cNvPr'))
        if el.tag.endswith('AlternateContent') or nm is None or nm.get('name') not in keep: tree.remove(el)
FOOT = '72 cảnh QuickDraw-Scenes × 2 hạt giống, chấm bằng OWLv2; kiểm định dấu cặp, hiệu chỉnh Holm.'

# ---------------------------------------------------------------- tuning slide -> development of the new M4
s = find('α là gì')
clear(s)
set_title(s, 'Tinh chỉnh M4-Tổng hợp cảnh trên tập phát triển')
s.shapes.add_picture('charts/f_dev.png', I(0.6), I(1.5), I(7.6), I(7.6 * 3.3 / 6.4))
tb(s, 0.6, 5.5, 7.6, 0.5, [[('36 cảnh của tập tinh chỉnh riêng × 2 hạt giống, không dùng để đánh giá. Pick = PickScore (độ tự nhiên). Số ở đây chỉ dùng để chọn cấu hình, không phải kết quả.', 9, MUT, False, True)]])
cards = [(GRY, 'Thử 10 cấu hình M4', 'M4 cũ, M4 mới, 7 biến thể mỗi cái đổi một chỗ, và bản kết hợp các thay đổi có lợi. Hình: 7 cấu hình tiêu biểu.'),
         (BLU_BG, 'Luật chọn viết trước khi chạy', 'Điểm = trung bình (OPR, mIoU, RA) + 0,05 × mức tăng PickScore so với M4 cũ. Máy tự chọn, khóa lại trước khi chạy đánh giá.'),
         (PEA, 'Chọn: kết hợp', 'α = 0,2 · nhả trong 40% số bước · ControlNet cảnh 0,6 · LoRA từ 45% số bước. Đây là OCSD cuối cùng.')]
for k, (bg, h, b) in enumerate(cards):
    y = 1.5 + k * 1.75
    box(s, 8.5, y, 4.2, 1.6, bg)
    tb(s, 8.68, y + 0.1, 3.85, 1.4, [[(h, 13, ORG if k == 2 else INK, True)], [(b, 11, INK, False)]], space=4)
notes(s, 'Phần dựng cảnh M4 được làm lại và tinh chỉnh trên tập phát triển riêng, 36 cảnh, không dùng để đánh giá. Nhóm thử mười cấu hình: M4 cũ, M4 mới, bảy biến thể mỗi cái chỉ đổi một chỗ, và bản kết hợp các thay đổi có lợi. '
         'Luật chọn được viết trước khi chạy: điểm bằng trung bình của OPR, mIoU và RA, cộng một phần nhỏ mức tăng độ tự nhiên PickScore. Máy tự chọn và khóa cấu hình trước khi tạo bất kỳ ảnh đánh giá nào. '
         'Cấu hình thắng là kết hợp: alpha 0,2, nhả vật dần trong 40 phần trăm số bước, ControlNet cảnh 0,6 và chỉ bật LoRA từ 45 phần trăm số bước. Đây là OCSD cuối cùng dùng cho mọi kết quả sau.')

# ---------------------------------------------------------------- by object count
s = find('Khi số đối tượng tăng')
repic(s, sh(s, 'Picture 12'), 'charts/f_count.png')
settext(sh(s, 'Text 4'), ['63% so với 39%'])
settext(sh(s, 'Text 5'), ['Giữ đối tượng ở cảnh 8+ đối tượng: OCSD so với mô hình nền. Từ 1 lên 8+ đối tượng, OCSD giảm 38 điểm, mô hình nền giảm 58 điểm.'])
card_text(sh(s, 'Text 7'), 'Chưa giải quyết xong', 'Ở 8+ đối tượng, Collage (69%) vẫn giữ được nhiều hơn; GLIGEN 55%. Mọi phương pháp đều giảm mạnh. 18 cảnh 8+, phân tích thăm dò.')
notes(s, 'Biểu đồ này cho thấy điều đề tài quan tâm nhất: khi số đối tượng tăng thì chuyện gì xảy ra. Mô hình nền, đường xám nét đứt, giữ 97 phần trăm đối tượng ở cảnh một đối tượng nhưng chỉ còn 39 phần trăm ở cảnh 8 đối tượng trở lên. '
         'OCSD, đường cam đậm, giảm chậm hơn nhiều, còn 63 phần trăm; đường cam nhạt là OCSD với M4 cũ, còn 52 phần trăm. Ở mức 8+, Collage vẫn giữ được nhiều hơn, 69 phần trăm; GLIGEN 55 phần trăm. '
         'Mỗi mức chỉ có 18 cảnh nên đây là phân tích thăm dò. Cảnh rất đông vẫn là bài toán chưa giải xong.')

# ---------------------------------------------------------------- comparison: 4 charts, 4 cards
s = find('OCSD so với các phương pháp khác')
clear(s)
for k, f in enumerate(['opr', 'miou', 'ra', 'pick']):
    s.shapes.add_picture(f'charts/f_{f}.png', I(0.45 + k * 3.12), I(1.45), I(3.05), I(3.05 * 2.75 / 3.1))
cards = [('Giữ đối tượng', '79% so với 66% (nền), 62% (T2I-Adapter), 63% (Zhang và cs.): hơn rõ. Ngang GLIGEN 80% và Collage 82%.'),
         ('Vị trí, hình dạng', '0,48: ngang mô hình nền (0,49); thấp hơn GLIGEN (0,65) và Collage (0,61), khác biệt có ý nghĩa.'),
         ('Đúng tương quan', '51% so với 32% (nền), 28%, 34%: hơn rõ. Collage (62%) và GLIGEN (56%) cao hơn, chưa có ý nghĩa.'),
         ('Độ tự nhiên', '21,89: cao hơn GLIGEN (21,41) và Collage (21,52), có ý nghĩa; hơn M4 cũ (21,16).')]
for k, (h, b) in enumerate(cards):
    x = 0.6 + k * 3.05
    box(s, x, 4.3, 2.9, 2.05, PEA if k in (0, 3) else GRY)
    tb(s, x + 0.15, 4.4, 2.6, 1.9, [[(h, 13, ORG if k in (0, 3) else INK, True)], [(b, 10.5, INK, False)]], space=4)
tb(s, 0.6, 6.5, 12.1, 0.45, [[(FOOT + ' “Hơn rõ”, “có ý nghĩa” = Holm p < 0,05; “ngang” = khác biệt không có ý nghĩa. Trả lời Q2.', 9.5, MUT, False, True)]])
notes(s, 'Đây là kết quả chính trên 72 cảnh, mỗi cảnh chạy 2 hạt giống, chấm tự động bằng OWLv2, so với sáu phương pháp. Thanh cam đậm là OCSD cuối cùng, thanh cam nhạt là OCSD với M4 cũ. '
         'Giữ đối tượng: OCSD 79 phần trăm, hơn rõ mô hình nền 66, T2I-Adapter 62 và Zhang 63; ngang GLIGEN 80 và Collage 82, khác biệt không có ý nghĩa thống kê. '
         'Vị trí, hình dạng: 0,48, ngang mô hình nền, nhưng thấp hơn GLIGEN và Collage, khác biệt có ý nghĩa; đây là điểm yếu nhóm nói rõ. '
         'Đúng tương quan: 51 phần trăm, hơn rõ mô hình nền 32; Collage và GLIGEN cao hơn nhưng chưa có ý nghĩa. '
         'Độ tự nhiên theo PickScore: OCSD cao nhất trong nhóm này, hơn GLIGEN và Collage có ý nghĩa. Nhóm không nói OCSD tốt nhất mọi độ đo: OCSD giữ đối tượng ngang GLIGEN và Collage, tự nhiên hơn, nhưng vị trí kém hơn.')

# ---------------------------------------------------------------- answers to Q1-Q3
s = find('Trả lời ba câu hỏi nghiên cứu')
for nm, v in [('TextBox 10', '97%'), ('TextBox 13', '64%'), ('TextBox 16', '39%'),
              ('TextBox 23', '79%'), ('TextBox 26', '80%'), ('TextBox 29', '82%'),
              ('TextBox 36', '51%'), ('TextBox 39', '1,36'), ('TextBox 42', '81%')]:
    settext(sh(s, nm), [v])
settext(sh(s, 'TextBox 24'), ['giữ đối tượng', 'OCSD (nền 66%)'])
settext(sh(s, 'TextBox 30'), ['giữ đối tượng', 'Collage'])
settext(sh(s, 'TextBox 37'), ['đúng tương quan', '(nền 32%)'])
settext(sh(s, 'TextBox 40'), ['độ tự nhiên 0–2', '(M4 cũ: 0,44)'])
settext(sh(s, 'TextBox 43'), ['giữ đối tượng khi', 'bỏ M5(b) (đủ: 68%)*'])
settext(sh(s, 'TextBox 21'), ['Có: hơn rõ mô hình nền, giữ đối tượng ngang GLIGEN và Collage; nhưng vị trí (mIoU) còn thấp hơn hai phương pháp này.'])
settext(sh(s, 'TextBox 34'), ['Đúng tương quan tăng rõ so với nền; M4 mới làm ảnh tự nhiên hơn hẳn. Nhưng bỏ riêng M5(a), M5(b) không làm kết quả giảm.'])
settext(sh(s, 'TextBox 45'), [FOOT + ' Độ tự nhiên: chấm mù 0–2 trên 36 ảnh, một người chấm. *Cắt bỏ: 18 cảnh, lần chạy luận văn (M4 cũ), không thành phần nào khác biệt có ý nghĩa (Holm p ≥ 0,24).'])
for para in sh(s, 'TextBox 45').text_frame.paragraphs:
    for r in para.runs: r.font.size = Pt(10)
notes(s, 'Nhóm trả lời ba câu hỏi nghiên cứu. Q1: mô hình nền mất đối tượng rất nhanh khi số đối tượng tăng, từ 97 phần trăm ở cảnh một đối tượng xuống 39 phần trăm ở cảnh 8+; độ phức tạp nét vẽ ảnh hưởng ít hơn. '
         'Q2: tách từng đối tượng rồi dựng cảnh giúp OCSD giữ 79 phần trăm đối tượng, hơn rõ mô hình nền 66, và ngang GLIGEN 80, Collage 82, khác biệt không có ý nghĩa thống kê; nhưng vị trí theo mIoU còn thấp hơn hai phương pháp này. '
         'Q3: đúng tương quan tăng rõ so với mô hình nền, 51 so với 32 phần trăm; M4 mới làm ảnh tự nhiên hơn hẳn, điểm chấm mù 1,36 so với 0,44 của M4 cũ. Tuy nhiên trong nghiên cứu cắt bỏ ở lần chạy luận văn, bỏ riêng chú ý theo vùng hay dẫn hướng năng lượng không làm kết quả giảm. Nhóm báo cáo trung thực các kết quả âm này.')

# ---------------------------------------------------------------- limitations
s = find('Hạn chế và hướng phát triển')
card_text(sh(s, 'Text 11'), 'Chậm hơn', 'M2 và M3 mất khoảng 70 giây mỗi cảnh, rồi khoảng 10 giây mỗi ảnh; mô hình nền khoảng 3 giây (GPU A100).')
card_text(sh(s, 'Text 13'), 'Vị trí chưa bằng GLIGEN, Collage', 'mIoU 0,48 so với 0,65 và 0,61; GLIGEN vẫn tự nhiên hơn khi chấm mù (1,81 so với 1,36).')
card_text(sh(s, 'Text 15'), 'Phạm vi', 'Mới kiểm chứng trên QuickDraw-Scenes và SD 1.5; OCSD cuối chưa chạy lại trên ảnh thật COCO-Sketch.')
card_text(sh(s, 'TextBox 25'), 'Giữ vị trí khi nhả vật', 'M4 nhả vật để hòa vào nền nên vật lệch nhẹ khỏi phác thảo; cần giữ chỗ chặt hơn.')
card_text(sh(s, 'TextBox 27'), 'Dữ liệu, người chấm', 'Thêm ảnh thật, nhiều người chấm độc lập, khảo sát người dùng.')
notes(s, 'Nhóm nêu rõ các hạn chế. Cảnh rất đông: 8+ đối tượng vẫn hay sai số lượng, sai màu, nhầm loài. OCSD chậm hơn: sinh đối tượng và học định danh mất khoảng 70 giây mỗi cảnh, sau đó khoảng 10 giây mỗi ảnh, mô hình nền khoảng 3 giây. '
         'Vị trí theo mIoU của OCSD còn thấp hơn GLIGEN và Collage, và khi chấm mù GLIGEN vẫn tự nhiên hơn. Nhóm mới kiểm chứng trên QuickDraw-Scenes với SD 1.5. '
         'Hướng tiếp theo: xử lý cảnh rất đông, tăng tốc, giữ vị trí vật chặt hơn khi M4 nhả vật, và thêm ảnh thật cùng nhiều người chấm độc lập.')
p.save(OUT); print('final results slides done')
