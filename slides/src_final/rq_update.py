"""Chain step 13: requirements, objectives and research questions approved by Bobby 2026-11-25
(/mnt/project-files/thesis/RQ_FINAL_v14.md). Rewrites slide 'Yêu cầu, mục tiêu và câu hỏi nghiên cứu', the answers
slide and the agenda/section subtitles. Usage (run here): rq_update.py IN OUT"""
import sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
p = Presentation(sys.argv[1])
INK, MUT, ORG, BLU, GRY = C(0x1F,0x24,0x30), C(0x5B,0x62,0x70), C(0xD9,0x69,0x2B), C(0x2E,0x5A,0x9C), C(0xF3,0xF4,0xF6)
Q = ['Khi tách riêng hai yếu tố trên một bộ kiểm thử có kiểm soát, yếu tố nào (số lượng vật thể hay độ phức tạp của phác thảo) thực sự làm mô hình nền vi phạm các yêu cầu R1 – R4, và lỗi chủ yếu thuộc dạng nào?',
     'Tách cảnh thành các vật thể có định danh riêng có thu hẹp được khoảng cách về R1, R2 và R4 của các phương pháp chỉ điều khiển bằng nét vẽ không; và so với các phương pháp dùng hộp giới hạn hoặc ảnh vật thể, đánh đổi giữa cấu trúc và độ tự nhiên diễn ra như thế nào?',
     'Hợp thành cảnh hài hòa (một quỹ đạo, neo mềm, giải phóng dần, ControlNet cấp cảnh) có giữ được nhiều vật thể hơn và đồng thời tự nhiên hơn so với cách trộn cứng hai nhánh hay không?']
MT = [('MT1 · Xác định nguyên nhân suy giảm', 'E1–E2'), ('MT2 · Đề xuất phương pháp và so sánh', 'E3'), ('MT3 · Hợp thành cảnh hài hòa', 'E4')]
SUB = 'Bốn yêu cầu R1–R4, ba mục tiêu MT1–MT3 và câu hỏi Q1–Q3'

def alltext(s): return ' '.join(t.text or '' for t in s._element.iter(qn('a:t')))
def find(marker): return next(s for s in p.slides if marker in alltext(s))
def sh(s, name): return [x for x in s.shapes if x.name == name][0]
def runs(shape, paras):
    """paras = [[(text, size, color, bold), ...], ...]: rewrite a text frame keeping its box."""
    tf = shape.text_frame
    for para in list(tf.paragraphs)[1:]: para._p.getparent().remove(para._p)
    for r in list(tf.paragraphs[0].runs): r._r.getparent().remove(r._r)
    for k, segs in enumerate(paras):
        para = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); para.space_after = Pt(2)
        for t, sz, col, b in segs:
            r = para.add_run(); r.text = t; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b; r.font.name = 'Arial'
def tb(s, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    t = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in 'margin_left', 'margin_right', 'margin_top', 'margin_bottom': setattr(tf, m, 0)
    for k, segs in enumerate(paras):
        para = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); para.alignment = align; para.space_after = Pt(3)
        for txt, sz, col, b, *it in segs:
            r = para.add_run(); r.text = txt; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b; r.font.name = 'Arial'
            r.font.italic = bool(it and it[0])
    return t
def move(x, l=None, t=None, w=None, h=None):
    for k, v in (('left', l), ('top', t), ('width', w), ('height', h)):
        if v is not None: setattr(x, k, I(v))

# ---------------------------------------------------------------- agenda + section divider
for s in p.slides:
    for x in s.shapes:
        if x.has_text_frame and x.text_frame.text.strip() == 'Bốn yêu cầu nhất quán R1–R4, ba câu hỏi Q1–Q3':
            r = x.text_frame.paragraphs[0].runs; r[0].text = SUB
            for q in r[1:]: q.text = ''
    if s.has_notes_slide:
        tf = s.notes_slide.notes_text_frame
        if 'Bốn yêu cầu nhất quán R1–R4, ba câu hỏi Q1–Q3' in tf.text: tf.text = tf.text.replace('Bốn yêu cầu nhất quán R1–R4, ba câu hỏi Q1–Q3', SUB)

# ---------------------------------------------------------------- slide: requirements, objectives, questions
s = next(x for x in p.slides if any(y.name == 'Text 1' and y.has_text_frame and y.text_frame.text.strip() in ('4 Yêu cầu và 3 câu hỏi nghiên cứu', 'Yêu cầu, mục tiêu và câu hỏi nghiên cứu') for y in x.shapes))
runs(sh(s, 'Text 1'), [[('Yêu cầu, mục tiêu và câu hỏi nghiên cứu', 30, INK, True)]])
R = [('Đúng vật thể', 'đúng lớp, không hòa trộn', 'OPR'),
     ('Đúng số lượng', 'không thừa, không thiếu theo từng loại', 'OCE theo loại, độ chính xác số lượng'),
     ('Đúng vị trí và hình dạng', 'vùng vật thể khớp vùng phác thảo', 'mIoU, chamfer'),
     ('Đúng quan hệ', 'trái/phải, trên/dưới, trước/sau', 'RA')]
for k, (box, tbx, num) in enumerate((('Shape 4', 'Text 6', 'Text 5'), ('Shape 7', 'Text 9', 'Text 8'),
                                     ('Shape 10', 'Text 12', 'Text 11'), ('Shape 13', 'Text 15', 'Text 14'))):
    y = 1.6 + k * 1.03
    move(sh(s, box), t=y, h=0.92); move(sh(s, num), t=y, h=0.92); move(sh(s, tbx), t=y + 0.07, h=0.8)
    a, b, m = R[k]
    runs(sh(s, tbx), [[(a, 15, INK, True)], [(b, 11.5, MUT, False)], [('đo: ', 11, ORG, True), (m, 11, ORG, False)]])
c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(0.6), I(5.78), I(5.6), I(0.92)); c.fill.solid(); c.fill.fore_color.rgb = C(0xFB, 0xE9, 0xDD)
c.line.fill.background(); c.shadow.inherit = False; c.adjustments[0] = 0.1
tb(s, 0.8, 5.85, 5.3, 0.8, [[('Ràng buộc: ', 13, ORG, True), ('giữ đúng vật thể nhưng ảnh không được kém tự nhiên hơn', 12, INK, False)],
                           [('đo: ', 11, ORG, True), ('PickScore, chấm mù (khớp câu: CLIP score)', 11, ORG, False)]], anchor=MSO_ANCHOR.MIDDLE)
for k, (circ, lab, qtx) in enumerate((('Shape 16', 'Text 17', 'Text 18'), ('Shape 19', 'Text 20', 'Text 21'), ('Shape 22', 'Text 23', 'Text 24'))):
    y = 1.6 + k * 1.6
    move(sh(s, circ), t=y + 0.05, w=0.7, h=0.7); move(sh(s, lab), t=y + 0.05, w=0.7, h=0.7)
    move(sh(s, qtx), l=7.7, t=y, w=5.0, h=1.5)
    h, e = MT[k]
    runs(sh(s, qtx), [[(h, 12, ORG, True), ('   → thí nghiệm ' + e, 10.5, MUT, False)], [(Q[k], 11.5, INK, False)]])
    sh(s, qtx).text_frame.vertical_anchor = MSO_ANCHOR.TOP
move(sh(s, 'Text 27'), t=6.45, h=0.4)
s.notes_slide.notes_text_frame.text = (
    'Nhóm định nghĩa một ảnh là nhất quán vật thể khi thỏa đồng thời bốn yêu cầu, mỗi yêu cầu có một độ đo. R1 đúng vật thể, đúng lớp và không hòa trộn, đo bằng OPR; '
    'thuộc tính như màu được đưa vào câu lệnh nhưng không tính vào R1 vì chưa có cách đo tự động đáng tin. R2 đúng số lượng theo từng loại, đo bằng sai số số lượng theo loại và độ chính xác số lượng. '
    'R3 đúng vị trí và hình dạng, đo bằng mIoU và chamfer. R4 đúng quan hệ, đo bằng RA; RA được báo cáo riêng vì IoU không phát hiện được hai vật đổi chỗ hay thứ tự trước sau. '
    'Kèm một ràng buộc: giữ vật thể nhưng ảnh không được kém tự nhiên hơn, đo bằng PickScore và chấm mù. '
    'Từ đó có ba mục tiêu, mỗi mục tiêu một câu hỏi và một nhóm thí nghiệm. MT1 xác định nguyên nhân suy giảm, câu hỏi Q1, thí nghiệm E1 và E2. '
    'MT2 đề xuất phương pháp và so sánh, câu hỏi Q2, thí nghiệm E3. MT3 hợp thành cảnh hài hòa, câu hỏi Q3, thí nghiệm E4.')

# ---------------------------------------------------------------- slide: answers
s = find('Trả lời ba câu hỏi nghiên cứu')
A = ['Số lượng vật thể: giữ vật thể 97% → 39%, sai số lượng mỗi loại 0,19 → 5,97; độ phức tạp phác thảo chỉ làm giảm khoảng 8 điểm. Lỗi chủ yếu: mất vật thể, sai số lượng.',
     'Có, so với các phương pháp chỉ dùng nét vẽ. So với GLIGEN, Collage: giữ vật thể ngang, vị trí (mIoU) thấp hơn, tự nhiên hơn theo PickScore; chấm mù hơn Collage, kém GLIGEN.',
     'Có: giữ vật thể 71% → 79% (Holm p = 0,011), PickScore +0,73 (p < 0,0001), chấm mù 0,44 → 1,36; mIoU và RA không giảm có ý nghĩa.']
for k, (qn_, an) in enumerate((('TextBox 7', 'TextBox 8'), ('TextBox 20', 'TextBox 21'), ('TextBox 33', 'TextBox 34'))):
    y = 1.5 + k * 1.55
    move(sh(s, qn_), t=y - 0.07, h=0.8); move(sh(s, an), t=y + 0.8, h=0.65)
    runs(sh(s, qn_), [[(MT[k][0].split(' · ')[1] + ': ', 10, ORG, True), (Q[k], 9.5, MUT, False)]])
    runs(sh(s, an), [[(A[k], 11.5, INK, True)]])
CH = [[('39%', 'giữ vật thể 8+', '(1 vật: 97%)'), ('5,97', 'sai số lượng/loại', '8+ (1 vật: 0,19)'), ('−8', 'điểm OPR khi phác', 'thảo phức tạp hơn')],
      [('79%', 'giữ vật thể', '(chỉ nét: 62–66%)'), ('51%', 'đúng quan hệ', '(chỉ nét: 28–34%)'), ('21,89', 'PickScore (GLIGEN', '21,41; Collage 21,52)')],
      [('79%', 'giữ vật thể', '(M4 trộn cứng: 71%)'), ('+0,73', 'PickScore,', 'hơn ở 65/72 cảnh'), ('1,36', 'chấm mù 0–2', '(trộn cứng: 0,44)')]]
BIG = [('TextBox 10', 'TextBox 11'), ('TextBox 13', 'TextBox 14'), ('TextBox 16', 'TextBox 17'),
       ('TextBox 23', 'TextBox 24'), ('TextBox 26', 'TextBox 27'), ('TextBox 29', 'TextBox 30'),
       ('TextBox 36', 'TextBox 37'), ('TextBox 39', 'TextBox 40'), ('TextBox 42', 'TextBox 43')]
for k, (bn, sn) in enumerate(BIG):
    v, l1, l2 = CH[k // 3][k % 3]
    col = sh(s, bn).text_frame.paragraphs[0].runs[0].font.color.rgb
    runs(sh(s, bn), [[(v, 22, col, True)]])
    runs(sh(s, sn), [[(l1, 9.5, MUT, False)], [(l2, 9.5, MUT, False)]])
    for para in list(sh(s, bn).text_frame.paragraphs) + list(sh(s, sn).text_frame.paragraphs): para.alignment = PP_ALIGN.CENTER
runs(sh(s, 'TextBox 45'), [[('72 cảnh QuickDraw-Scenes × 2 hạt giống, chấm bằng OWLv2; kiểm định dấu cặp, hiệu chỉnh Holm. Chấm mù 0–2: 36 phác thảo, một người chấm. '
                             'Hạn chế: trên tập đánh giá chỉ so M4 hài hòa với M4 trộn cứng; M3 và M5(a, b, d) mới chỉ quan sát trên tập phát triển.', 10, MUT, False)]])
s.notes_slide.notes_text_frame.text = (
    'Nhóm trả lời ba câu hỏi. Q1, mục tiêu xác định nguyên nhân: khi tách riêng hai yếu tố trên bộ kiểm thử có kiểm soát, số lượng vật thể mới là nguyên nhân chính. '
    'Mô hình nền giữ 97 phần trăm vật thể ở cảnh một vật nhưng chỉ còn 39 phần trăm ở cảnh 8+, sai số lượng mỗi loại tăng từ 0,19 lên 5,97; còn độ phức tạp phác thảo chỉ làm giảm khoảng 8 điểm, từ 70 xuống 62 phần trăm. '
    'Điều này sửa lại kết luận của nghiên cứu sơ bộ trên FS-COCO, vốn chưa tách hai yếu tố. Lỗi chủ yếu là mất vật thể và sai số lượng. '
    'Q2, mục tiêu đề xuất và so sánh: có, OCSD thu hẹp khoảng cách so với các phương pháp chỉ dùng nét vẽ, giữ 79 phần trăm vật thể so với 62 đến 66, đúng quan hệ 51 so với 28 đến 34 phần trăm, có ý nghĩa thống kê. '
    'So với GLIGEN và Collage: giữ vật thể ngang nhau, vị trí theo mIoU thấp hơn có ý nghĩa, nhưng tự nhiên hơn theo PickScore; khi chấm mù OCSD hơn Collage và kém GLIGEN, 1,36 so với 1,81. OCSD đánh đổi độ chính xác vị trí lấy độ tự nhiên; nhóm không nói OCSD hơn GLIGEN hay Collage về cấu trúc. '
    'Q3, mục tiêu hợp thành cảnh hài hòa: có, cả hai giả thuyết đăng ký trước đều đúng. Giữ vật thể tăng từ 71 lên 79 phần trăm, p Holm 0,011; PickScore tăng 0,73, tốt hơn ở 65 trên 72 cảnh; chấm mù từ 0,44 lên 1,36; mIoU và RA không giảm có ý nghĩa. '
    'Hạn chế: trên tập đánh giá nhóm chỉ so M4 hài hòa với M4 trộn cứng; M3 và M5 phần a, b, d mới chỉ được quan sát trên tập phát triển.')
p.save(sys.argv[2]); print('ok')
