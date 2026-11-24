"""24-11 deck -> one OCSD only (SD 1.5, all five modules M1-M5).
Removes the SDXL variant (slide 35 and every mention) and puts the results slides back on the thesis run
results/paper_v2 (QuickDraw-Scenes E3/E1/E4). Usage: python single_ocsd.py IN.pptx OUT.pptx (run in this folder)."""
import re, sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

SRC, OUT = sys.argv[1:3]
p = Presentation(SRC)
INK, MUT, ORG, BLU, GRY, PEA = C(0x1F,0x24,0x30), C(0x5B,0x62,0x70), C(0xD9,0x69,0x2B), C(0x2E,0x5A,0x9C), C(0xF3,0xF4,0xF6), C(0xFB,0xE9,0xDD)
SL = list(p.slides)
def S(n): return SL[n - 1]
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
def repic(s, old, path):
    new = s.shapes.add_picture(path, old.left, old.top, old.width, old.height)
    old._element.addprevious(new._element); old._element.getparent().remove(old._element)
def rm(x): x._element.getparent().remove(x._element)
def notes(s, text): s.notes_slide.notes_text_frame.text = text
def box(s, x, y, w, h, fill):
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, I(x), I(y), I(w), I(h)); b.fill.solid(); b.fill.fore_color.rgb = fill
    b.line.fill.background(); b.shadow.inherit = False; return b
def tb(s, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=4):
    t = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in 'margin_left', 'margin_right', 'margin_top', 'margin_bottom': setattr(tf, m, 0)
    for k, line in enumerate(lines):
        para = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); para.alignment = align; para.space_after = Pt(space)
        for txt, sz, col, b in line:
            r = para.add_run(); r.text = txt; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b; r.font.name = 'Arial'
    return t
def replace_runs(s, pairs):
    for t in s._element.iter(qn('a:t')):
        for a, b in pairs:
            if t.text and a in t.text: t.text = t.text.replace(a, b)

# ---- slides 10, 11, 23: no second OCSD, no InstanceDiffusion
replace_runs(S(10), [(' Bản cải tiến của OCSD chạy cùng năm mô-đun trên SDXL.', '')])
replace_runs(S(11), [('Ảnh tĩnh 2D, 512 × 512 điểm ảnh (OCSD trên SDXL: 1024 × 1024).', 'Ảnh tĩnh 2D, kích thước 512 × 512 điểm ảnh.'),
                     ('Stable Diffusion 1.5 và ControlNet Scribble; bản cải tiến của OCSD dùng SDXL. Không huấn luyện lại mô hình lớn.',
                      'Stable Diffusion 1.5 và ControlNet Scribble; không huấn luyện lại mô hình lớn.')])
replace_runs(S(23), [('(ControlNet, GLIGEN, T2I-Adapter, InstanceDiffusion)', '(ControlNet, T2I-Adapter, GLIGEN, Zhang và cs.)')])

# ---- slide 40: by object count
s = S(40)
repic(s, sh(s, 'Picture 11'), 'charts/c_count.png')
settext(sh(s, 'Text 1'), ['Khi số đối tượng tăng, OCSD giảm chậm hơn mô hình nền'])
settext(sh(s, 'Text 4'), ['59% so với 40%'])
settext(sh(s, 'Text 5'), ['Giữ đối tượng ở cảnh 8+ đối tượng: OCSD so với mô hình nền. Từ 1 lên 8+ đối tượng, OCSD giảm 30 điểm, mô hình nền giảm 54 điểm.'])
card_text(sh(s, 'Text 7'), 'Chưa giải quyết xong', 'Ở 8+ đối tượng, GLIGEN (64%) và Ghép ảnh (72%) vẫn giữ được nhiều hơn; mọi phương pháp đều giảm mạnh. Chỉ 9 cảnh 8+ nên đây là phân tích thăm dò.')
notes(s, 'Biểu đồ này cho thấy điều đề tài quan tâm nhất: khi số đối tượng tăng thì chuyện gì xảy ra. Mô hình nền, đường xám nét đứt, giữ 94 phần trăm đối tượng ở cảnh một đối tượng nhưng chỉ còn 40 phần trăm ở cảnh 8 đối tượng trở lên. OCSD, đường cam, giảm chậm hơn, còn 59 phần trăm. Tuy nhiên nhóm nói rõ: GLIGEN và cách ghép ảnh Collage còn giữ được nhiều hơn OCSD ở mức này, và chỉ có 9 cảnh 8+ nên khác biệt chưa có ý nghĩa thống kê. Cảnh rất đông vẫn là bài toán chưa giải xong.')

# ---- slide 41: comparison
s = S(41)
for nm, f in [('Picture 18', 'c_opr'), ('Picture 19', 'c_miou'), ('Picture 20', 'c_ra')]: repic(s, sh(s, nm), f'charts/{f}.png')
card_text(sh(s, 'Text 5'), 'Giữ đối tượng', '74% so với 67% của mô hình nền và 62% của Zhang và cs.; còn thấp hơn GLIGEN (82%) và Ghép ảnh (84%).')
card_text(sh(s, 'Text 7'), 'Vị trí, hình dạng', '0,54 so với 0,51 của mô hình nền; GLIGEN cao nhất (0,67), hơn OCSD có ý nghĩa thống kê.')
card_text(sh(s, 'Text 9'), 'Đúng tương quan', '51% so với 37% của mô hình nền và 35% của Zhang và cs.; Ghép ảnh (66%) và GLIGEN (58%) cao hơn.')
settext(sh(s, 'Text 10'), ['36 cảnh QuickDraw-Scenes × 2 hạt giống, chấm bằng OWLv2. Mọi phương pháp được tinh chỉnh trên cùng một tập tinh chỉnh riêng. Kiểm định Wilcoxon cặp, hiệu chỉnh Holm. Trả lời Q2.'])
notes(s, 'Đây là kết quả chính, 36 cảnh, mỗi cảnh chạy 2 lần, chấm tự động bằng OWLv2. Nhóm so với bốn cách làm: mô hình nền ControlNet, phương pháp gốc của Zhang và cộng sự, GLIGEN dùng hộp chữ nhật, và cách ghép ảnh Collage dùng chính các đối tượng do M2 của OCSD sinh ra. Mọi phương pháp đều được tinh chỉnh công bằng trên cùng một tập riêng. OCSD giữ 74 phần trăm đối tượng, hơn mô hình nền 67 và Zhang 62; đúng tương quan 51 so với 37 phần trăm. Nhưng nhóm báo cáo trung thực: GLIGEN và Ghép ảnh cao hơn OCSD ở cả ba độ đo; riêng vị trí thì GLIGEN hơn có ý nghĩa thống kê. Việc Ghép ảnh dùng đối tượng của M2 mà đạt 84 phần trăm cho thấy nhánh đối tượng là phần mạnh nhất; phần dựng cảnh còn làm mất đối tượng.')

# ---- slide 42: qualitative, 4 columns (drop the second OCSD column)
s = S(42)
col4 = [x for x in s.shapes if abs(x.left / 914400 - 8.75) < 0.01]
ocsd_pics = sorted([x for x in col4 if x.shape_type == 13], key=lambda x: x.top)
col5 = [x for x in s.shapes if abs(x.left / 914400 - 10.75) < 0.01]
for old, new in zip(sorted([x for x in col5 if x.shape_type == 13], key=lambda x: x.top), ocsd_pics):
    new.left = old.left; rm(old)                 # the SD 1.5 OCSD images go into the orange (OCSD) column
rm(sh(s, 'Text 7'))
for x in col4:
    if x.shape_type != 13 and x.name != 'Text 7': rm(x)
for k, L in enumerate([4.75, 6.75, 10.75]):       # spread: sketch 2.75 | baseline | GLIGEN | OCSD
    for x in list(s.shapes):
        if abs(x.left / 914400 - L) < 0.01: x.left = I([5.35, 7.95, 10.55][k])
settext(sh(s, 'Text 27'), ['Ảnh thật (hạt giống 0), cùng cấu hình OCSD (SD 1.5, đủ năm mô-đun) với các bảng kết quả. OCSD cũng có ca sai (phần hạn chế).'])
notes(s, 'Đây là ảnh thật; mỗi hàng là một phác thảo, mỗi cột là một phương pháp. Hàng đầu, con bò xám bên hồ: mô hình nền không vẽ ra con bò, chỉ có một cái bè; GLIGEN vẽ bò nhưng màu đen; OCSD vẽ con bò xám trắng đúng chỗ đã phác thảo, tuy còn hơi mờ. Hàng hai, hai xe buýt cam và một băng ghế ở nhà ga: mô hình nền phủ cả ảnh một mảng cam; GLIGEN có đủ hai xe và băng ghế nhưng không có nhà ga; OCSD có xe buýt, băng ghế và nhà ga, nhưng xe thứ hai nhỏ và thấp. Hàng ba, cảnh rất đông: mọi phương pháp đều sai; OCSD có chó và nhà nhưng chó sai màu và các vật trông như được dán vào. Nhóm trình bày ca sai này ở phần hạn chế.')

# ---- slide 45: answers to Q1-Q3
s = S(45)
keep = {x._element for x in s.shapes if x.name in ('Text 0', 'Text 1', 'Text 2', 'Text 3')}
tree = s.shapes._spTree
for el in list(tree)[2:]:
    if el not in keep: tree.remove(el)
def chip(x, y, big, lab):
    box(s, x, y, 1.38, 1.35, GRY)
    tb(s, x + 0.05, y + 0.12, 1.28, 0.5, [[(big, 22, BLU, True)]], align=PP_ALIGN.CENTER)
    tb(s, x + 0.05, y + 0.66, 1.28, 0.62, [[(l, 10, MUT, False)] for l in lab.split('\n')], align=PP_ALIGN.CENTER, space=0)
def qcirc(y, q):
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, I(0.6), I(y + 0.25), I(0.8), I(0.8)); c.fill.solid(); c.fill.fore_color.rgb = BLU
    c.line.fill.background(); c.shadow.inherit = False
    tb(s, 0.6, y + 0.25, 0.8, 0.8, [[(q, 17, C(255, 255, 255), True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
Q = [('Q1', 'Mô hình nền suy giảm thế nào khi số đối tượng và độ phức tạp phác thảo tăng?',
      'Giảm mạnh khi số đối tượng tăng; độ phức tạp nét vẽ ảnh hưởng ít hơn.',
      [('94%', 'giữ đối tượng\ncảnh 1 đối tượng'), ('63%', 'giữ đối tượng\ncảnh 3 đối tượng'), ('40%', 'giữ đối tượng\ncảnh 8+')]),
     ('Q2', 'Tách thành sinh cấp đối tượng + dựng cảnh với token định danh có giúp nhất quán hơn?',
      'Có so với mô hình nền, nhưng chưa bằng GLIGEN; Ghép ảnh dùng đối tượng của M2 đạt cao nhất.',
      [('74%', 'giữ đối tượng\nOCSD (nền 67%)'), ('82%', 'giữ đối tượng\nGLIGEN'), ('84%', 'giữ đối tượng\nGhép ảnh (M2)')]),
     ('Q3', 'Điều kiện hóa nhận biết đối tượng (định danh, vị trí, mối tương quan) có giảm thêm lỗi mất, sai số lượng, sai mối tương quan?',
      'Đúng tương quan và đếm đúng tăng so với nền; nhưng bỏ riêng M5(a), M5(b) không làm kết quả giảm.',
      [('51%', 'đúng tương quan\n(nền 37%)'), ('35%', 'đếm đúng\n(nền 25%)'), ('81%', 'giữ đối tượng khi\nbỏ M5(b) (đủ: 68%)')])]
for (q, ques, ans, ch), y in zip(Q, [1.5, 3.05, 4.6]):
    qcirc(y, q)
    tb(s, 1.65, y, 6.3, 0.6, [[(ques, 14, MUT, False)]])
    tb(s, 1.65, y + 0.66, 6.3, 0.7, [[(ans, 15, INK, True)]])
    for k, (b, l) in enumerate(ch): chip(8.2 + k * 1.5, y, b, l)
box(s, 0.6, 6.12, 12.1, 0.78, PEA)
tb(s, 0.85, 6.12, 11.6, 0.78, [[('36 cảnh QuickDraw-Scenes × 2 hạt giống, chấm bằng OWLv2; cắt bỏ: 18 cảnh, không thành phần nào khác biệt có ý nghĩa thống kê (Holm p ≥ 0,24). Kết quả âm được báo cáo trung thực.', 11, INK, False)]], anchor=MSO_ANCHOR.MIDDLE)
notes(s, 'Nhóm trả lời ba câu hỏi nghiên cứu. Q1: mô hình nền mất đối tượng rất nhanh khi số đối tượng tăng, từ 94 phần trăm ở cảnh một đối tượng xuống 40 phần trăm ở cảnh 8+; độ phức tạp nét vẽ ảnh hưởng ít hơn. Q2: tách từng đối tượng rồi dựng cảnh giúp OCSD giữ 74 phần trăm so với 67 của mô hình nền, nhưng GLIGEN đã tinh chỉnh đạt 82; cách ghép ảnh dùng chính đối tượng của M2 đạt 84, nghĩa là sinh từng đối tượng là ý đúng, phần dựng cảnh còn làm mất đối tượng. Q3: đúng tương quan và đếm đúng tăng so với mô hình nền, nhưng trong nghiên cứu cắt bỏ, bỏ riêng chú ý theo vùng hay dẫn hướng năng lượng không làm kết quả giảm. Nhóm báo cáo trung thực các kết quả âm này.')

# ---- slide 46: limitations, SD 1.5 OCSD failure case
s = S(46)
repic(s, sh(s, 'Picture 28'), 'crop_8p_sketch.jpg')
repic(s, sh(s, 'Picture 29'), 'crop_8p_ocsd.jpg')
settext(sh(s, 'Text 6'), ['Phác thảo 8+ đối tượng'])
settext(sh(s, 'Text 7'), ['OCSD: chó sai màu, vật như dán vào'])
card_text(sh(s, 'Text 9'), 'Cảnh rất đông', '8+ đối tượng vẫn hay sai số lượng, sai màu, nhầm loài.')
card_text(sh(s, 'Text 11'), 'Chậm hơn', 'M2 và M3 mất khoảng 48 giây mỗi cảnh, rồi khoảng 5 giây mỗi ảnh; mô hình nền khoảng 2 giây.')
card_text(sh(s, 'Text 13'), 'Chưa hơn GLIGEN, Ghép ảnh', 'Giữ đối tượng 74% so với 82% và 84%; M5(a), M5(b), M3 chưa cho lợi ích rõ khi cắt bỏ.')
card_text(sh(s, 'Text 15'), 'Phạm vi', 'Ảnh thật COCO-Sketch: 46% so với 54% của GLIGEN; mới thử trên SD 1.5.')
card_text(sh(s, 'TextBox 25'), 'Dựng cảnh tốt hơn', 'Giữ được đối tượng của M2 khi dựng cảnh; làm M5 thực sự có ích.')
card_text(sh(s, 'TextBox 27'), 'Dữ liệu', 'Thêm ảnh thật và người chấm độc lập, khảo sát người dùng.')
notes(s, 'Nhóm nêu rõ các hạn chế. Cảnh rất đông: ví dụ phác thảo 8+ đối tượng này, OCSD vẽ chó sai màu và các vật trông như được dán vào. OCSD chậm hơn: sinh đối tượng và học định danh mất khoảng 48 giây mỗi cảnh, sau đó khoảng 5 giây mỗi ảnh, trong khi mô hình nền chỉ khoảng 2 giây. Khi các phương pháp so sánh được tinh chỉnh công bằng, OCSD chưa hơn GLIGEN và Ghép ảnh, và các thành phần M5(a), M5(b), M3 chưa cho lợi ích rõ trong nghiên cứu cắt bỏ. Trên ảnh thật COCO-Sketch, OCSD đạt 46 phần trăm so với 54 của GLIGEN. Hướng tiếp theo: xử lý cảnh rất đông, tăng tốc, làm phần dựng cảnh giữ được đối tượng của M2, và thêm dữ liệu ảnh thật cùng người chấm độc lập.')

# ---- slide 35 (SDXL method slide): delete
lst = p.slides._sldIdLst; el = list(lst)[34]
rid = el.get(qn('r:id')); p.part.drop_rel(rid); lst.remove(el)

# ---- renumber
tot = len(p.slides)
for i, s in enumerate(p.slides, 1):
    for t in s._element.iter(qn('a:t')):
        if t.text and re.fullmatch(r'\s*\d+ / \d+\s*', t.text): t.text = f'{i} / {tot}'
for i, s in enumerate(p.slides, 1):
    txt = ' '.join(t.text or '' for t in s._element.iter(qn('a:t')))
    n = s.notes_slide.notes_text_frame.text if s.has_notes_slide else ''
    for kw in ('SDXL', 'XL', 'gốc', 'InstanceDiffusion', 'bám nét', '1,08', 'tự nhiên /2'):
        if kw in txt or kw in n: print('CHECK', i, kw)
p.save(OUT)
print('slides', tot)
