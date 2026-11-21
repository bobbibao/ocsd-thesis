"""Diffusion explainer in 2 slides (replaces slides 14-20). No editable equations: formulas are images.
Usage (in this folder): python build_diffusion2.py IN.pptx OUT.pptx   (figures: python figs2.py <deck media> figs)"""
import re, sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
SRC, OUT = sys.argv[1:3]
F = 'figs'
p = Presentation(SRC)
INK, MUT, ORG, BLU = C(0x1F,0x24,0x30), C(0x5B,0x62,0x70), C(0xD9,0x69,0x2B), C(0x2E,0x5A,0x9C)
GRY, PEA, LBL, WHT, LINE = C(0xF3,0xF4,0xF6), C(0xFB,0xE9,0xDD), C(0xE3,0xEC,0xF7), C(255,255,255), C(0xD5,0xD8,0xDE)
SL = list(p.slides)
A, B = SL[13], SL[14]
assert A.shapes[1].text_frame.text.startswith('Ý tưởng') or 'Ý tưởng' in ''.join(x.text_frame.text for x in A.shapes if x.has_text_frame)
def clear(s):
    keep = {x._element for x in s.shapes if x.name in ('Text 0', 'Text 1', 'TextBox 32', 'Text 3')}
    tree = s.shapes._spTree
    for el in list(tree)[2:]:
        if el not in keep: tree.remove(el)
    # drop any mc:AlternateContent leftovers (old OMML text boxes)
    for el in list(tree):
        if el.tag.endswith('AlternateContent'): tree.remove(el)
def title(s, t):
    x = [x for x in s.shapes if x.name == 'Text 1'][0]; r = x.text_frame.paragraphs[0].runs
    r[0].text = t
    for q in r[1:]: q.text = ''
def box(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    b = s.shapes.add_shape(shape, I(x), I(y), I(w), I(h)); b.fill.solid(); b.fill.fore_color.rgb = fill
    if line is None: b.line.fill.background()
    else: b.line.color.rgb = line; b.line.width = Pt(1.25)
    b.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: b.adjustments[0] = 0.08
    return b
def tb(s, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=3):
    t = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in 'margin_left', 'margin_right', 'margin_top', 'margin_bottom': setattr(tf, m, 0)
    for k, line in enumerate(lines):
        para = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); para.alignment = align; para.space_after = Pt(space)
        for seg in line:
            txt, sz, col, b = seg[:4]; it = seg[4] if len(seg) > 4 else False
            r = para.add_run(); r.text = txt; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b; r.font.italic = it; r.font.name = 'Arial'
    return t
def pic(s, f, x, y, w=None, h=None, frame=False):
    q = s.shapes.add_picture(f'{F}/{f}', I(x), I(y), I(w) if w else None, I(h) if h else None)
    if frame: q.line.color.rgb = LINE; q.line.width = Pt(0.75)
    return q
def arrow(s, x, y, w):
    a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, I(x), I(y), I(w), I(0.28)); a.fill.solid(); a.fill.fore_color.rgb = ORG
    a.line.fill.background(); a.shadow.inherit = False
def strip(s, files, labels, head, y=1.72, sz=1.85, gap=0.2):
    tb(s, 0.6, y - 0.36, 12.1, 0.3, [[head]])
    for k, (f, lab) in enumerate(zip(files, labels)):
        x = 0.6 + k * (sz + gap)
        pic(s, f, x, y, sz, sz, frame=True)
        tb(s, x, y + sz + 0.06, sz, 0.5, [[(l, 10.5 if j else 11.5, INK if j == 0 else MUT, j == 0)] for j, l in enumerate(lab.split('\n'))],
           align=PP_ALIGN.CENTER, space=0)

# ------------------------------------------------------------------ slide A: forward + training
s = A; clear(s)
title(s, 'Mô hình khuếch tán: thêm nhiễu, rồi học cách xóa')
strip(s, [f'a_fwd_{t}.png' for t in (0, 100, 300, 500, 800, 1000)],
      ['t = 0\nảnh gốc', 't = 100\nảnh × 0,95 + nhiễu × 0,32', 't = 300\nảnh × 0,63 + nhiễu × 0,78',
       't = 500\nảnh × 0,28 + nhiễu × 0,96', 't = 800\nảnh × 0,04 + nhiễu × 1,00', 't = 1000\nchỉ còn nhiễu'],
      ('Quá trình thuận: trộn thêm nhiễu ngẫu nhiên qua 1000 bước, ảnh mờ dần thành nhiễu (không cần học, chỉ là phép cộng)', 13, ORG, True))
# training (left)
tb(s, 0.6, 4.42, 7.6, 0.3, [[('Huấn luyện: dạy mạng U-Net đoán phần nhiễu đã trộn vào', 14, INK, True)]])
y0, z = 4.82, 1.2
pic(s, 'a_x0.png', 0.6, y0, z, z, True); tb(s, 1.85, y0 + 0.4, 0.3, 0.4, [[('+', 20, MUT, True)]], align=PP_ALIGN.CENTER)
pic(s, 'a_eps.png', 2.2, y0, z, z, True); tb(s, 3.45, y0 + 0.4, 0.3, 0.4, [[('=', 20, MUT, True)]], align=PP_ALIGN.CENTER)
pic(s, 'a_x300.png', 3.8, y0, z, z, True)
for x, l in [(0.6, 'ảnh thật'), (2.2, 'nhiễu (biết trước)'), (3.8, 'ảnh nhiễu, t = 300')]:
    tb(s, x - 0.1, y0 + z + 0.04, z + 0.2, 0.25, [[(l, 10, MUT, False)]], align=PP_ALIGN.CENTER)
arrow(s, 5.08, y0 + 0.46, 0.38)
u = box(s, 5.52, y0 + 0.1, 1.1, 1.0, LBL, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
tb(s, 5.52, y0 + 0.1, 1.1, 1.0, [[('U-Net', 14, BLU, True)], [('mạng nơ-ron', 9.5, MUT, False)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0)
arrow(s, 6.68, y0 + 0.46, 0.38)
tb(s, 7.12, y0 + 0.12, 1.1, 1.0, [[('đoán:', 10.5, MUT, False)], [('nhiễu nào đã', 11, INK, True)], [('trộn vào?', 11, INK, True)]],
   anchor=MSO_ANCHOR.MIDDLE, space=0)
pic(s, 'eq_loss.png', 0.6, 6.42, h=0.3)
tb(s, 2.55, 6.42, 5.7, 0.4, [[('So nhiễu đoán với nhiễu thật, sai bao nhiêu thì sửa trọng số; lặp lại với rất nhiều ảnh và mọi bước t.', 10.5, MUT, False)]],
   anchor=MSO_ANCHOR.MIDDLE)
# one-pixel example (right)
box(s, 8.5, 4.42, 4.2, 2.45, PEA, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
tb(s, 8.72, 4.55, 3.8, 1.75, [[('Ví dụ với một điểm ảnh', 14, ORG, True)],
                               [('Điểm ảnh có độ sáng ', 11.5, INK, False), ('0,8', 11.5, INK, True), ('; rút ngẫu nhiên một nhiễu ', 11.5, INK, False), ('0,5', 11.5, INK, True), ('.', 11.5, INK, False)],
                               [('Ở bước t = 300: 0,63 × 0,8 + 0,78 × 0,5 ≈ ', 11.5, INK, False), ('0,89', 11.5, INK, True)],
                               [('Mạng chỉ thấy 0,89 (và biết t = 300), phải đoán ra nhiễu 0,5.', 11.5, INK, False)]], space=5)
pic(s, 'eq_fwd.png', 8.72, 6.35, h=0.36)
A.notes_slide.notes_text_frame.text = (
    'Mô hình khuếch tán dựa trên một ý rất đơn giản. Hàng ảnh trên là quá trình thuận: lấy ảnh con bò, mỗi bước trộn thêm một chút nhiễu ngẫu nhiên. '
    'Sau 1000 bước ảnh chỉ còn là nhiễu. Bước này không cần học, chỉ là phép cộng có hệ số: ví dụ ở bước 300, ảnh nhân 0,63 cộng nhiễu nhân 0,78. '
    'Điều mô hình phải học là chiều ngược lại. Khi huấn luyện, ta lấy ảnh thật, rút một nhiễu ngẫu nhiên, trộn lại thành ảnh nhiễu ở một bước t bất kỳ, '
    'rồi đưa cho mạng U-Net và hỏi: nhiễu nào đã được trộn vào? Vì ta biết nhiễu thật nên tính được mạng sai bao nhiêu và sửa trọng số. '
    'Ví dụ một điểm ảnh sáng 0,8, nhiễu rút được là 0,5; ở bước 300 điểm ảnh thành 0,63 nhân 0,8 cộng 0,78 nhân 0,5, khoảng 0,89. Mạng chỉ thấy 0,89 và phải đoán ra 0,5. '
    'Lặp lại với rất nhiều ảnh, mạng học được ảnh thật trông như thế nào.')

# ------------------------------------------------------------------ slide B: sampling
s = B; clear(s)
title(s, 'Sinh ảnh: bắt đầu từ nhiễu, xóa dần từng bước')
strip(s, [f'b_x_{t}.png' for t in (1000, 800, 600, 400, 200, 0)],
      ['t = 1000\nnhiễu ngẫu nhiên', 't = 800', 't = 600', 't = 400', 't = 200', 't = 0\nảnh mới'],
      ('Quá trình ngược: lặp lại “đoán nhiễu rồi trừ bớt”, từ nhiễu thuần đến ảnh rõ nét', 13, ORG, True))
tb(s, 0.6, 4.22, 12.1, 0.2, [[('Dãy ảnh tính thật bằng công thức DDPM (T = 1000). Để chạy được trên máy thường, bộ đoán nhiễu ở đây là bộ đoán lý tưởng trên 12 ảnh mẫu thay cho U-Net đã huấn luyện.', 9, MUT, False, True)]])
box(s, 0.6, 4.55, 6.6, 2.32, GRY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
tb(s, 0.82, 4.67, 6.2, 1.6, [[('Một bước làm gì? (tiếp ví dụ điểm ảnh)', 14, INK, True)],
                              [('1. Mạng nhìn thấy 0,89 ở bước 300 và đoán nhiễu ≈ 0,5.', 11.5, INK, False)],
                              [('2. Suy ra ảnh sạch: (0,89 − 0,78 × 0,5) / 0,63 ≈ ', 11.5, INK, False), ('0,8', 11.5, INK, True)],
                              [('3. Chỉ đi một bước nhỏ về phía 0,8 (sang bước 299), rồi lặp lại đến t = 0.', 11.5, INK, False)]], space=5)
pic(s, 'eq_x0.png', 0.82, 6.05, h=0.34)
box(s, 7.45, 4.55, 5.25, 2.32, LBL, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
tb(s, 7.67, 4.67, 4.85, 2.1, [[('Stable Diffusion dùng đúng cách này, thêm:', 14, BLU, True)],
                               [('• Câu lệnh: ', 11.5, INK, True), ('mạng đoán nhiễu theo câu mô tả, nên ảnh cuối khớp câu.', 11.5, INK, False)],
                               [('• Ảnh nén: ', 11.5, INK, True), ('khử nhiễu trên ảnh nén 64 × 64, nhanh hơn nhiều.', 11.5, INK, False)],
                               [('• DDIM: ', 11.5, INK, True), ('chỉ cần khoảng 30 bước thay vì 1000.', 11.5, INK, False)],
                               [('• Nhiễu ban đầu khác (hạt giống khác) → ảnh khác.', 11.5, INK, False)]], space=5)
B.notes_slide.notes_text_frame.text = (
    'Khi đã có mạng biết đoán nhiễu, sinh ảnh mới là đi ngược lại. Bắt đầu từ một nhiễu hoàn toàn ngẫu nhiên ở bước 1000. '
    'Ở mỗi bước, mạng đoán phần nhiễu, từ đó suy ra ảnh sạch trông thế nào, rồi chỉ đi một bước nhỏ về phía đó. Lặp lại đến bước 0 thì được ảnh rõ nét. '
    'Tiếp ví dụ điểm ảnh: mạng thấy 0,89 ở bước 300, đoán nhiễu khoảng 0,5, suy ra ảnh sạch là 0,89 trừ 0,78 nhân 0,5, chia 0,63, đúng bằng 0,8. '
    'Dãy ảnh trên được tính thật bằng công thức DDPM; để chạy trên máy thường, nhóm dùng bộ đoán lý tưởng trên 12 ảnh thay cho U-Net, nên ảnh cuối là một ảnh trong tập. '
    'Stable Diffusion dùng đúng cách này nhưng thêm ba điều: mạng đoán nhiễu theo câu lệnh, khử nhiễu trên ảnh nén nhỏ nên nhanh, và dùng DDIM chỉ khoảng 30 bước. '
    'Bây giờ quay lại slide 13 để nói về Stable Diffusion.')

# ------------------------------------------------------------------ delete old explainer slides 16-20
lst = p.slides._sldIdLst; ids = list(lst)
for el in ids[15:20]:
    p.part.drop_rel(el.get(qn('r:id'))); lst.remove(el)
tot = len(p.slides)
for i, s in enumerate(p.slides, 1):
    for t in s._element.iter(qn('a:t')):
        if t.text and re.fullmatch(r'\s*\d+ / \d+\s*', t.text): t.text = f'{i} / {tot}'
# the repeated slide 13 tag "Quay lại slide 13" stays correct (slide 13 unchanged)
p.save(OUT); print('slides', tot)
