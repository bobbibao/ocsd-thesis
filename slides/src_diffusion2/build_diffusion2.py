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

# ------------------------------------------------------------------ slide A: forward + training (pictures only)
def lab(s, x, y, w, text, col=MUT, sz=10, b=False):
    tb(s, x, y, w, 0.3, [[(text, sz, col, b)]], align=PP_ALIGN.CENTER)
def op(s, x, y, text):
    tb(s, x, y, 0.5, 0.5, [[(text, 11, ORG, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
s = A; clear(s)
title(s, 'Mô hình khuếch tán: thêm nhiễu, rồi học cách xóa')
strip(s, [f'a_fwd_{t}.png' for t in (0, 100, 300, 500, 800, 1000)],
      ['t = 0\nảnh gốc', 't = 100\nthêm một ít nhiễu', 't = 300\ncon bò mờ đi', 't = 500\ngần như không thấy', 't = 800\nchỉ còn vệt mờ', 't = 1000\nnhiễu hoàn toàn'],
      ('Quá trình thuận: mỗi bước rắc thêm một ít nhiễu, sau 1000 bước ảnh thành nhiễu (không cần học)', 13, ORG, True))
tb(s, 0.6, 4.4, 7.8, 0.3, [[('Huấn luyện = cho mạng làm bài tập “tìm nhiễu”', 14, INK, True)]])
y0, z = 4.8, 1.35
pic(s, 'c_tr0_noisy.png', 0.6, y0, z, z, True); lab(s, 0.5, y0 + z + 0.03, z + 0.2, 'đề bài: ảnh bị nhiễu')
arrow(s, 2.05, y0 + 0.53, 0.35)
box(s, 2.5, y0 + 0.2, 1.0, 0.95, LBL, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
tb(s, 2.5, y0 + 0.2, 1.0, 0.95, [[('U-Net', 14, BLU, True)], [('mạng nơ-ron', 9, MUT, False)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0)
arrow(s, 3.6, y0 + 0.53, 0.35)
pic(s, 'c_tr0_noise.png', 4.05, y0, z, z, True); lab(s, 3.95, y0 + z + 0.03, z + 0.2, 'mạng chỉ ra: đây là nhiễu')
op(s, 5.45, y0 + 0.42, 'bỏ đi')
pic(s, 'c_tr0_clean.png', 5.95, y0, z, z, True); lab(s, 5.85, y0 + z + 0.03, z + 0.2, 'còn lại: ảnh gốc ✓', INK, 10, True)
tb(s, 7.45, y0 + 0.15, 0.95, 1.1, [[('Sai thì sửa mạng, làm lại với ảnh khác.', 9.5, MUT, False)]], anchor=MSO_ANCHOR.MIDDLE)
box(s, 8.6, 4.4, 4.1, 2.5, PEA, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
tb(s, 8.8, 4.5, 3.7, 0.3, [[('Luyện ở mọi mức nhiễu', 14, ORG, True)]])
for k, (f, l) in enumerate([('c_tr0_noisy.png', 'nhẹ'), ('c_tr1_noisy.png', 'vừa'), ('c_tr2_noisy.png', 'nặng')]):
    x = 8.8 + k * 1.25; pic(s, f, x, 4.9, 1.1, 1.1, True); lab(s, x, 6.02, 1.1, l, INK, 10, True)
tb(s, 8.8, 6.33, 3.7, 0.5, [[('Học từ rất nhiều ảnh, mạng dần “biết” ảnh thật trông thế nào.', 10.5, INK, False)]])
A.notes_slide.notes_text_frame.text = (
    'Mô hình khuếch tán dựa trên một ý rất đơn giản. Hàng ảnh trên: lấy ảnh con bò, mỗi bước rắc thêm một ít nhiễu, giống như ảnh bị phủ dần bởi hạt nhiễu. '
    'Sau 1000 bước ảnh chỉ còn là nhiễu. Bước này không cần học. '
    'Điều mô hình phải học là chiều ngược lại. Khi huấn luyện, ta cho mạng U-Net làm bài tập tìm nhiễu: đưa một ảnh bị nhiễu, mạng phải chỉ ra phần nào là nhiễu. '
    'Vì chính ta đã thêm nhiễu nên ta biết đáp án; bỏ phần nhiễu đi thì phải còn lại đúng ảnh gốc. Sai thì sửa mạng, rồi làm lại với ảnh khác. '
    'Bài tập có đủ mức nhiễu, từ nhẹ đến rất nặng. Luyện với rất nhiều ảnh, mạng dần biết ảnh thật trông như thế nào.')

# ------------------------------------------------------------------ slide B: sampling (pictures only)
s = B; clear(s)
title(s, 'Sinh ảnh: bắt đầu từ nhiễu, xóa dần từng bước')
strip(s, [f'b_x_{t}.png' for t in (1000, 800, 600, 400, 200, 0)],
      ['t = 1000\nnhiễu ngẫu nhiên', 't = 800\nchưa thấy gì', 't = 600\nbắt đầu có hình', 't = 400\nthấy dáng con bò', 't = 200\nrõ dần', 't = 0\nảnh mới'],
      ('Quá trình ngược: lặp lại “tìm nhiễu rồi xóa bớt một ít”, từ nhiễu thuần đến ảnh rõ nét', 13, ORG, True))
tb(s, 0.6, 4.2, 12.1, 0.2, [[('Ảnh tính thật theo DDPM (1000 bước); để chạy được trên máy thường, bộ đoán nhiễu ở đây là bộ đoán lý tưởng trên 12 ảnh mẫu thay cho U-Net đã huấn luyện.', 9, MUT, False, True)]])
box(s, 0.6, 4.5, 6.7, 2.4, GRY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
tb(s, 0.8, 4.6, 6.3, 0.3, [[('Một bước làm gì?', 14, INK, True)]])
y0, z = 5.0, 1.3
pic(s, 'c_step_xa.png', 0.8, y0, z, z, True); lab(s, 0.7, y0 + z + 0.03, z + 0.2, 'ảnh đang nhiễu')
arrow(s, 2.2, y0 + 0.5, 0.4)
pic(s, 'c_step_guess.png', 2.7, y0, z, z, True); lab(s, 2.5, y0 + z + 0.03, z + 0.4, 'mạng đoán: ảnh sạch như này')
arrow(s, 4.1, y0 + 0.5, 0.4)
pic(s, 'c_step_xb.png', 4.6, y0, z, z, True); lab(s, 4.5, y0 + z + 0.03, z + 0.2, 'xóa bớt một ít nhiễu')
tb(s, 6.0, y0 + 0.1, 1.2, 1.1, [[('Lặp lại đến khi hết nhiễu.', 10, ORG, True)]], anchor=MSO_ANCHOR.MIDDLE)
box(s, 7.5, 4.5, 5.2, 2.4, LBL, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
tb(s, 7.7, 4.6, 4.8, 0.3, [[('Nhiễu ban đầu khác → ảnh khác', 14, BLU, True)]])
for k in range(3):
    x = 7.75 + k * 1.6
    pic(s, f'c_seed{k}_noise.png', x, 5.0, 0.8, 0.8, True); tb(s, x + 0.82, 5.22, 0.3, 0.35, [[('→', 12, BLU, True)]])
    pic(s, f'c_seed{k}_img.png', x + 0.55, 5.55, 0.9, 0.9, True)
tb(s, 7.7, 6.5, 4.8, 0.35, [[('Stable Diffusion thêm câu lệnh để chọn ảnh theo mô tả (slide 13).', 10.5, INK, True)]])
B.notes_slide.notes_text_frame.text = (
    'Khi mạng đã biết tìm nhiễu, sinh ảnh mới là đi ngược lại. Bắt đầu từ một nhiễu hoàn toàn ngẫu nhiên. '
    'Ở mỗi bước, mạng nhìn ảnh đang nhiễu và đoán ảnh sạch trông thế nào, rồi chỉ xóa bớt một ít nhiễu theo hướng đó. '
    'Lặp lại nhiều bước: lúc đầu chưa thấy gì, rồi có dáng con bò, cuối cùng là ảnh rõ nét. '
    'Nhiễu ban đầu khác thì ra ảnh khác, như ba ví dụ bên phải. Dãy ảnh được tính thật theo công thức DDPM; để chạy được trên máy thường, nhóm dùng bộ đoán lý tưởng trên 12 ảnh thay cho U-Net. '
    'Stable Diffusion làm đúng như vậy nhưng thêm câu lệnh, để ảnh sinh ra khớp với câu mô tả; bây giờ quay lại slide 13.')

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
