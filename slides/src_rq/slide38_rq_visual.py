"""Bobby 2026-11-26: slide 38 'Trả lời ba câu hỏi nghiên cứu' had too much text, no visuals, no separation between
questions. Rebuild as 3 columns (Q1 | Q2 | Q3) split by vertical rules; each column = short question, one native bar
visual, one short answer box. One click per column. Numbers unchanged (m4_study_v1 + Q1 controlled test).
Usage: slide38_rq_visual.py IN OUT"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_bobby'))
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from anim_lib import add_clicks

BLUE, OR, PEACH, DARK = '2E5A9C', 'D9692B', 'FBE9DD', '1F2430'
GRAY, LGRAY, GREEN, RED, RULE = '6B7280', 'C9CED6', '2E8B57', 'C0392B', 'D5D9E0'
p = Presentation(sys.argv[1]); s = p.slides[37]
assert s.shapes[1].text_frame.text.startswith('Trả lời ba câu hỏi'), 'slide 38 moved'
for x in list(s.shapes):
    if x.shape_id >= 6: x._element.getparent().remove(x._element)
I = Inches

def box(x, y, w, h, fill=None, shape=MSO_SHAPE.RECTANGLE, adj=None):
    b = s.shapes.add_shape(shape, I(x), I(y), I(w), I(h))
    if fill: b.fill.solid(); b.fill.fore_color.rgb = RGBColor.from_string(fill)
    else: b.fill.background()
    b.line.fill.background(); b.shadow.inherit = False
    if adj is not None: b.adjustments[0] = adj
    return b

def text(x, y, w, h, runs, size=12, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, shape=None):
    """runs: list of paragraphs; each paragraph = list of (text, bold, color[, size])."""
    t = shape or s.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = I(0.04); tf.margin_top = tf.margin_bottom = I(0.02)
    for k, para in enumerate(runs):
        pa = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); pa.alignment = align
        for r in para:
            rr = pa.add_run(); rr.text = r[0]; f = rr.font
            f.bold = r[1]; f.color.rgb = RGBColor.from_string(r[2]); f.size = Pt(r[3] if len(r) > 3 else size); f.name = 'Arial'
    return t

def bar(x, y, label, val, maxv, width, color, vtxt, lw=1.05, h=0.26):
    g = [text(x, y - 0.02, lw, h + 0.04, [[(label, False, DARK, 11)]], anchor=MSO_ANCHOR.MIDDLE)]
    g.append(box(x + lw, y, max(0.05, width * val / maxv), h, color))
    g.append(text(x + lw + width * val / maxv + 0.05, y - 0.02, 0.9, h + 0.04, [[(vtxt, True, DARK, 11.5)]], anchor=MSO_ANCHOR.MIDDLE))
    return g

COLX, COLW, TOP = [0.6, 4.72, 8.84], 3.86, 1.55
cols = []
for c in range(3):
    if c: ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(COLX[c] - 0.15), I(TOP), I(COLX[c] - 0.15), I(6.85)); \
          ln.line.color.rgb = RGBColor.from_string(RULE); ln.line.width = Pt(1.5)
HEAD = [('Q1', 'Xác định nguyên nhân suy giảm', 'Số lượng vật thể hay độ phức tạp phác thảo khiến mô hình nền vi phạm R1–R4?'),
        ('Q2', 'Đề xuất phương pháp và so sánh', 'Tách cảnh thành từng vật thể có định danh riêng có giúp đạt R1, R2, R4 tốt hơn? Đánh đổi gì?'),
        ('Q3', 'Hợp thành cảnh hài hòa', 'Hợp thành cảnh hài hòa có giữ nhiều vật thể hơn và tự nhiên hơn cách trộn cứng?')]
ANS = ['Chủ yếu do số lượng vật thể; nét vẽ phức tạp chỉ giảm 8 điểm. Lỗi chính: mất vật, sai số lượng.',
       'Có, so với phương pháp chỉ dùng nét vẽ. Ngang GLIGEN, Collage về giữ vật; đổi độ chính xác vị trí lấy độ tự nhiên.',
       'Có, cả hai giả thuyết đăng ký trước đều đúng; mIoU và RA không giảm có ý nghĩa.']
for c, (q, title, ques) in enumerate(HEAD):
    x = COLX[c]; g = []
    o = box(x, TOP, 0.62, 0.62, BLUE, MSO_SHAPE.OVAL); g.append(o)
    text(0, 0, 0, 0, [[(q, True, 'FFFFFF', 16)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=o)
    g.append(text(x + 0.72, TOP, COLW - 0.72, 0.62, [[(title, True, OR, 14.5)]], anchor=MSO_ANCHOR.MIDDLE))
    g.append(text(x, TOP + 0.72, COLW, 0.62, [[(ques, False, GRAY, 11.5)]]))
    vis = box(x, 2.95, COLW, 2.6, 'F4F5F7', MSO_SHAPE.ROUNDED_RECTANGLE, 0.06); g.append(vis)
    a = box(x, 5.7, COLW, 1.15, PEACH, MSO_SHAPE.ROUNDED_RECTANGLE, 0.1); g.append(a)
    g.append(box(x, 5.7, 0.07, 1.15, OR))
    g.append(text(x + 0.18, 5.7, COLW - 0.28, 1.15, [[(ANS[c], True, DARK, 13)]], anchor=MSO_ANCHOR.MIDDLE))
    cols.append(g)

# Q1 visual: object count vs sketch complexity, base model OPR
x = COLX[0] + 0.15; g = cols[0]; W = 1.75
g.append(text(x, 3.03, 3.6, 0.3, [[('Giữ vật thể của mô hình nền (OPR)', True, DARK, 11.5)]]))
g.append(text(x, 3.35, 3.6, 0.25, [[('Tăng số vật thể', False, GRAY, 10.5)]]))
g += bar(x, 3.62, '1 vật', 97, 100, W, GRAY, '97%') + bar(x, 3.93, '8+ vật', 39, 100, W, RED, '39%  ▼58')
g.append(text(x, 4.3, 3.6, 0.25, [[('Tăng độ phức tạp nét vẽ', False, GRAY, 10.5)]]))
g += bar(x, 4.57, 'đơn giản', 70, 100, W, GRAY, '70%') + bar(x, 4.88, 'phức tạp', 62, 100, W, LGRAY, '62%  ▼8')
g.append(text(x, 5.2, 3.6, 0.3, [[('Sai số lượng mỗi loại: 0,19 → ', False, DARK, 11), ('5,97', True, RED, 11)]]))

# Q2 visual: OPR vs baselines + trade-off chips
x = COLX[1] + 0.15; g = cols[1]; W = 1.55
g.append(text(x, 3.03, 3.6, 0.3, [[('Giữ vật thể (OPR), 72 cảnh × 2 hạt giống', True, DARK, 11.5)]]))
g += bar(x, 3.4, 'Collage', 82, 100, W, LGRAY, '82%') + bar(x, 3.7, 'GLIGEN', 80, 100, W, LGRAY, '80%')
g += bar(x, 4.0, 'OCSD', 79, 100, W, OR, '79%') + bar(x, 4.3, 'Chỉ nét vẽ*', 66, 100, W, GRAY, '62–66%')
g.append(text(x, 4.62, 3.6, 0.3, [[('Vị trí (mIoU): ', False, DARK, 11), ('0,48 < 0,61–0,65  ▼', True, RED, 11)]]))
g.append(text(x, 4.9, 3.6, 0.3, [[('Tự nhiên (PickScore): ', False, DARK, 11), ('21,89 > 21,41–21,52  ▲', True, GREEN, 11)]]))
g.append(text(x, 5.2, 3.6, 0.3, [[('*ControlNet, T2I-Adapter, Zhang (2025)', False, GRAY, 9.5)]]))

# Q3 visual: hard blend vs harmonised M4
x = COLX[2] + 0.15; g = cols[2]; W = 1.6
g.append(text(x, 3.03, 3.6, 0.3, [[('Trộn cứng  ', True, GRAY, 11.5), ('→  ', True, DARK, 11.5), ('Hợp thành hài hòa', True, OR, 11.5)]]))
g.append(text(x, 3.35, 3.6, 0.25, [[('Giữ vật thể (OPR)', False, GRAY, 10.5)]]))
g += bar(x, 3.62, 'trộn cứng', 71, 100, W, GRAY, '71%') + bar(x, 3.93, 'hài hòa', 79, 100, W, OR, '79%  ▲')
g.append(text(x, 4.3, 3.6, 0.25, [[('Chấm mù độ tự nhiên (thang 0–2)', False, GRAY, 10.5)]]))
g += bar(x, 4.57, 'trộn cứng', 0.44, 2, W, GRAY, '0,44') + bar(x, 4.88, 'hài hòa', 1.36, 2, W, OR, '1,36  ▲')
g.append(text(x, 5.2, 3.6, 0.3, [[('PickScore ', False, DARK, 11), ('+0,73', True, GREEN, 11), (' (p < 0,0001), OPR p = 0,011', False, DARK, 10.5)]]))

add_clicks(s, [[(sh, 'fade') for sh in g] for g in cols])
p.save(sys.argv[2]); print('ok')
