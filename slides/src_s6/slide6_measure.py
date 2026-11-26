"""Bobby 2026-11-26: slide 6 must not say 'Kiểm chứng'; it says what was measured, on which model, how, and what was
seen, with animation and no empty space. Numbers: paper Section 5 Q1 (ControlNet OPR 0.97 -> 0.39 by object count,
0.70 -> 0.62 by sketch complexity, count error 0.19 -> 5.97; QuickDraw-Scenes 72 scenes, OWLv2).
Usage: slide6_measure.py IN OUT [--state N]
2nd pass (Bobby 13:07): shorter; title = "Đo trực tiếp trên mô hình tạo ảnh SD 1.5 + ControlNet và thấy" (tạo ảnh per term rule)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src_bobby'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from anim_lib import add_clicks
IN, OUT = sys.argv[1:3]
state = int(sys.argv[sys.argv.index('--state') + 1]) if '--state' in sys.argv else 0
HERE = os.path.dirname(os.path.abspath(__file__)); IMG = os.path.join(HERE, 'img', 'opr_by_count.png')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(6.2, 4.6))
v = [97, 64, 62, 39]; b = ax.bar(['1 vật', '3 vật', '5 vật', '8+ vật'], v, color=['#c9ced4', '#b0b7bf', '#8f98a2', '#e0662a'], width=.58)
for r, x_ in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, x_ + 2.5, f'{x_}%', ha='center', fontsize=17, fontweight='bold')
ax.set_title('Tỉ lệ vật thể còn giữ được', fontsize=15, fontweight='bold'); ax.set_ylim(0, 112); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.tick_params(axis='x', length=0, labelsize=14)
fig.tight_layout(); fig.savefig(IMG, dpi=220); plt.close(fig)

p = Presentation(IN); s = p.slides[5]
S = {x.name: x for x in s.shapes}
assert S['Text 1'].text_frame.text.startswith('Kiểm chứng'), 'slide 6 moved'
t = S['Text 1'].text_frame.paragraphs[0].runs; t[0].text = 'Đo trực tiếp trên mô hình tạo ảnh SD 1.5 + ControlNet và thấy'
for r in t: r.font.size = Pt(28)
for r in t[1:]: r.text = ''
for n in ('Picture 12', 'Text 4', 'Text 5'): S[n]._element.getparent().remove(S[n]._element)
INK, GREY, OR, BLUE = RGBColor(0x1F, 0x24, 0x30), RGBColor(0x5B, 0x62, 0x70), RGBColor(0xD9, 0x69, 0x2B), RGBColor(0x2E, 0x5A, 0x9C)
def tb(l, t_, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, shape=None):
    x = shape or s.shapes.add_textbox(I(l), I(t_), I(w), I(h)); tf = x.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = I(0.1); tf.margin_top = tf.margin_bottom = I(0.04)
    for k, para in enumerate(paras):
        pa = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); pa.alignment = align
        for r in para:
            rr = pa.add_run(); rr.text = r[0]; f = rr.font; f.bold = r[1]; f.color.rgb = r[2]; f.size = Pt(r[3]); f.name = 'Arial'
    return x
def card(l, t_, w, h, fill, line=None):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(l), I(t_), I(w), I(h)); c.adjustments[0] = 0.1
    c.fill.solid(); c.fill.fore_color.rgb = RGBColor.from_string(fill); c.shadow.inherit = False
    if line: c.line.color.rgb = RGBColor.from_string(line); c.line.width = Pt(1.25)
    else: c.line.fill.background()
    return c
def dot(l, t_, n, col):
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, I(l), I(t_), I(0.42), I(0.42)); o.fill.solid(); o.fill.fore_color.rgb = col; o.line.fill.background(); o.shadow.inherit = False
    tb(0, 0, 0, 0, [[(str(n), True, RGBColor(0xFF, 0xFF, 0xFF), 14)]], PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, shape=o); return o
sub = tb(0.6, 1.45, 12.1, 0.35, [[('72 cảnh QuickDraw-Scenes · đếm vật trong ảnh bằng OWLv2', False, GREY, 13)]])
chart = s.shapes.add_picture(IMG, I(0.6), I(1.95), I(6.7), I(6.7 * 4.6 / 6.2))
c1 = card(7.6, 1.95, 5.13, 4.85, 'FBE9DD')
tb(0, 0, 0, 0, [[('97% → 39%', True, OR, 44)], [('vật thể còn giữ được, khi phác thảo', False, INK, 16)], [('tăng từ 1 lên 8+ vật thể', False, INK, 16)]],
   PP_ALIGN.LEFT, MSO_ANCHOR.MIDDLE, shape=c1)
c1.text_frame.margin_left = I(0.35)
steps = [[chart], [c1]]
# 4th pass (Bobby 13:16): no conclusion on this slide (it would pre-empt Q1)
# 3rd pass (Bobby 13:13): the 70% -> 62% box removed
s.notes_slide.notes_text_frame.text = (
    'Trước khi đề xuất phương pháp, nhóm đo xem mô hình nền hỏng ở đâu. Một, mô hình được đo là SD 1.5 cộng ControlNet nhận nét vẽ, cách làm phổ biến nhất. '
    'Hai, dữ liệu là bộ QuickDraw-Scenes nhóm tự xây, 72 cảnh, trong đó số vật thể và độ phức tạp nét vẽ được thay đổi riêng rẽ. '
    'Ba, cách đo: dùng OWLv2, một máy phát hiện vật độc lập mà không phương pháp nào dùng bên trong, tìm từng vật trong câu mô tả; tỉ lệ vật tìm thấy là tỉ lệ vật thể còn giữ được. '
    'Kết quả: với một vật thể, 97 phần trăm được giữ; với 8 vật trở lên chỉ còn 39 phần trăm. '
    'Câu hỏi đặt ra là vì sao, và đó là câu hỏi nghiên cứu Q1.')
if state:
    for st in steps[state:]:
        for x in st: x._element.getparent().remove(x._element)
    ids = p.slides._sldIdLst
    for i, sid in reversed(list(enumerate(list(ids)))):
        if i != 5: p.part.drop_rel(sid.rId); ids.remove(sid)
else:
    add_clicks(s, [[(x, 'wipe' if x is chart else 'fade') for x in st] for st in steps])
p.save(OUT); print('ok', state, len(steps))
