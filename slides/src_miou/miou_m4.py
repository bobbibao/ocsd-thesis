"""v49: callout uses old hard-blend M4 on the 72 paper scenes (mIoU 0.529, OPR 0.711, docs/M4_STUDY_RESULTS.md).
v48 (Bobby 2026-11-26, page 41 = slide 34): show that tuning M4 raised mIoU. Honest source: development split of
the M4 study (results/m4_study_v1/SELECTION.md, 12 tuning scenes, seed 0): harmonised M4 before tuning (ocsd_m4)
mIoU 0.405 -> tuned M4 (ocsd_m4_combo = final OCSD) 0.511. Adds a callout on the mIoU chart + one note sentence.
Usage: miou_m4.py IN OUT"""
import sys, copy
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
p = Presentation(sys.argv[1]); s = p.slides[33]
assert s.shapes.title is None or True
names = {x.name: x for x in s.shapes}
assert names['TextBox 12'].text_frame.text.startswith('Vị trí, hình dạng')
ORANGE = RGBColor(0xD9, 0x6A, 0x2B); DARK = RGBColor(0x1F, 0x24, 0x33)
X, Y, W, H = float(sys.argv[3]) if len(sys.argv) > 3 else 3.8, float(sys.argv[4]) if len(sys.argv) > 4 else 5.5, 2.6, 0.7
b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(X), I(Y), I(W), I(H))
b.name = 'mIoU M4 callout'
b.fill.solid(); b.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
b.line.color.rgb = ORANGE; b.line.width = Pt(1)
b.shadow.inherit = False
tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
for m in ('margin_left', 'margin_right'): setattr(tf, m, I(0.06))
tf.margin_top = tf.margin_bottom = I(0.02)
lines = [('M4 cũ (trộn cứng): mIoU 0,53', 12, True, ORANGE),
         ('(cùng 72 cảnh; giữ vật thể 71%, kém tự nhiên hơn)', 9, False, DARK)]
for i, (t, sz, bold, col) in enumerate(lines):
    para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    para.alignment = PP_ALIGN.CENTER
    r = para.add_run(); r.text = t; r.font.name = 'Arial'; r.font.size = Pt(sz); r.font.bold = bold; r.font.color.rgb = col
nt = s.notes_slide.notes_text_frame; t = nt.text
a = 'đây là điểm yếu nhóm nói rõ.'
assert a in t
nt.text = t.replace(a, a + ' Ô viền cam trong thẻ vị trí, hình dạng là bản M4 cũ (trộn cứng) trên cùng 72 cảnh: mIoU 0,53, cao hơn bản cuối 0,48, nhưng giữ vật thể chỉ 71 phần trăm và kém tự nhiên hơn, nên nhóm chọn M4 hài hòa.')
p.save(sys.argv[2]); print('ok')
