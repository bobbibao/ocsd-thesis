"""Slide 'Hai bộ dữ liệu kiểm thử': replace the two code boxes with key numbers and what each set tests."""
import sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
SRC, OUT = sys.argv[1:3]
p = Presentation(SRC)
INK, MUT, ORG, BLU, GRY, PEA = C(0x1F,0x24,0x30), C(0x5B,0x62,0x70), C(0xD9,0x69,0x2B), C(0x2E,0x5A,0x9C), C(0xF3,0xF4,0xF6), C(0xFB,0xE9,0xDD)
s = [s for s in p.slides if any(x.name == 'Text 1' and x.text_frame.text.startswith('Hai bộ dữ liệu kiểm thử do nhóm') for x in s.shapes)][0]
for x in list(s.shapes):
    if x.name in ('Rectangle 45', 'TextBox 46', 'TextBox 47', 'Rectangle 48', 'TextBox 49', 'TextBox 50'):
        x._element.getparent().remove(x._element)
def tb(x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=2):
    t = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in 'margin_left', 'margin_right', 'margin_top', 'margin_bottom': setattr(tf, m, 0)
    for k, line in enumerate(paras):
        par = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); par.alignment = align; par.space_after = Pt(space)
        for txt, sz, col, b in line:
            r = par.add_run(); r.text = txt; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b; r.font.name = 'Arial'
def panel(x0, fill, acc, stats, tests):
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, I(x0), I(3.95), I(5.9), I(1.5)); b.fill.solid(); b.fill.fore_color.rgb = fill; b.line.fill.background(); b.shadow.inherit = False
    cw = 5.6 / 3
    for k, (num, cap) in enumerate(stats):
        tb(x0 + 0.15 + k * cw, 4.02, cw - 0.1, 0.8, [[(num, 20, acc, True)], [(cap, 9.5, MUT, False)]], align=PP_ALIGN.CENTER, space=0)
    tb(x0 + 0.15, 4.86, 5.6, 0.55, [[('Kiểm tra: ', 11, INK, True), (tests, 11, INK, False)]])
panel(0.6, GRY, BLU, [('72', 'cảnh đánh giá'), ('1 · 3 · 5 · 8+', 'đối tượng mỗi cảnh'), ('3 mức', 'độ chi tiết nét vẽ')],
      'R1–R4 với đáp án chính xác (biết vùng từng vật); tách được ảnh hưởng của số vật và độ khó nét vẽ.')
panel(6.83, PEA, ORG, [('32', 'cảnh đánh giá'), ('1 – 8', 'đối tượng, chia 4 mức'), ('Ảnh thật', 'cảnh đời thực của COCO')],
      'phương pháp có còn giữ đúng đối tượng khi cảnh và phác thảo giống đời thực hay không.')
n = s.notes_slide.notes_text_frame
t = n.text.replace(' Code nằm trong ocsd/data.py, hàm build_quickdraw_scenes.', ' Bộ này có 72 cảnh đánh giá, mỗi cảnh 1, 3, 5 hoặc từ 8 đối tượng trở lên, ba mức độ chi tiết nét vẽ; vì biết vùng của từng vật nên đo được chính xác cả bốn yêu cầu R1 đến R4.')
t = t.replace(' Hàm build_coco_sketch.', ' Bộ này có 32 cảnh đánh giá chia bốn mức số đối tượng, dùng để kiểm tra phương pháp trên cảnh đời thực.')
n.text = t
p.save(OUT)
