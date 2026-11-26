"""Bobby 2026-11-26: M1 card (slide 24) gets a third picture showing the pair relations M1 infers (truck below cow 1,
tree right of truck, cow 2 above cow 3), with a native-text legend. Sketch and mask panels shrink to make room.
Usage: slide24_rel.py IN OUT REL_PNG"""
import sys, copy
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor
p = Presentation(sys.argv[1]); s = p.slides[23]
sh = {x.name: x for x in s.shapes}
assert s.shapes[1].text_frame.text.startswith('M1-Phân rã'), 'slide 24 moved'
def put(x, l, t, w, h): x.left, x.top, x.width, x.height = I(l), I(t), I(w), I(h)
put(sh['Picture 60'], 0.8, 2.2, 1.6, 1.6); put(sh['Picture 61'], 2.8, 2.2, 1.6, 1.6)
put(sh['Right Arrow 28'], 2.45, 2.88, 0.3, 0.22)
a2 = copy.deepcopy(sh['Right Arrow 28']._element); sh['Right Arrow 28']._element.addnext(a2)
a2.xpath('./p:nvSpPr/p:cNvPr')[0].set('id', '901'); a2.xpath('./p:nvSpPr/p:cNvPr')[0].set('name', 'Right Arrow rel')
a2.xpath('.//a:off')[0].set('x', str(int(I(4.45))))
put(sh['TextBox 27'], 0.65, 3.85, 1.9, 0.55); put(sh['TextBox 30'], 2.65, 3.85, 1.9, 0.3)
pic = s.shapes.add_picture(sys.argv[3], I(4.8), I(2.2), I(1.6), I(1.6)); pic.name = 'Picture rel'
pic.line.color.rgb = RGBColor(0xC9, 0xCE, 0xD6); pic.line.width = Pt(0.75)
tb = s.shapes.add_textbox(I(4.72), I(3.83), I(1.8), I(0.75)); tf = tb.text_frame; tf.word_wrap = True
tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
for k, (n, t) in enumerate([('1', 'xe tải dưới bò 1'), ('2', 'cây bên phải xe tải'), ('3', 'bò 2 trên bò 3')]):
    pa = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
    for txt, b, col in ((f'{n} ', True, 'D9692B'), (t, False, '1F2430')):
        r = pa.add_run(); r.text = txt; r.font.size = Pt(10.5); r.font.bold = b; r.font.name = 'Arial'
        r.font.color.rgb = RGBColor.from_string(col)
p.save(sys.argv[2]); print('ok')
