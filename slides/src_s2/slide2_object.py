"""v41 (Bobby 2026-11-26, page 2): short 'Vật thể' definition; object words coloured like the sketch strokes
(house red, bicycle blue, tree green, dog gold) with arrows to the object in the sketch picture.
Box moved right under the picture, caption below it. Right-side click reveal rebuilt.
Usage: slide2_object.py IN OUT"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_bobby'))
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from lxml import etree
from anim_lib import add_clicks
I = Inches
p = Presentation(sys.argv[1]); s = p.slides[1]
sh = {x.shape_id: x for x in s.shapes}
pic, cap, bg, tb = sh[32], sh[31], sh[33], sh[34]
assert tb.text_frame.text.startswith('Vật thể (object)')
# picture px -> slide inches
PX = pic.width / I(1) / 2000
def pt(px, py): return pic.left / I(1) + px * PX, pic.top / I(1) + py * PX

bg.top, bg.height = I(4.45), I(1.3)
tb._element.getparent().remove(tb._element)
cap.top = I(5.9)
t = s.shapes.add_textbox(I(8.35), I(4.95), I(4.35), I(0.78)); tf = t.text_frame; tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.TOP
pa = tf.paragraphs[0]
for txt, bold, col in [('Vật thể (object): ', True, 'D9692B'),
                       ('mỗi thứ riêng lẻ được vẽ trong phác thảo. OCSD phải giữ đủ, đúng chỗ từng vật thể trong ảnh.', False, '1F2430')]:
    r = pa.add_run(); r.text = txt; r.font.bold = bold; r.font.size = Pt(12); r.font.name = 'Arial'; r.font.color.rgb = RGBColor.from_string(col)
new = [t]
OBJ = [('ngôi nhà', 'C62828', (150, 400)), ('xe đạp', '2E5A9C', (240, 560)),
       ('cái cây', '558B2F', (360, 420)), ('con chó', 'C99700', (470, 440))]
CW, X0, CY = 0.95, 8.4, 4.55
for k, (word, col, (px, py)) in enumerate(OBJ):
    x = X0 + k * (CW + 0.12)
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(x), I(CY), I(CW), I(0.32))
    c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.line.color.rgb = RGBColor.from_string(col); c.line.width = Pt(1.5); c.shadow.inherit = False
    f = c.text_frame; f.margin_left = f.margin_right = f.margin_top = f.margin_bottom = 0
    f.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = f.paragraphs[0].add_run(); r.text = word; f.paragraphs[0].alignment = PP_ALIGN.CENTER
    r.font.bold = True; r.font.size = Pt(12); r.font.name = 'Arial'; r.font.color.rgb = RGBColor.from_string(col)
    tx, ty = pt(px, py)
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(x + CW / 2), I(CY), I(tx), I(ty))
    ln.line.color.rgb = RGBColor.from_string(col); ln.line.width = Pt(1.75)
    lnx = ln.line._get_or_add_ln()
    lnx.append(etree.SubElement(lnx, '{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd', type='triangle', w='med', len='med'))
    new += [c, ln]
# click: whole right side (picture, box, chips, arrows, caption) as before
right = [x for x in s.shapes if x.left >= I(8.0) and I(1.2) <= x.top < I(7.0)]
add_clicks(s, [[(x, 'fade') for x in right]])
p.save(sys.argv[2]); print('ok', len(right))
