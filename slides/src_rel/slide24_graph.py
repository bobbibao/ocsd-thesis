"""Bobby 2026-11-26 (2nd pass): the relation picture on slide 24 must read like a graph without a legend. Replace it
with a native scene graph: nodes = each object's own sketch (M1 crop) with a box in its M1 mask colour and its name,
arrows subject -> object labelled with the relation (from trace.json m1.relations). Sketch and mask panels shrink.
Usage: slide24_graph.py IN(v26) OUT NODE_DIR"""
import sys
from lxml import etree
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
p = Presentation(sys.argv[1]); s = p.slides[23]
sh = {x.name: x for x in s.shapes}
for x in list(s.shapes):  # drop the v26 relation picture and its legend
    if x.name == 'Picture rel' or (x.has_text_frame and x.text_frame.text.startswith('1 xe tải')):
        x._element.getparent().remove(x._element)
def put(x, l, t, w, h): x.left, x.top, x.width, x.height = I(l), I(t), I(w), I(h)
put(sh['Picture 60'], 0.75, 2.3, 1.3, 1.3); put(sh['Right Arrow 28'], 2.09, 2.84, 0.22, 0.2)
put(sh['Picture 61'], 2.35, 2.3, 1.3, 1.3); put(sh['Right Arrow rel'], 3.69, 2.84, 0.22, 0.2)
put(sh['TextBox 27'], 0.62, 3.66, 1.56, 0.6); put(sh['TextBox 30'], 2.2, 3.66, 1.6, 0.3)
DARK = RGBColor.from_string('1F2430')
N = 0.46  # node picture size
NODES = {  # name: (crop file, colour, left, top)
    'bò 1': ('obj1_cow', 'D9692B', 4.0, 2.15), 'bò 2': ('obj4_cow', '8E44AD', 5.92, 2.15),
    'xe tải': ('obj3_truck', '2E8B57', 4.0, 3.2), 'cây': ('obj2_tree', '2E5A9C', 5.12, 3.2),
    'bò 3': ('obj5_cow', 'C9A200', 5.92, 3.2)}
def tbox(l, t, w, h, txt, size, bold=False, fill=None, col=DARK):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(l), I(t), I(w), I(h)) if fill else s.shapes.add_textbox(I(l), I(t), I(w), I(h))
    if fill:
        b.fill.solid(); b.fill.fore_color.rgb = RGBColor.from_string(fill); b.line.color.rgb = DARK; b.line.width = Pt(0.75); b.shadow.inherit = False
    tf = b.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = False; pa = tf.paragraphs[0]; pa.alignment = PP_ALIGN.CENTER
    r = pa.add_run(); r.text = txt; f = r.font; f.size = Pt(size); f.bold = bold; f.color.rgb = col; f.name = 'Arial'
    return b
cen = {}
for name, (f, col, l, t) in NODES.items():
    pic = s.shapes.add_picture(f'{sys.argv[3]}/{f}_sketch_crop.png', I(l), I(t), I(N), I(N)); pic.name = f'node {name}'
    pic.line.color.rgb = RGBColor.from_string(col); pic.line.width = Pt(2.25)
    tbox(l - 0.1, t + N + 0.01, N + 0.2, 0.18, name, 9.5, True, col=RGBColor.from_string(col))
    cen[name] = (l + N / 2, t + N / 2)
def arrow(a, b, label, lx, ly, lw):
    (x0, y0), (x1, y1) = cen[a], cen[b]
    # start/end on node borders (axis-aligned layout)
    if abs(x1 - x0) > abs(y1 - y0):
        sx = 1 if x1 > x0 else -1; x0 += sx * (N / 2 + 0.03); x1 -= sx * (N / 2 + 0.03)
    else:
        sy = 1 if y1 > y0 else -1; y0 += sy * (N / 2 + 0.03); y1 -= sy * (N / 2 + 0.22 if sy < 0 else N / 2 + 0.03)
        if sy > 0: y0 += 0.19
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(x0), I(y0), I(x1), I(y1))
    c.line.color.rgb = DARK; c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln(); ln.append(etree.SubElement(ln, '{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd', type='triangle', w='med', len='med'))
    tbox(lx, ly, lw, 0.19, label, 8.5, True, fill='FFFFFF')
# subject -> object, read as "subject <relation> object"
arrow('bò 1', 'bò 2', 'bên trái', 4.96, 2.29, 0.5)
arrow('xe tải', 'bò 1', 'dưới', 4.3, 2.9, 0.36)
arrow('cây', 'xe tải', 'bên phải', 4.55, 3.5, 0.5)
arrow('bò 2', 'bò 3', 'trên', 5.72, 2.9, 0.36)
tbox(3.95, 3.9, 2.5, 0.2, 'đồ thị tương quan giữa các vật', 9.5)
p.save(sys.argv[2]); print('ok')
