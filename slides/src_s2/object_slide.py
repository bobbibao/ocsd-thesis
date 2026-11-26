"""v44 (Bobby 2026-11-26, page 2): move the 'Vật thể' explanation off the agenda slide to its own slide 3
('Vật thể (object) là gì?'): big sketch, coloured chips with arrows to each object, short definition box.
Slide 2 keeps only the input -> output figure + caption. Footers renumbered N / 41.
Usage: object_slide.py IN OUT   (IN = v43, PICTURE = sketch figure saved from slide 2)"""
import sys, os, re, copy, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_bobby'))
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from lxml import etree
from PIL import Image
from anim_lib import add_clicks, set_transition
I = Inches
p = Presentation(sys.argv[1]); s2 = p.slides[1]
by = {x.shape_id: x for x in s2.shapes}
pic, cap = by[32], by[31]
blob = pic.image.blob
# ---- slide 2: remove definition box, chips, arrows; centre figure + caption
for x in list(s2.shapes):
    if x.left >= I(8.0) and I(4.4) <= x.top < I(5.9) and x.shape_id not in (32, 31):
        x._element.getparent().remove(x._element)
for x in list(s2.shapes):   # connectors (arrows) start at the chips
    if x.shape_type == 9 and x.left >= I(8.0) and x.shape_id not in (32, 31): x._element.getparent().remove(x._element)
pic.top = I(2.55); cap.top = pic.top + pic.height + I(0.15)
right = [x for x in s2.shapes if x.left >= I(8.0) and I(1.2) <= x.top < I(7.0)]
add_clicks(s2, [[(x, 'fade') for x in right]])
nt = s2.notes_slide.notes_text_frame; t = nt.text; i = t.index(' Trong bài, vật thể')
nt.text = t[:i] + ' Bên phải là ví dụ: đầu vào là phác thảo cảnh và câu mô tả, đầu ra là ảnh cảnh giữ đúng các vật thể.'

# ---- new slide 3
s = p.slides.add_slide(s2.slide_layout)
for ph in list(s.placeholders): ph._element.getparent().remove(ph._element)
for sid in (2, 3, 4, 5):
    s.shapes._spTree.append(copy.deepcopy(by[sid]._element))
sh = {x.name: x for x in s.shapes}
sh['Text 0'].text_frame.paragraphs[0].runs[0].text = 'THUẬT NGỮ'
r = sh['Text 1'].text_frame.paragraphs[0].runs; r[0].text = 'Vật thể (object) là gì?'
for e in r[1:]: e._r.getparent().remove(e._r)
crop = Image.open(io.BytesIO(blob)).crop((0, 0, 646, 920)); buf = io.BytesIO(); crop.save(buf, 'PNG'); buf.seek(0)
X0, Y0, H = 3.0, 1.6, 5.0; S = H / 920
im = s.shapes.add_picture(buf, I(X0), I(Y0), height=I(H))
def P(px, py): return X0 + px * S, Y0 + py * S
def chip(x, y, w, word, col):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(x), I(y), I(w), I(0.5))
    c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.line.color.rgb = RGBColor.from_string(col); c.line.width = Pt(2); c.shadow.inherit = False
    f = c.text_frame; f.margin_left = f.margin_right = f.margin_top = f.margin_bottom = 0; f.vertical_anchor = MSO_ANCHOR.MIDDLE
    pa = f.paragraphs[0]; pa.alignment = PP_ALIGN.CENTER; rr = pa.add_run(); rr.text = word
    rr.font.bold = True; rr.font.size = Pt(18); rr.font.name = 'Arial'; rr.font.color.rgb = RGBColor.from_string(col)
    return c
def arrow(x1, y1, x2, y2, col):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(x1), I(y1), I(x2), I(y2))
    ln.line.color.rgb = RGBColor.from_string(col); ln.line.width = Pt(2.25)
    lnx = ln.line._get_or_add_ln()
    etree.SubElement(lnx, '{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd', type='triangle', w='med', len='med')
    return ln
objs = []
# (word, colour, chip side, target px)
for word, col, side, (px, py) in [('ngôi nhà', 'C62828', 'L', (85, 330)), ('xe đạp', '2E5A9C', 'L', (123, 560)),
                                  ('mặt trời', '7B1FA2', 'R', (622, 135)), ('cái cây', '558B2F', 'R', (420, 250)), ('con chó', 'C99700', 'R', (575, 420))]:
    tx, ty = P(px, py)
    if side == 'L': c = chip(0.7, ty - 0.25, 1.75, word, col); a = arrow(2.45, ty, tx - 0.03, ty, col)
    else:
        cy = ty - 0.2 if word == 'mặt trời' else ty   # lift the sun chip so it does not touch the tree chip
        c = chip(6.75, cy - 0.25, 1.75, word, col); a = arrow(6.75, cy, tx + 0.03, ty, col)
    objs += [c, a]
def card(y, h, fill, paras):
    bx = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(8.95), I(y), I(3.8), I(h))
    bx.fill.solid(); bx.fill.fore_color.rgb = RGBColor.from_string(fill); bx.line.fill.background(); bx.shadow.inherit = False
    bx.adjustments[0] = 0.08
    tf = bx.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = I(0.22)
    for k, runs in enumerate(paras):
        pa = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); pa.space_after = Pt(6)
        for t_, b_, c_, z_ in runs:
            rr = pa.add_run(); rr.text = t_; rr.font.bold = b_; rr.font.size = Pt(z_); rr.font.name = 'Arial'
            rr.font.color.rgb = RGBColor.from_string(c_)
    return bx
DK = '1F2430'
bx = card(1.75, 2.35, 'FBE9DD', [[('Vật thể (object)', True, 'D9692B', 19)],
                                 [('Mỗi thứ riêng lẻ được vẽ trong phác thảo.', False, DK, 15)],
                                 [('Ảnh tạo ra cần có ', False, DK, 15), ('đủ', True, DK, 15), (' và ', False, DK, 15),
                                  ('đúng chỗ', True, DK, 15), (' từng vật thể.', False, DK, 15)]])
bx2 = card(4.3, 2.35, 'E6EEFA', [[('Thuộc tính (attribute)', True, '2E5A9C', 19)],
                                 [('Đặc điểm của vật thể, như màu sắc hay chất liệu, lấy từ câu mô tả:', False, DK, 14)],
                                 [('brown dog', True, 'C99700', 14), (' → chó màu nâu, ', False, DK, 14),
                                  ('red bicycle', True, '2E5A9C', 14), (' → xe đạp màu đỏ.', False, DK, 14)]])
add_clicks(s, [[(x, 'fade') for x in objs], [(bx, 'fade')], [(bx2, 'fade')]])
set_transition(s, 'fade')
s.notes_slide.notes_text_frame.text = ('Trước khi vào bài, nhóm giải thích một thuật ngữ dùng xuyên suốt: vật thể, tiếng Anh là object. '
    'Vật thể là mỗi thứ riêng lẻ được vẽ trong phác thảo, ở ví dụ này là ngôi nhà, xe đạp, cái cây, con chó và mặt trời. '
    'Thuộc tính, tiếng Anh là attribute, là đặc điểm của vật thể như màu sắc hay chất liệu, lấy từ câu mô tả, ví dụ brown dog là chó màu nâu, red bicycle là xe đạp màu đỏ. '
    'Nhất quán vật thể nghĩa là ảnh tạo ra có đủ các vật thể đó, đúng số lượng, đúng vị trí và đúng tương quan như trong phác thảo.')
# move new slide to position 3
lst = p.slides._sldIdLst; el = lst[-1]; lst.remove(el); lst.insert(2, el)
tot = len(p.slides)
for k, sl in enumerate(p.slides, 1):
    for x in sl.shapes:
        if x.has_text_frame and re.fullmatch(r'\d+ / \d+', x.text_frame.text.strip()):
            rr = x.text_frame.paragraphs[0].runs; rr[0].text = f'{k} / {tot}'
            for e in rr[1:]: e._r.getparent().remove(e._r)
p.save(sys.argv[2]); print('ok', tot)
