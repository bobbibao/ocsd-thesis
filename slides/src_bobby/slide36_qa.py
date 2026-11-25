"""Bobby 2026-11-26: slide 36 'Trả lời ba câu hỏi nghiên cứu' - only questions and answers, answers stand out.
Usage: slide36_qa.py IN OUT"""
import sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
INK, MUT, ORG, BLU, WHT, PEA = C(0x1F,0x24,0x30), C(0x5B,0x62,0x70), C(0xD9,0x69,0x2B), C(0x2E,0x5A,0x9C), C(255,255,255), C(0xFB,0xE9,0xDD)
p = Presentation(sys.argv[1]); s = p.slides[35]
S = {x.name: x for x in s.shapes}
QA = []
for q, a in (('TextBox 7', 'TextBox 8'), ('TextBox 20', 'TextBox 21'), ('TextBox 33', 'TextBox 34')):
    r = S[q].text_frame.paragraphs[0].runs
    QA.append((r[0].text.strip(), ''.join(x.text for x in r[1:]).strip(), S[a].text_frame.text.strip()))
for x in list(s.shapes):
    if x.name not in ('Text 0', 'Text 1', 'Text 2', 'Text 3'): x._element.getparent().remove(x._element)
def tb(x, y, w, h, runs, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT):
    t = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    para = tf.paragraphs[0]; para.alignment = align
    for txt, sz, col, b in runs:
        r = para.add_run(); r.text = txt; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b; r.font.name = 'Arial'
    return t
for k, (lab, q, a) in enumerate(QA):
    y = 1.6 + k * 1.78
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, I(0.6), I(y + 0.35), I(0.85), I(0.85)); o.fill.solid(); o.fill.fore_color.rgb = BLU; o.line.fill.background(); o.shadow.inherit = False
    tb(0.6, y + 0.35, 0.85, 0.85, [(f'Q{k + 1}', 18, WHT, True)], MSO_ANCHOR.MIDDLE, PP_ALIGN.CENTER)
    tb(1.75, y, 10.95, 0.55, [(lab + ' ', 13, ORG, True), (q, 13, MUT, False)])
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(1.75), I(y + 0.62), I(10.95), I(0.95))
    card.adjustments[0] = 0.12; card.fill.solid(); card.fill.fore_color.rgb = PEA; card.line.fill.background(); card.shadow.inherit = False
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, I(1.75), I(y + 0.62), I(0.08), I(0.95)); bar.fill.solid(); bar.fill.fore_color.rgb = ORG; bar.line.fill.background(); bar.shadow.inherit = False
    tb(2.0, y + 0.62, 10.55, 0.95, [(a, 15.5, INK, True)], MSO_ANCHOR.MIDDLE)
p.save(sys.argv[2]); print('ok', [x[0] for x in QA])
