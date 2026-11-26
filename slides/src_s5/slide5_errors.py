"""Bobby 2026-11-26: slide 5 'Năm kiểu lỗi' shows all 5 errors on ONE input, the deck's farm sketch + caption, with
one click per error. Real seed-0 outputs of the benchmark (m4_study_v1 review sheet 24, key: D=zhang2025,
F=t2i_adapter, G=controlnet). Usage: slide5_errors.py IN OUT IMG_DIR [--state N]  (--state: static copy, slide 5 only)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src_bobby'))
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from anim_lib import add_clicks
IN, OUT, IMG = sys.argv[1:4]
state = int(sys.argv[sys.argv.index('--state') + 1]) if '--state' in sys.argv else 0
p = Presentation(IN); s = p.slides[4]
assert s.shapes[1].text_frame.text.startswith('Năm kiểu lỗi'), 'slide 5 moved'
keep = {'Text 0', 'Text 1', 'TextBox 35', 'TextBox 36', 'TextBox 37'}
for x in list(s.shapes):
    if x.name not in keep: x._element.getparent().remove(x._element)
S = {x.name: x for x in s.shapes}
r = S['TextBox 37'].text_frame.paragraphs[0].runs; r[0].text = '5 / 41'; [q._r.getparent().remove(q._r) for q in r[1:]]
RED, INK, GREY = RGBColor(0xC8, 0x1E, 0x32), RGBColor(0x1F, 0x24, 0x30), RGBColor(0x5B, 0x62, 0x70)
def tb(l, t, w, h, runs, size=11, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, shape=None):
    b = shape or s.shapes.add_textbox(I(l), I(t), I(w), I(h)); tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = I(0.05); tf.margin_top = tf.margin_bottom = I(0.02)
    for k, para in enumerate(runs):
        pa = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); pa.alignment = align
        for t in para:
            rr = pa.add_run(); rr.text = t[0]; f = rr.font; f.size = Pt(t[3] if len(t) > 3 else size); f.bold = t[1]; f.color.rgb = t[2]; f.name = 'Arial'
            if len(t) > 4: f.italic = t[4]
    return b
Z, Y = 2.8, 1.95
COLS = {'sketch': 0.6, 'zhang': 3.82, 't2i': 6.87, 'controlnet': 9.92}
HEAD = {'sketch': 'Phác thảo + câu mô tả', 'zhang': 'Zhang và cs. (2025)', 't2i': 'SD + T2I-Adapter', 'controlnet': 'SD + ControlNet'}
for k, x in COLS.items():
    pic = s.shapes.add_picture(f'{IMG}/{k}.png', I(x), I(Y), I(Z), I(Z))
    pic.line.color.rgb = RGBColor(0xD0, 0xD4, 0xDA); pic.line.width = Pt(0.75)
    tb(x, Y - 0.36, Z, 0.3, [[(HEAD[k], True, INK, 12.5)]], align=PP_ALIGN.CENTER)
tb(0.45, Y + Z + 0.04, Z + 0.3, 0.5, [[('“three small cows, a tall tree and a red truck on a farm”', True, INK, 11.5, True)]], align=PP_ALIGN.CENTER)
def ell(k, x0, y0, x1, y1):
    X = COLS[k]; e = s.shapes.add_shape(MSO_SHAPE.OVAL, I(X + x0 * Z), I(Y + y0 * Z), I((x1 - x0) * Z), I((y1 - y0) * Z))
    e.fill.background(); e.line.color.rgb = RED; e.line.width = Pt(2.5); e.line.dash_style = MSO_LINE_DASH_STYLE.DASH; e.shadow.inherit = False
    return e
def tag(k, fx, fy, txt, w):
    X = COLS[k]; t = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(X + fx * Z), I(Y + fy * Z), I(w), I(0.24))
    t.fill.solid(); t.fill.fore_color.rgb = RED; t.line.fill.background(); t.shadow.inherit = False
    tb(0, 0, 0, 0, [[(txt, True, RGBColor(0xFF, 0xFF, 0xFF), 10)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=t)
    return t
CARDS = [('Mất vật thể', 'Cây thành mảng lá dán, xe thành tấm biển trắng.'),
         ('Sai số lượng', 'Vẽ 3 bò, 1 xe → ra 2 bò, 2 xe.'),
         ('Hòa trộn vật thể', 'Bò 1 và xe tải dính thành một khối máy đỏ.'),
         ('Sai vị trí', 'Chỗ vẽ bò 3 lại là một chiếc xe tải.'),
         ('Rò rỉ thuộc tính', 'Màu và chất đất ruộng lan sang cây: cây thành đống đất.')]
cards = []
for i, (t, d) in enumerate(CARDS):
    x = 0.6 + i * 2.45
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(x), I(5.4), I(2.33), I(0.98))
    c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xFB, 0xE9, 0xE9); c.line.color.rgb = RED; c.line.width = Pt(1); c.shadow.inherit = False
    c.adjustments[0] = 0.08
    tb(0, 0, 0, 0, [[(f'{i + 1}  ', True, RED, 15), (t, True, INK, 13)], [(d, False, INK, 10.5)]], anchor=MSO_ANCHOR.MIDDLE, shape=c)
    c.text_frame.margin_left = c.text_frame.margin_right = I(0.12)
    cards.append(c)
steps = [
    [cards[0], ell('zhang', 0.22, 0.46, 0.62, 0.88), ell('zhang', 0.04, 0.46, 0.27, 0.63), tag('zhang', 0.03, 0.03, 'mất cây, mất xe', 1.3)],
    [cards[1], ell('t2i', 0.03, 0.43, 0.33, 0.65), ell('t2i', 0.6, 0.66, 0.93, 0.9), tag('t2i', 0.03, 0.03, '2 xe (vẽ 1), 2 bò (vẽ 3)', 1.85)],
    [cards[2], ell('controlnet', 0.04, 0.32, 0.36, 0.64), tag('controlnet', 0.03, 0.2, 'bò 1 + xe dính nhau', 1.55)],
    [cards[3], ell('sketch', 0.6, 0.62, 0.97, 0.93), tag('sketch', 0.55, 0.52, 'bò 3 vẽ ở đây', 1.1),
     ell('t2i', 0.58, 0.64, 0.95, 0.92), tag('t2i', 0.3, 0.84, 'ra xe tải', 0.8)],
    [cards[4], ell('controlnet', 0.27, 0.46, 0.55, 0.88), tag('controlnet', 0.3, 0.88, 'cây thành đống đất', 1.45)]]
s.notes_slide.notes_text_frame.text = (
    'Cùng một phác thảo và câu mô tả “three small cows, a tall tree and a red truck on a farm” như slide trước, đưa qua ba phương pháp hiện có; '
    'đây là ảnh thật trong thực nghiệm, cùng hạt giống. Mỗi click chỉ ra một kiểu lỗi. '
    'Một, mất vật thể: Zhang và cộng sự vẽ cây thành một mảng lá dán, xe thành tấm biển trắng, nên mất cây và xe. '
    'Hai, sai số lượng: T2I-Adapter vẽ ba bò, một xe thành hai bò, hai xe. '
    'Ba, hòa trộn: ControlNet làm con bò thứ nhất và chiếc xe tải dính thành một khối máy màu đỏ. '
    'Bốn, sai vị trí: ở chỗ phác thảo vẽ con bò thứ ba, T2I-Adapter lại đặt một chiếc xe tải. '
    'Năm, rò rỉ thuộc tính: màu và chất của đất ruộng lan sang cái cây, cây thành đống đất. '
    'Nguyên nhân chung là mô hình không biết nét nào thuộc vật thể nào.')
if state:
    for st in steps[state:]:
        for x in st: x._element.getparent().remove(x._element)
    ids = p.slides._sldIdLst
    for i, sid in reversed(list(enumerate(list(ids)))):
        if i != 4: p.part.drop_rel(sid.rId); ids.remove(sid)
else:
    add_clicks(s, [[(x, 'zoom' if x.auto_shape_type == MSO_SHAPE.OVAL else 'fade') for x in st] for st in steps])
p.save(OUT); print('ok', state)
