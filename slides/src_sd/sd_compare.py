"""v43 (Bobby 2026-11-26, page 25): drop the compact slide 18; on slide 17 redesign the diffusion-vs-SD comparison:
each row = problem -> how SD solves it -> result chip; column 'Ở SLIDE SAU' removed. Footers renumbered N / 40.
Clicks on slide 17: 1 = SD intro, 2-4 = one row each.
Usage: sd_compare.py IN OUT"""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_bobby'))
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from anim_lib import add_clicks
I = Inches
p = Presentation(sys.argv[1]); s = p.slides[16]
sh = {x.name: x for x in s.shapes}
assert 'SDWEAK h3' in sh
for x in list(s.shapes):
    if x.name.startswith('SDWEAK') and x.name != 'SDWEAK idea': x._element.getparent().remove(x._element)

def box(x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None):
    b = s.shapes.add_shape(shape, I(x), I(y), I(w), I(h))
    b.fill.solid(); b.fill.fore_color.rgb = RGBColor.from_string(fill)
    if line: b.line.color.rgb = RGBColor.from_string(line); b.line.width = Pt(1.25)
    else: b.line.fill.background()
    b.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: b.adjustments[0] = 0.12
    return b

def txt(b, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, m=0.1):
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = I(m); tf.margin_top = tf.margin_bottom = I(0.02)
    for k, runs in enumerate(paras):
        pa = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); pa.alignment = align
        for t, bold, col, size in runs:
            r = pa.add_run(); r.text = t; f = r.font; f.bold = bold; f.size = Pt(size); f.name = 'Arial'
            f.color.rgb = RGBColor.from_string(col)
    return b

RED, BLUE, DARK, GREEN, OR = 'B23434', '2E5A9C', '1F2430', '2E7D32', 'D9692B'
head = []
for x, w, t, col in ((4.7, 3.3, 'VẤN ĐỀ CỦA DIFFUSION GỐC', RED), (8.35, 3.05, 'STABLE DIFFUSION GIẢI QUYẾT', BLUE),
                     (11.55, 1.2, 'KẾT QUẢ', GREEN)):
    tb = s.shapes.add_textbox(I(x), I(4.5), I(w), I(0.3)); txt(tb, [[(t, True, col, 10)]], m=0.02); head.append(tb)
ROWS = [('Chậm, tốn bộ nhớ', 'khử nhiễu trên ảnh gốc 512 × 512 × 3 ≈ 786 nghìn số',
         'Nén ảnh bằng VAE', 'khử nhiễu trên ảnh nén 64 × 64 × 4 ≈ 16 nghìn số', 'nhỏ hơn\n~48 lần'),
        ('Không điều khiển bằng chữ', 'tạo ảnh ngẫu nhiên, không nhận câu lệnh (Ho và cs., 2020)',
         'CLIP + cross-attention', 'câu lệnh thành vector, dẫn hướng U-Net (chú ý chéo)', 'ảnh theo\ncâu lệnh'),
        ('Cần rất nhiều bước', '~1000 bước khử nhiễu cho mỗi ảnh',
         'DDIM + CFG', 'chỉ ~30–50 bước (Song và cs., 2021); CFG giúp bám câu lệnh', '1000 →\n30–50 bước')]
rows = []
for i, (pt, pd, st, sd, res) in enumerate(ROWS):
    y, h = 4.83 + i * 0.72, 0.66; g = []
    n = box(4.35, y + h / 2 - 0.15, 0.3, 0.3, OR, MSO_SHAPE.OVAL)
    txt(n, [[(str(i + 1), True, 'FFFFFF', 12)]], align=PP_ALIGN.CENTER, m=0); g.append(n)
    g.append(txt(box(4.7, y, 3.3, h, 'FDECE1'), [[(pt, True, RED, 11.5)], [(pd, False, DARK, 10)]]))
    g.append(box(8.05, y + h / 2 - 0.14, 0.25, 0.28, OR, MSO_SHAPE.RIGHT_ARROW))
    g.append(txt(box(8.35, y, 3.05, h, 'E6EEFA'), [[(st, True, BLUE, 11.5)], [(sd, False, DARK, 10)]]))
    g.append(box(11.42, y + h / 2 - 0.1, 0.11, 0.2, GREEN, MSO_SHAPE.RIGHT_ARROW))
    g.append(txt(box(11.55, y, 1.2, h, 'E3F1E6', line=GREEN),
                 [[(l, True, GREEN, 10.5)] for l in res.split('\n')], align=PP_ALIGN.CENTER, m=0.03))
    rows.append(g)
intro = [sh[n] for n in ('Connector 25', 'TextBox 28', 'Connector 29', 'TextBox 30', 'SDWEAK idea')]
add_clicks(s, [[(x, 'fade') for x in intro + head]] + [[(x, 'fade') for x in g] for g in rows])

# delete slide 18 (compact copy)
s18 = p.slides[17]
assert 'Diffusion Model và Stable Diffusion' in ''.join(x.text_frame.text for x in s18.shapes if x.has_text_frame)
lst = p.slides._sldIdLst; sid = lst[17]
p.part.drop_rel(sid.rId); lst.remove(sid)
tot = len(p.slides)
for k, sl in enumerate(p.slides, 1):
    for x in sl.shapes:
        if x.has_text_frame and re.fullmatch(r'\d+ / \d+', x.text_frame.text.strip()):
            r = x.text_frame.paragraphs[0].runs; r[0].text = f'{k} / {tot}'
            for e in r[1:]: e._r.getparent().remove(e._r)
p.save(sys.argv[2]); print('ok', tot)
