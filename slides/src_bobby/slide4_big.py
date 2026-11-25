"""Bobby 2026-11-26: slide 4 bottom is empty -> text on top full width, the 4 example images in one big row below.
Usage: slide4_big.py IN OUT"""
import sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
p = Presentation(sys.argv[1]); s = p.slides[3]; S = {x.name: x for x in s.shapes}
def put(x, l, t, w, h): x.left, x.top, x.width, x.height = I(l), I(t), I(w), I(h)
def size(x, pt):
    for para in x.text_frame.paragraphs:
        for r in para.runs: r.font.size = Pt(pt)
put(S['Text 4'], 0.6, 1.55, 12.1, 1.75)
z, gap, y = 2.75, (12.1 - 4 * 2.75) / 3, 3.5
for k in range(4):
    x = 0.6 + k * (z + gap)
    put(S[f'TextBox {6 + k}'], x, y - 0.34, z, 0.3); size(S[f'TextBox {6 + k}'], 12)
    put(S[f'Picture {10 + k}'], x, y, z, z)
    put(S[f'TextBox {14 + k}'], x, y + z + 0.05, z, 0.42); size(S[f'TextBox {14 + k}'], 10.5)
for para in S['Text 4'].text_frame.paragraphs:
    for r in para.runs: r.text = r.text.replace('ví dụ bên phải', 'ví dụ bên dưới')
p.save(sys.argv[2]); print('ok')
