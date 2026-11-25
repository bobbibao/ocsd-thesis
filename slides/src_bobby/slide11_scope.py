"""Bobby 2026-11-26: slide 11 too empty -> bigger text, more air, same two cards. Usage: slide11_scope.py IN OUT"""
import sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
p = Presentation(sys.argv[1]); s = p.slides[10]; S = {x.name: x for x in s.shapes}
for nm in ('Shape 4', 'Shape 7'): S[nm].top, S[nm].height = I(1.6), I(5.1)
for nm in ('Text 5', 'Text 8'):
    S[nm].top, S[nm].height = I(1.85), I(0.5)
    for r in S[nm].text_frame.paragraphs[0].runs: r.font.size = Pt(22)
for nm in ('Text 6', 'Text 9'):
    S[nm].top, S[nm].height = I(2.6), I(3.9)
    for para in S[nm].text_frame.paragraphs:
        para.space_after = Pt(16)
        for r in para.runs:
            r.font.size = Pt(19); r.text = r.text.replace('vật thể;', 'vật thể.')
p.save(sys.argv[2]); print('ok')
