"""v41 (Bobby 2026-11-26, page 16): slide 9 drops the 'Ràng buộc' box and the module tags (M1–M3), (M4)
in the questions (slide 38 keeps its wording). Notes follow.
Usage: slide9_noconstraint.py IN OUT"""
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); s = p.slides[8]
for x in list(s.shapes):
    if x.name in ('Rounded Rectangle 29', 'TextBox 30'): x._element.getparent().remove(x._element)
for x in s.shapes:
    if x.name in ('Text 21', 'Text 24'):
        for r in x.text_frame.paragraphs[0].runs: r.text = r.text.replace(' (M1–M3)', '').replace(' (M4)', '')
nt = s.notes_slide.notes_text_frame; t = nt.text
for a, b in [(' Kèm một ràng buộc: giữ vật thể nhưng ảnh không được kém tự nhiên hơn, đo bằng PickScore và chấm mù.', ''),
             (', tức M1 đến M3,', ''), (' ở M4', '')]:
    assert a in t, a; t = t.replace(a, b)
nt.text = t
assert 'Ràng buộc' not in ''.join(x.text_frame.text for x in s.shapes if x.has_text_frame)
p.save(sys.argv[2]); print('ok')
