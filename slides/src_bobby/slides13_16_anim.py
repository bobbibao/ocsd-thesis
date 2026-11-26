"""Bobby 2026-11-26: background slides 13-16 more lively, with animations and transitions.
- New slide before 13 ('13a'): only the Diffusion Model part, large. Slide 13 then MORPHS from it: the Diffusion part
  shrinks into place; the Stable Diffusion part appears only on click (heading, then weakness -> fix rows 1, 2, 3).
- Slides 14, 15: push transition; the noise steps appear click by click, then the lower boxes.
- Slide 16 (recap of 13 with SD highlighted): fade, the highlight appears by itself.
- Footer page numbers 'k / N' renumbered.
Run LAST in the chain (after any image swaps). Usage: slides13_16_anim.py IN OUT"""
import copy, sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from anim_lib import add_clicks, set_transition

p = Presentation(sys.argv[1])
src = p.slides[12]
assert src.shapes.title is None or True
S13 = {x.name: x for x in src.shapes}
assert 'SDWEAK h1' in S13 and 'Picture 21' in S13, 'slide 13 is not the Diffusion/SD slide'

# ---- duplicate slide 13 -> 13a, placed before it
new = p.slides.add_slide(src.slide_layout)
for el in list(new.shapes._spTree):
    if el.tag.endswith('}sp') or el.tag.endswith('}pic') or el.tag.endswith('}grpSp') or el.tag.endswith('}cxnSp'):
        new.shapes._spTree.remove(el)
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
for el in src.shapes._spTree:
    if el.tag.endswith(('}sp', '}pic', '}grpSp', '}cxnSp', '}graphicFrame')):
        c = copy.deepcopy(el)
        for node in c.iter():
            for att in (R + 'embed', R + 'link', R + 'id'):
                rid = node.get(att)
                if rid and rid in src.part.rels:
                    rel = src.part.rels[rid]
                    node.set(att, new.part.relate_to(rel._target, rel.reltype))
        new.shapes._spTree.append(c)
ids = p.slides._sldIdLst
el = ids[-1]; ids.remove(el); ids.insert(12, el)
A = {x.name: x for x in new.shapes}
for nm in [n for n in A if n.startswith('SDWEAK') or n in ('TextBox 28', 'Connector 29', 'TextBox 30', 'Connector 25')]:
    A[nm]._element.getparent().remove(A[nm]._element)
def put(x, l, t, w, h): x.left, x.top, x.width, x.height = I(l), I(t), I(w), I(h)
def size(x, pt):
    for pa in x.text_frame.paragraphs:
        for r in pa.runs: r.font.size = Pt(pt)
put(A['TextBox 22'], 0.6, 1.6, 12.1, 0.42); size(A['TextBox 22'], 16)
put(A['Connector 23'], 0.6, 2.1, 12.1, 0)
put(A['TextBox 24'], 0.6, 2.35, 6.0, 3.3); size(A['TextBox 24'], 19)
put(A['TextBox 31'], 0.6, 5.4, 6.0, 1.2); size(A['TextBox 31'], 15.5)
put(A['Picture 21'], 6.9, 3.0, 5.8, 5.8 / (6.55 / 2.12))
set_transition(new, 'fade')

# ---- slide 13 (now index 13): morph in, SD part on click
s13 = p.slides[13]; B = {x.name: x for x in s13.shapes}
set_transition(s13, 'morph')
row = lambda k: [(B[f'SDWEAK n{k}'], 'zoom'), (B[f'SDWEAK wbg{k}'], 'fade'), (B[f'SDWEAK w{k}'], 'fade'),
                 (B[f'SDWEAK arr{k}'], 'wipe'), (B[f'SDWEAK sbg{k}'], 'wipe'), (B[f'SDWEAK s{k}'], 'wipe'), (B[f'SDWEAK l{k}'], 'fade')]
add_clicks(s13, [
    [(B['Connector 25'], 'wipe'), (B['TextBox 28'], 'wipe'), (B['Connector 29'], 'wipe'), (B['TextBox 30'], 'fade'), (B['SDWEAK idea'], 'fade')],
    [(B['SDWEAK h1'], 'fade'), (B['SDWEAK h2'], 'fade'), (B['SDWEAK h3'], 'fade')] + row(0),
    row(1), row(2)])

# ---- slides 14, 15: noise steps one click at a time
def by(slide):
    return {x.name: x for x in slide.shapes}
s14 = p.slides[14]; C = by(s14); set_transition(s14, 'push')
st = lambda a: [(C[n], 'wipe' if n.startswith('Picture') else 'fade') for n in a]
add_clicks(s14, [
    st(['TextBox 36', 'Picture 37', 'TextBox 38', 'TextBox 39', 'Picture 40', 'TextBox 41']),
    st(['TextBox 42', 'Picture 43', 'TextBox 44', 'TextBox 45', 'Picture 46', 'TextBox 47']),
    st(['TextBox 48', 'Picture 49', 'TextBox 50', 'TextBox 51', 'Picture 52', 'TextBox 53']),
    st(['TextBox 54', 'Picture 55', 'TextBox 56', 'TextBox 57', 'Picture 58', 'TextBox 59', 'TextBox 60', 'Picture 61', 'TextBox 62', 'TextBox 63']),
    st(['Rounded Rectangle 64', 'TextBox 65', 'Picture 66', 'TextBox 67', 'Right Arrow 68', 'Rounded Rectangle 69', 'TextBox 70',
        'Right Arrow 71', 'Picture 72', 'TextBox 73', 'TextBox 74', 'TextBox 75'])])
s15 = p.slides[15]; C = by(s15); set_transition(s15, 'push')
add_clicks(s15, [
    st(['TextBox 36', 'Rounded Rectangle 54', 'TextBox 55', 'Picture 37', 'TextBox 38', 'TextBox 39', 'Picture 40', 'TextBox 41']),
    st(['TextBox 42', 'Rounded Rectangle 56', 'TextBox 57', 'Picture 43', 'TextBox 44', 'TextBox 45', 'Picture 46', 'TextBox 47']),
    st(['TextBox 48', 'Rounded Rectangle 58', 'TextBox 59', 'Picture 49', 'TextBox 50', 'TextBox 51', 'Picture 52', 'TextBox 53', 'TextBox 60']),
    st(['Rounded Rectangle 61', 'TextBox 62'] + [f'Picture {k}' if k % 2 else f'TextBox {k}' for k in range(63, 75)] + ['TextBox 75']),
    st(['Rounded Rectangle 76', 'TextBox 77', 'Picture 78', 'TextBox 79', 'Picture 80', 'Picture 81', 'TextBox 82', 'Picture 83',
        'Picture 84', 'TextBox 85', 'Picture 86', 'TextBox 87'])])

# ---- slide 16: recap, highlight comes in by itself
s16 = p.slides[16]; D = by(s16); set_transition(s16, 'fade')
add_clicks(s16, [[(D['Rectangle 33'], 'fade'), (D['Rounded Rectangle 34'], 'zoom')]], after_prev=True)

# slide 16 note pointed at the SD slide, which is now one later
for x in s15.shapes:
    if x.has_text_frame:
        for pa in x.text_frame.paragraphs:
            for r in pa.runs: r.text = r.text.replace('(slide 13)', '(slide 14)')

# ---- renumber 'k / N' footers
N = len(p.slides)
for k, sl in enumerate(p.slides, 1):
    for x in sl.shapes:
        if x.name == 'Text 3' and x.has_text_frame and '/' in x.text_frame.text and x.text_frame.text.split('/')[0].strip().isdigit():
            r = x.text_frame.paragraphs[0].runs
            r[0].text = f'{k} / {N}'
            for q in r[1:]: q.text = ''
p.save(sys.argv[2]); print('ok', N, 'slides')
