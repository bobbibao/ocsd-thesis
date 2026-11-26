"""One PPTX per click state of a slide (click order and entrance/exit read from the slide timing).
Usage: state_pages.py DECK SLIDE_NO OUT_PREFIX -> OUT_PREFIX_k.pptx = slide after click k (k = 1..n)"""
import sys
from pptx import Presentation
P = '{http://schemas.openxmlformats.org/presentationml/2006/main}'
deck, no, pre = sys.argv[1], int(sys.argv[2]), sys.argv[3]
def effects(sl):
    out, k = [], 0
    for c in sl._element.iter(P + 'cTn'):
        nt = c.get('nodeType')
        if nt == 'clickEffect': k += 1
        if nt in ('clickEffect', 'withEffect', 'afterEffect'):
            for t in c.iter(P + 'spTgt'): out.append((k, c.get('presetClass'), int(t.get('spid'))))
    return out, k
_, n = effects(Presentation(deck).slides[no - 1])
for k in range(1, n + 1):
    p = Presentation(deck); sl = p.slides[no - 1]; ef, _ = effects(sl)
    hide = {sp for (c, cls, sp) in ef if cls == 'entr' and c > k} | {sp for (c, cls, sp) in ef if cls == 'exit' and c <= k}
    for x in list(sl.shapes):
        if x.shape_id in hide: x._element.getparent().remove(x._element)
    lst = p.slides._sldIdLst
    for i, sid in enumerate(list(lst)):
        if i != no - 1: lst.remove(sid)
    p.save(f'{pre}_{k}.pptx')
print(n)
