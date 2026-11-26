"""v40 (Bobby 2026-11-26): every slide without a transition gets a plain fade; content slides without click
animations get 1-2 clicks that reveal one paired block at a time (left/right, top/bottom). The first block
is visible on entry. Slides that already have animation (4, 5, 6, 13-17, 38) are untouched; slide 18 keeps
no animation (Bobby asked to remove it there).
Usage: anim_all.py IN OUT"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_bobby'))
from pptx import Presentation
from anim_lib import add_clicks, set_transition
P = '{http://schemas.openxmlformats.org/presentationml/2006/main}'
IN = 914400
# slide number -> function(x, y) in inches (shape left/top) returning group 0 (static), 1, 2 (click order)
RULES = {
    2:  lambda x, y: 1 if x >= 8.0 else 0,
    7:  lambda x, y: 1 if y >= 5.4 else 0,
    9:  lambda x, y: 1 if x >= 6.6 else 0,
    10: lambda x, y: 1 if y >= 4.0 else 0,
    11: lambda x, y: 1 if x >= 6.6 else 0,
    20: lambda x, y: 1 if y >= 4.7 else 0,
    21: lambda x, y: 1 if y >= 5.4 else 0,
    24: lambda x, y: 1 if (x >= 6.6 or y >= 6.1) else 0,
    25: lambda x, y: 2 if y >= 5.0 else (1 if x >= 6.6 else 0),
    26: lambda x, y: 1 if y >= 4.9 else 0,
    27: lambda x, y: 0 if y >= 6.6 else (2 if x >= 9.6 else (1 if x >= 6.6 else 0)),
    28: lambda x, y: 1 if y >= 4.3 else 0,
    30: lambda x, y: 2 if y >= 5.5 else (1 if x >= 6.6 else 0),
    31: lambda x, y: 0 if y >= 6.5 else (2 if x >= 8.7 else (1 if x >= 4.6 else 0)),
    32: lambda x, y: 1 if x >= 8.3 else 0,
    33: lambda x, y: 1 if x >= 8.3 else 0,
    34: lambda x, y: 0 if y >= 6.45 else (1 if x >= 6.6 else 0),
    35: lambda x, y: 1 if x >= 10.4 else 0,
    36: lambda x, y: 1 if x >= 8.0 else (2 if y >= 4.5 else 0),
    39: lambda x, y: 1 if y >= 5.6 else 0,
}
p = Presentation(sys.argv[1]); report = []
for n, s in enumerate(p.slides, 1):
    if n == 18: continue
    if s._element.find('.//' + P + 'transition') is None: set_transition(s, 'fade')
    if n not in RULES: continue
    assert s._element.find(P + 'timing') is None or not list(s._element.find(P + 'timing').iter(P + 'par')), n
    groups = [[], [], []]
    for sh in s.shapes:
        x, y = sh.left / IN, sh.top / IN
        if y < 1.2 or y >= 7.0: continue          # header, title, module tracker, footer
        groups[RULES[n](x, y)].append(sh)
    steps = [[(sh, 'fade') for sh in g] for g in groups[1:] if g]
    add_clicks(s, steps)
    report.append((n, len(groups[0]), [len(g) for g in groups[1:]]))
p.save(sys.argv[2])
for r in report: print(r)
