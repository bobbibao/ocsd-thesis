"""Relation figure for the farm scene (M1 output): sketch in grey, M1 boxes, lines between object centres with a
numbered badge; the legend (what each number means) is native slide text. Relations are from trace.json m1.relations.
Usage: rel_fig.py SKETCH.png OUT.png"""
import sys
from PIL import Image, ImageDraw, ImageFont, ImageOps
S = 3
B = {0: (40, 170, 176, 268), 1: (137, 255, 286, 412), 2: (35, 242, 157, 317), 3: (307, 179, 419, 261), 4: (322, 346, 482, 457)}
COL = {0: (217, 105, 43), 1: (46, 90, 156), 2: (46, 139, 87), 3: (142, 68, 173), 4: (201, 162, 0)}
# (from, to, badge position in 512 space): 1 truck below cow1, 2 tree right of truck, 3 cow2 above cow3
EDGES = [(2, 0, (70, 365)), (2, 1, (205, 300)), (3, 4, (395, 305))]
sk = ImageOps.grayscale(Image.open(sys.argv[1]).convert('RGB')).resize((512 * S, 512 * S))
sk = sk.point(lambda v: 255 - (255 - v) * 0.3)
im = Image.merge('RGB', [sk] * 3); d = ImageDraw.Draw(im)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 40 * S)
c = lambda k: ((B[k][0] + B[k][2]) / 2 * S, (B[k][1] + B[k][3]) / 2 * S)
for k, b in B.items():
    d.rectangle([v * S for v in b], outline=COL[k], width=5 * S)
DARK = (31, 36, 48)
for n, (a, b, (bx, by)) in enumerate(EDGES, 1):
    (x0, y0), (x1, y1) = c(a), c(b)
    d.line([x0, y0, x1, y1], fill=DARK, width=6 * S)
    for (x, y) in ((x0, y0), (x1, y1)):
        d.ellipse([x - 10 * S, y - 10 * S, x + 10 * S, y + 10 * S], fill=DARK)
    d.line([(x0 + x1) / 2, (y0 + y1) / 2, bx * S, by * S], fill=DARK, width=3 * S)
    r = 30 * S; X, Y = bx * S, by * S
    d.ellipse([X - r, Y - r, X + r, Y + r], fill=(217, 105, 43), outline='white', width=4 * S)
    w = d.textlength(str(n), font=font); d.text((X - w / 2, Y - 25 * S), str(n), font=font, fill='white')
im.resize((768, 768), Image.LANCZOS).save(sys.argv[2]); print('ok')
