import json
from PIL import Image, ImageDraw
R = {c: [json.loads(l) for l in open(f'{c}.ndjson')] for c in ('cow', 'tree', 'truck')}
def place(g, d, box, flip=False, w=4):
    xs = [x for s in d['drawing'] for x in s[0]]; ys = [y for s in d['drawing'] for y in s[1]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys); bw, bh = box[2] - box[0], box[3] - box[1]
    k = min(bw / max(1, x1 - x0), bh / max(1, y1 - y0)); ox = box[0] + (bw - k * (x1 - x0)) / 2; oy = box[1] + (bh - k * (y1 - y0)) / 2
    for s in d['drawing']:
        pts = [((x1 - x if flip else x - x0) * k + ox, (y - y0) * k + oy) for x, y in zip(s[0], s[1])]
        if len(pts) > 1: g.line(pts, fill=0, width=w, joint='curve')
LAYOUTS = {
 'A': [('cow', 33, (40, 70, 200, 185), False), ('cow', 39, (320, 70, 480, 185), True), ('truck', 38, (30, 290, 215, 400), False),
       ('tree', 33, (225, 170, 330, 420), False), ('cow', 83, (345, 290, 495, 420), False)],
 'B': [('cow', 49, (35, 60, 195, 175), False), ('cow', 7, (320, 60, 480, 175), True), ('truck', 66, (30, 300, 215, 400), False),
       ('tree', 27, (220, 160, 335, 420), False), ('cow', 33, (345, 290, 495, 415), True)]}
for name, objs in LAYOUTS.items():
    im = Image.new('L', (512, 512), 255); g = ImageDraw.Draw(im)
    for c, i, box, fl in objs: place(g, R[c][i], box, fl)
    im.save(f'farm_clean_{name}.png')
a, b = Image.open('farm_clean_A.png'), Image.open('farm_clean_B.png'); s = Image.new('L', (1034, 512), 128); s.paste(a, (0, 0)); s.paste(b, (522, 0)); s.save('both.png')
