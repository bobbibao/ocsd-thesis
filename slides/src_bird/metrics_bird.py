"""Bobby 2026-11-26: metrics slide pictures on the same real example as M1-M5 (2 gray birds + 1 tree, final OCSD).
Boxes: M1 sketch boxes (drawn) vs boxes found in the final image. Usage (run here): metrics_bird.py IN OUT"""
import json, sys
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
M = json.load(open('img/trace.json'))
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
base = Image.open('img/final.png').convert('RGB').resize((512, 512))
dets = M['m5d']['checks'][0]['detections']; objs = M['m1']['objects']
GRN, BLU, ORG, WHT = (40, 200, 90), (60, 120, 255), (235, 110, 40), (255, 255, 255)
NM = {'bird': 'chim', 'tree': 'cây'}
def lab(d, x, y, t, col):
    w = d.textlength(t, font=F); x = min(max(0, x), 512 - w - 10); y = min(max(0, y), 512 - 28)
    d.rectangle([x, y, x + w + 10, y + 28], fill=col); d.text((x + 5, y + 2), t, font=F, fill=WHT)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)
# one-to-one match per class (greedy by IoU)
pairs = sorted(((iou(o['box'], x['box']), i, j) for i, o in enumerate(objs) for j, x in enumerate(dets) if x['cls'] == o['cls']), reverse=True)
match, used = {}, set()
for v, i, j in pairs:
    if i not in match and j not in used: match[i] = (j, v); used.add(j)
im = base.copy(); d = ImageDraw.Draw(im)
for det in dets:
    b = det['box']; d.rectangle(b, outline=GRN, width=5); lab(d, b[0], b[1] - 30, NM[det['cls']] + ' ✓', GRN)
im.save('figs/m_opr.png')
im = base.copy(); d = ImageDraw.Draw(im)
for o in objs:
    b = o['box']
    for k in range(0, int(b[2] - b[0]), 16): d.line([b[0] + k, b[1], min(b[0] + k + 8, b[2]), b[1]], fill=BLU, width=4); d.line([b[0] + k, b[3], min(b[0] + k + 8, b[2]), b[3]], fill=BLU, width=4)
    for k in range(0, int(b[3] - b[1]), 16): d.line([b[0], b[1] + k, b[0], min(b[1] + k + 8, b[3])], fill=BLU, width=4); d.line([b[2], b[1] + k, b[2], min(b[1] + k + 8, b[3])], fill=BLU, width=4)
for det in dets: d.rectangle(det['box'], outline=GRN, width=5)
IOU = []
for i, o in enumerate(objs):
    v = match[i][1]; IOU.append(v); b = o['box']; lab(d, b[0] + 4, b[1] + 4, 'IoU ' + f'{v:.2f}'.replace('.', ','), ORG)
lab(d, 8, 470, 'xanh dương: hộp đã vẽ', BLU); lab(d, 8, 436, 'xanh lá: hộp tìm thấy', GRN)
im.save('figs/m_miou.png')
im = base.copy(); d = ImageDraw.Draw(im)
c = [((dets[match[i][0]]['box'][0] + dets[match[i][0]]['box'][2]) / 2, (dets[match[i][0]]['box'][1] + dets[match[i][0]]['box'][3]) / 2) for i in range(3)]
for a, b_, t in [(0, 1, 'bên phải ✓'), (1, 2, 'ở dưới ✓')]:
    d.line([c[a], c[b_]], fill=ORG, width=6)
    mx, my = (c[a][0] + c[b_][0]) / 2, (c[a][1] + c[b_][1]) / 2; lab(d, mx - 60, my - 14, t, ORG)
for q in c: d.ellipse([q[0] - 9, q[1] - 9, q[0] + 9, q[1] + 9], fill=ORG)
im.save('figs/m_ra.png')
f2 = lambda v: f'{v:.2f}'.replace('.', ',')
# order shown in the formula: tree, lower bird, upper bird
p = Presentation(sys.argv[1])
s = next(x for x in p.slides if any(q.name == 'Text 1' and q.text_frame.text.startswith('Cách đo') for q in x.shapes))
S = {x.name: x for x in s.shapes}
for nm, f in (('Picture 22', 'figs/m_opr.png'), ('Picture 29', 'figs/m_miou.png'), ('Picture 36', 'figs/m_ra.png')):
    old = S[nm]; new = s.shapes.add_picture(f, old.left, old.top, old.width, old.height)
    old._element.addprevious(new._element); old._element.getparent().remove(old._element)
mi = sum(IOU) / 3
r = S['TextBox 31'].text_frame.paragraphs[1].runs; r[0].text = f'mIoU = ({f2(IOU[0])} + {f2(IOU[1])} + {f2(IOU[2])}) ÷ 3 = {f2(mi)}'
for q in r[1:]: q.text = ''
r = S['TextBox 35'].text_frame.paragraphs[0].runs; r[0].text = 'R4 · tương quan'
for q in r[1:]: q.text = ''
s.notes_slide.notes_text_frame.text = (
    'Nhóm dùng ba độ đo, cũng là ba độ đo trong các slide kết quả. Mọi ảnh được chấm tự động bằng OWLv2, một bộ phát hiện độc lập, khác bộ dùng bên trong OCSD, '
    'để phương pháp không “học tủ” theo công cụ chấm. Hình minh họa dùng chính cảnh ví dụ ở phần phương pháp: một cây và hai con chim. '
    'Thứ nhất, giữ vật thể, OPR: trong các vật đã vẽ, bao nhiêu vật được tìm thấy đúng loại và gần đúng chỗ. Vẽ 3 vật, tìm thấy đủ 3, nên 100 phần trăm. '
    f'Thứ hai, vị trí và hình dạng, mIoU. Với mỗi vật, lấy phần hai hộp chồng lên nhau chia cho phần gộp của hai hộp, gọi là IoU, từ 0 đến 1. '
    f'Ở cảnh ví dụ: cây được {f2(IOU[0])} vì cây vẽ ra thấp hơn nét phác thảo, chim dưới {f2(IOU[1])}, chim trên {f2(IOU[2])}. Trung bình ba số là {f2(mi)}. '
    'Thứ ba, đúng tương quan, RA: các cặp vật có giữ đúng trái phải, trên dưới như phác thảo không; ví dụ cây bên phải chim dưới, chim dưới ở dưới chim trên. Cả 6 cặp đều đúng. '
    'Khi so sánh các phương pháp, nhóm dùng kiểm định thống kê cặp, hiệu chỉnh Holm.')
p.save(sys.argv[2]); print('ok', [round(v, 3) for v in IOU], round(mi, 3))
