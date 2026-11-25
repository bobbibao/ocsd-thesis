"""Bobby 2026-11-26 (answered 'ok'): fix the architecture diagram on slide 21:
(1) M5(d) loop: final image -> check -> redraw the faulty region in M4 (<= 2 times);
(2) M5 (a)-(c) act during all denoising steps -> arrows into both denoising boxes (1-24 and 25-30);
(3) the M3 tokens are in the prompt from step 1 -> token arrow also into the 1-24 box, box texts fixed.
Patches the raster in arch/arch_in.png (canvas widened 120 px for the loop). Usage: slide21_arch.py IN OUT"""
import math, sys
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
R = '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'
f = lambda n: ImageFont.truetype(R, n)
src = Image.open('arch/arch_in.png').convert('RGB'); W0, H0 = src.size; EXT = 120
im = Image.new('RGB', (W0 + EXT, H0), 'white'); im.paste(src, (0, 0)); d = ImageDraw.Draw(im)
BOX, INK, PUR, GRY = (220, 232, 247), (25, 25, 25), (112, 48, 160), (50, 50, 50)
def arrow(pts, col, w=5, head=16):
    d.line(pts, fill=col, width=w, joint='curve')
    (x1, y1), (x2, y2) = pts[-2], pts[-1]; a = math.atan2(y2 - y1, x2 - x1)
    d.polygon([(x2, y2), (x2 - head * math.cos(a - .45), y2 - head * math.sin(a - .45)), (x2 - head * math.cos(a + .45), y2 - head * math.sin(a + .45))], fill=col)
def centre(box, lines, size=21, col=INK):
    x0, y0, x1, y1 = box; d.rectangle(box, fill=BOX); F = f(size); lh = size + 5
    y = (y0 + y1) / 2 - lh * len(lines) / 2
    for t in lines:
        w = d.textlength(t, font=F); d.text(((x0 + x1) / 2 - w / 2, y), t, font=F, fill=col); y += lh
# box texts (interiors only, borders kept)
centre((1274, 585, 1665, 805), ['M4  Giữ vật (bước 1–12)', 'rồi nhả dần (13–24)', 'một quỹ đạo chung cho vật và nền', 'câu lệnh chứa token ‹o₀›, ‹o₁›, …'])
centre((1734, 585, 2135, 805), ['M4  Tự do (bước 25–30)', 'vật và nền vẽ tự do,', 'vẫn dùng câu lệnh có token'])
# (3) token arrow branch into the 1-24 box
arrow([(1652, 470), (1590, 574)], GRY, w=4, head=15)
# (2) M5 acts during all steps: second purple arrow into the 1-24 box; recolour the existing one purple
d.rectangle((1915, 816, 1955, 866), fill=(242, 246, 252))
for x in (1470, 1934): arrow([(x, 866), (x, 818)], PUR, w=5, head=16)
d.text((1500, 826), '(a)–(c) ở mọi bước', font=f(19), fill=PUR)
# (1) M5(d) loop: image -> check -> back into M4
arrow([(2162, 300), (W0 + 70, 300), (W0 + 70, 700), (2146, 700)], PUR, w=5, head=18)
F = f(19)
# label on the free space right of the up-arrow, above box 3
for k, t in enumerate(['(d) kiểm tra ảnh: thiếu / lệch', '→ vẽ lại vùng lỗi (≤ 2 lần)']):
    d.rectangle((1996, 446 + 23 * k, 2004 + d.textlength(t, font=F), 468 + 23 * k), fill='white'); d.text((2000, 446 + 23 * k), t, font=F, fill=PUR)
im.save('arch/arch_out.png')
p = Presentation(sys.argv[1]); s = p.slides[20]
old = next(x for x in s.shapes if x.shape_type == 13)
w = int(old.height * im.size[0] / im.size[1]); left = int(old.left - (w - old.width) / 2)
new = s.shapes.add_picture('arch/arch_out.png', left, old.top, w, old.height)
old._element.addprevious(new._element); old._element.getparent().remove(old._element)
tf = s.notes_slide.notes_text_frame
tf.text = ('Đây là kiến trúc tổng thể. M1 tách phác thảo và văn bản thành thông tin từng vật thể: lớp, mặt nạ, hộp, thuộc tính và tương quan vị trí. '
           'Nhánh cấp vật thể gồm M2 sinh ảnh từng vật thể và M3 học token định danh. Nhánh cấp cảnh gồm M4 dựng cảnh: đặt vật của M2 khớp nét vẽ, '
           'giữ vật ở các bước đầu rồi nhả dần để vật hòa vào nền; câu lệnh chứa token định danh được dùng từ bước đầu. '
           'M5 điều kiện hóa nhận biết vật thể: phần a, b, c tác động ở mọi bước khử nhiễu; phần d kiểm tra ảnh cuối, '
           'nếu thiếu hoặc lệch vật thì quay lại M4 vẽ lại riêng vùng lỗi, tối đa hai lần.')
p.save(sys.argv[2]); print('ok', im.size)
