"""Bobby 2026-11-26 ('vậy vẽ lại cho đúng đi'): OCSD architecture as an EXECUTION flow:
inputs -> M1 -> M2 -> M3 -> [denoising 30 steps: M4 (hold 1-12 / release 13-24 / free 25-30) with M5(a,b,c) inside,
every step] -> M5(d) check -> image; loop 'thiếu/lệch -> vẽ lại vùng lỗi (<= 2 lần)'. Crops the sketch, text and
output picture from arch/arch_in.png. Writes arch/arch_flow.png. Usage: arch_flow.py IN OUT (pptx, slide 21)"""
import math, sys
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches as I
SR, SB, SI = ('/usr/share/fonts/truetype/liberation/LiberationSerif-' + k + '.ttf' for k in ('Regular', 'Bold', 'Italic'))
f = lambda n, w=SR: ImageFont.truetype(w, n)
src = Image.open('arch/arch_in.png').convert('RGB')
W, H = 2700, 1000
im = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(im)
INK, GRY, MUT = (25, 25, 25), (60, 60, 60), (110, 110, 110)
OF, OL = (250, 228, 214), (196, 89, 17)        # object branch
BF, BL = (220, 232, 247), (46, 84, 150)        # scene / M4
PF, PL = (234, 220, 244), (112, 48, 160)       # M5
GF, GL = (236, 236, 236), (90, 90, 90)         # M1
def box(b, fill, line, lines, size=26, bold_first=True, r=10, w=3):
    d.rounded_rectangle(b, r, fill=fill, outline=line, width=w)
    lh = size + 8; y = (b[1] + b[3]) / 2 - lh * len(lines) / 2
    for k, t in enumerate(lines):
        F = f(size + (2 if k == 0 and bold_first else 0), SB if k == 0 and bold_first else SR)
        d.text(((b[0] + b[2]) / 2 - d.textlength(t, font=F) / 2, y), t, font=F, fill=INK); y += lh
def arrow(pts, col=GRY, w=5, head=18):
    d.line(pts, fill=col, width=w, joint='curve')
    (x1, y1), (x2, y2) = pts[-2], pts[-1]; a = math.atan2(y2 - y1, x2 - x1)
    d.polygon([(x2, y2), (x2 - head * math.cos(a - .45), y2 - head * math.sin(a - .45)), (x2 - head * math.cos(a + .45), y2 - head * math.sin(a + .45))], fill=col)
def note(x, y, t, size=21, col=MUT, centre=True):
    F = f(size, SI); w = d.textlength(t, font=F); d.text((x - w / 2 if centre else x, y), t, font=F, fill=col)
def dashed(b, col, w=3, dash=14):
    x0, y0, x1, y1 = b
    for x in range(x0, x1, 2 * dash): d.line([(x, y0), (min(x + dash, x1), y0)], fill=col, width=w); d.line([(x, y1), (min(x + dash, x1), y1)], fill=col, width=w)
    for y in range(y0, y1, 2 * dash): d.line([(x0, y), (x0, min(y + dash, y1))], fill=col, width=w); d.line([(x1, y), (x1, min(y + dash, y1))], fill=col, width=w)
# inputs (cropped from the old figure)
sk = src.crop((18, 158, 323, 462)).resize((250, 250)); im.paste(sk, (30, 200))
tx = src.crop((17, 496, 323, 653)).resize((250, 128)); im.paste(tx, (30, 560))
F = f(24, SB)
for t, y in (('Phác thảo cảnh', 160), ('Câu mô tả', 700)): d.text((155 - d.textlength(t, font=F) / 2, y), t, font=F, fill=INK)
# M1, M2, M3
Y0, Y1 = 330, 620
box((350, Y0, 610, Y1), GF, GL, ['M1', 'Phân rã', 'vật thể', '', 'mặt nạ, hộp,', 'tương quan,', 'cụm từ'], 24)
arrow([(285, 325), (345, 410)]); arrow([(285, 620), (345, 545)])
box((680, Y0, 940, Y1), OF, OL, ['M2', 'Sinh ảnh', 'từng vật thể', '', 'ControlNet', '+ Grounded-SAM,', 'chọn ứng viên'], 24)
box((1010, Y0, 1270, Y1), OF, OL, ['M3', 'Học định danh', '', 'token ‹oᵢ›', '+ LoRA cho', 'từng vật thể'], 24)
arrow([(612, 475), (676, 475)]); arrow([(942, 475), (1006, 475)])
note(645, 432, 'nét', 20); note(975, 432, 'ảnh', 20)
d.text((680, 250), 'Nhánh xử lý vật thể', font=f(27, SB), fill=OL)
note(680, 284, '(chạy một lần, cho từng vật)', 21, OL, centre=False)
dashed((660, 235, 1290, 690), OL)
T1 = 'Tiền xử lý'; d.text((480 - d.textlength(T1, font=f(25, SB)) / 2, 290), T1, font=f(25, SB), fill=GL)
# denoising block: M4 with M5(a,b,c) inside
BX0, BY0, BX1, BY1 = 1350, 150, 2150, 820
d.rounded_rectangle((BX0, BY0, BX1, BY1), 14, fill=(242, 246, 252)); dashed((BX0, BY0, BX1, BY1), BL)
T = 'Nhánh xử lý cảnh: khử nhiễu 30 bước'; F = f(27, SB); d.text(((BX0 + BX1) / 2 - d.textlength(T, font=F) / 2, BY0 + 18), T, font=F, fill=BL)
arrow([(1272, 475), (1346, 475)])
note(1310, 432, 'token', 20)
sw = (BX1 - BX0 - 80) / 3; sy0, sy1 = 300, 520
for k, (a, b_, c_) in enumerate((('M4  Neo vật', 'bước 1–12', 'giữ chặt vật M2'), ('M4  Nhả dần', 'bước 13–24', 'giảm dần lực giữ'), ('M4  Tự do', 'bước 25–30', 'vật hòa vào nền'))):
    x0 = BX0 + 20 + k * (sw + 20); box((x0, sy0, x0 + sw, sy1), BF, BL, [a, b_, c_], 24)
    if k < 2: arrow([(x0 + sw + 2, (sy0 + sy1) / 2), (x0 + sw + 18, (sy0 + sy1) / 2)], BL, w=4, head=12)
note((BX0 + BX1) / 2, 245, 'M4 và M5 cùng chạy ở mỗi bước, một quỹ đạo chung cho vật và nền', 22, BL)
box((BX0 + 20, 580, BX1 - 20, 790), PF, PL, ['M5  Điều kiện hóa nhận biết vật thể (mọi bước 1–30)', '(a) chú ý chéo theo vùng của từng vật', '(b) dẫn hướng năng lượng: vật bị quên / tràn → sửa ngay',
    '(c) ControlNet phác thảo cả cảnh, lực 0,6'], 23)
for k in range(3):
    x = BX0 + 20 + k * (sw + 20) + sw / 2; arrow([(x, 578), (x, 524)], PL, w=4, head=13)
# M5(d) check, then the image
DX0, DX1 = 2230, 2450
box((DX0, 380, DX1, 570), PF, PL, ['M5(d)', 'Kiểm tra:', 'đếm lại vật', '(Grounding DINO)'], 23)
arrow([(BX1 + 2, 475), (DX0 - 4, 475)], GRY)
out = src.crop((1798, 78, 2163, 443)).resize((210, 210)); im.paste(out, (2470, 120))
d.rectangle((2470, 120, 2680, 330), outline=(90, 90, 90), width=2)
d.text((2575 - d.textlength('Ảnh cảnh', font=f(24, SB)) / 2, 80), 'Ảnh cảnh', font=f(24, SB), fill=INK)
arrow([(2400, 378), (2470, 335)], (46, 125, 50), w=5); note(2470, 360, 'đạt', 22, (46, 125, 50), centre=False)
# loop back: missing / wrong -> redraw the faulty region
LP = (200, 40, 40)
arrow([(2340, 572), (2340, 900), ((BX0 + BX1) / 2, 900), ((BX0 + BX1) / 2, 824)], LP, w=5, head=18)
note(2190, 912, 'thiếu / lệch → vẽ lại riêng vùng lỗi (tối đa 2 lần)', 23, LP)
im.save('arch/arch_flow.png')
p = Presentation(sys.argv[1]); s = p.slides[20]
old = next(x for x in s.shapes if x.shape_type == 13)
wid = I(12.1); hei = int(wid * H / W)
new = s.shapes.add_picture('arch/arch_flow.png', I(0.6), I(1.55), wid, hei)
old._element.addprevious(new._element); old._element.getparent().remove(old._element)
s.notes_slide.notes_text_frame.text = (
    'Đây là kiến trúc OCSD theo đúng thứ tự chạy, gồm bước tiền xử lý và hai nhánh: nhánh xử lý vật thể và nhánh xử lý cảnh. Đầu vào là phác thảo và câu mô tả. '
    'Tiền xử lý: M1 phân rã phác thảo thành từng vật thể, có mặt nạ, hộp, tương quan vị trí và cụm từ. Nhánh xử lý vật thể chạy một lần cho từng vật: M2 sinh ảnh riêng cho từng vật bằng ControlNet và tách vật bằng Grounded-SAM; '
    'M3 học token định danh và LoRA cho từng vật. Sau đó là nhánh xử lý cảnh, khử nhiễu 30 bước, trong đó M4 và M5 cùng chạy ở mỗi bước: '
    'M4 đặt vật của M2 khớp nét vẽ, neo vật ở bước 1 đến 12, nhả dần ở 13 đến 24, và để mô hình vẽ tự do ở 25 đến 30; '
    'M5 phần a, b, c tác động ở mọi bước: chú ý chéo theo vùng, dẫn hướng năng lượng khi vật bị quên hoặc tràn ra ngoài, và ControlNet phác thảo cả cảnh với lực 0,6. '
    'Cuối cùng M5 phần d đếm lại vật trong ảnh; nếu đạt thì xuất ảnh, nếu thiếu hoặc lệch thì quay lại khối khử nhiễu, chỉ vẽ lại vùng lỗi, tối đa hai lần.')
p.save(sys.argv[2]); print('ok')
