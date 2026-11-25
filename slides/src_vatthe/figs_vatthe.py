"""Re-letter the deck's text-bearing images with 'vật thể' (Bobby 2026-11-25) and refresh the stale M4 boxes of
the architecture figure (harmonised M4). Inputs: img/s2.png (concept, also slide 7), img/s21.png (architecture),
img/s35.png (code architecture), extracted from the deck. Outputs img/new_*.png and the slide-6 chart."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
F = '/usr/share/fonts/truetype/liberation/LiberationSerif-'
def font(style, size): return ImageFont.truetype(F + style + '.ttf', size)
def patch(im, box, lines, style='Regular', size=22, color=(0, 0, 0), fill=None, gap=1.18):
    d = ImageDraw.Draw(im); x0, y0, x1, y1 = box
    if fill is None: fill = im.getpixel((x0 + 2, y0 + 2))
    d.rectangle(box, fill=fill)
    f = font(style, size); h = size * gap; ytop = (y0 + y1) / 2 - h * len(lines) / 2
    for k, t in enumerate(lines):
        w = d.textlength(t, font=f); d.text(((x0 + x1) / 2 - w / 2, ytop + k * h), t, font=f, fill=color)
# ---- concept (slides 2 and 7)
im = Image.open('img/s2.png').convert('RGB')
patch(im, (40, 0, 660, 33), ['Phác thảo cảnh (nét vẽ nhóm theo vật thể)'], 'Bold', 27)
patch(im, (1260, 603, 1900, 641), ['Bố cục tuân theo phác thảo – thuộc tính đúng vật thể –'], 'Regular', 24)
patch(im, (60, 845, 625, 882), ['Văn bản mô tả (cụm từ ứng với từng vật thể)'], 'Italic', 23)
im.save('img/new_concept.png')
# ---- architecture (2200 x 1120)
im = Image.open('img/s21.png').convert('RGB'); k = 1.1
K = lambda *b: tuple(int(round(v * k)) for v in b)
ORG, BLU, PUR = (196, 89, 17), (46, 84, 150), (112, 48, 160)
patch(im, K(960, 38, 1240, 74), ['Nhánh cấp vật thể'], 'Bold', 27, ORG)
patch(im, K(455, 349, 555, 373), ['vật thể'], 'Regular', 22)
patch(im, K(845, 206, 965, 230), ['vật thể'], 'Regular', 22)
blue = [(K(742, 534, 1094, 730), ['M4  Đặt vật thể của M2', 'khớp nét phác thảo', '(tô nền quanh vật,', 'làm mềm viền)']),
        (K(1160, 534, 1512, 730), ['M4  Giữ vật (bước 1–12)', 'rồi nhả dần (13–24)', 'một quỹ đạo chung', 'cho vật và nền']),
        (K(1578, 534, 1939, 730), ['M4  Tự do (bước 25–30)', 'câu lệnh toàn cục', 'chứa token định danh', '‹o₀›, ‹o₁›, …'])]
for b, t in blue: patch(im, b, t, 'Regular', 22)
patch(im, K(1165, 838, 1935, 865), ['M5  Điều kiện hóa nhận biết vật thể:'], 'Regular', 22)
patch(im, K(760, 900, 905, 924), ['cụm từ vật thể'], 'Regular', 19, PUR)
im.save('img/new_arch.png')
# ---- code architecture
im = Image.open('img/s35.png').convert('RGB')
patch(im, (70, 290, 570, 325), ['Mô tả từng vật thể'], 'Regular', 24)
patch(im, (1130, 505, 1690, 535), ['Bộ đệm: ảnh vật thể, mặt nạ,'], 'Regular', 24)
im.save('img/new_code.png')
# ---- slide 6 chart (same style as src_hoidong/charts.py)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(6, 4.4))
v = [97, 64, 62, 39]; b = ax.bar(['1', '3', '5', '8+'], v, color=['#c9ced4', '#b0b7bf', '#8f98a2', '#e0662a'], width=.6)
for r, x_ in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, x_ + 2, f'{x_}%', ha='center', fontsize=18, fontweight='bold')
ax.set_ylim(0, 112); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlabel('Số vật thể trong phác thảo')
ax.set_title('Mô hình nền: tỉ lệ vật thể còn giữ được', fontsize=15)
fig.tight_layout(); fig.savefig('img/new_prob.png', dpi=200); plt.close(fig)
print('ok')
