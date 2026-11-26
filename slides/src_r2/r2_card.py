"""v51 (Bobby 2026-11-26, slide 31 metrics): add an R2 'Đúng số lượng' card (OCE + count accuracy, ocsd/matching.py:
oce_c = sum over classes |found - drawn|, count_acc = 1 if oce_c == 0) on the same farm image; four cards R1-R4 in
order; caption lines under the cards removed to keep space. Usage: r2_card.py IN OUT"""
import sys, io, os, copy
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches as I, Pt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src_bobby'))
from anim_lib import add_clicks
p = Presentation(sys.argv[1]); s = p.slides[30]
sh = {x.name: x for x in s.shapes}
assert sh['TextBox 20'].text_frame.text == 'Giữ vật thể (OPR)'
# --- R2 picture: farm image + count panel
im = Image.open(io.BytesIO(sh['Picture 41'].image.blob)).convert('RGB')
d = ImageDraw.Draw(im, 'RGBA')
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 25)
F2 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
rows = [('bò', 3, 3), ('cây', 1, 1), ('xe tải', 1, 1)]
x0, y0, rh = 16, 14, 40
d.rounded_rectangle((x0, y0, 496, y0 + 36 + rh * 3 + 8), 10, fill=(255, 255, 255, 255), outline=(217, 105, 43, 255), width=3)
for cx, t in ((x0 + 16, 'loại'), (210, 'đã vẽ'), (330, 'tìm thấy')):
    d.text((cx, y0 + 8), t, font=F2, fill=(91, 98, 112))
for k, (c, a, b) in enumerate(rows):
    y = y0 + 40 + k * rh
    d.text((x0 + 16, y), c, font=F, fill=(31, 36, 48))
    d.text((240, y), str(a), font=F, fill=(31, 36, 48))
    d.text((370, y), str(b), font=F, fill=(31, 36, 48))
    d.text((440, y), '✓', font=F, fill=(46, 160, 67))
buf = io.BytesIO(); im.save(buf, 'PNG'); buf.seek(0)
# --- layout: 4 cards
W, G, X0 = 2.87, 0.2, 0.6
groups = [['Rounded Rectangle 19', 'TextBox 20', 'TextBox 21', 'Picture 41', 'Rounded Rectangle 23', 'TextBox 24'],
          None,
          ['Rounded Rectangle 26', 'TextBox 27', 'TextBox 28', 'Picture 42', 'Rounded Rectangle 30', 'TextBox 31'],
          ['Rounded Rectangle 33', 'TextBox 34', 'TextBox 35', 'Picture 43', 'Rounded Rectangle 37', 'TextBox 38']]
for n in ('TextBox 25', 'TextBox 32', 'TextBox 39'):
    sh[n]._element.getparent().remove(sh[n]._element)
# clone R1 group for R2
new = []
for n in groups[0]:
    el = copy.deepcopy(sh[n]._element); s.shapes._spTree.append(el); new.append(s.shapes[-1])
ids = max(x.shape_id for x in s.shapes)
for k, x in enumerate(new):
    x._element.xpath('./*[1]/p:cNvPr')[0].set('id', str(ids + 1 + k)); x.name = groups[0][k] + ' R2'
groups[1] = [x.name for x in new]
sh = {x.name: x for x in s.shapes}
put_pic = sh['Picture 41 R2']; _, rid = put_pic.part.get_or_add_image_part(buf); put_pic._element.blipFill.blip.rEmbed = rid
def settext(box, texts):
    for para, t in zip(box.text_frame.paragraphs, texts):
        para.runs[0].text = t
        for r in para.runs[1:]: r._r.getparent().remove(r._r)
settext(sh['TextBox 20 R2'], ['Đúng số lượng (OCE)'])
settext(sh['TextBox 21 R2'], ['R2 · số lượng'])
settext(sh['TextBox 24 R2'], ['OCE = tổng |số tìm thấy − số đã vẽ| của từng loại', '= 0 + 0 + 0 = 0'])
tb = sh['TextBox 24 R2'].text_frame; pp = tb.add_paragraph(); pp.alignment = tb.paragraphs[0].alignment
r = pp.add_run(); r.text = 'OCE = 0 → ảnh đếm đúng (tỉ lệ đếm đúng)'
r.font.name = 'Arial'; r.font.size = Pt(9); r.font.color.rgb = tb.paragraphs[0].runs[0].font.color.rgb
for i, g in enumerate(groups):
    cx = X0 + i * (W + G)
    card, title, sub, pic, fbox, ftxt = (sh[n] for n in g)
    card.left, card.top, card.width, card.height = I(cx), I(1.55), I(W), I(4.95)
    for t in (title, sub): t.left, t.width = I(cx + 0.15), I(W - 0.25)
    title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
    pic.left, pic.top, pic.width, pic.height = I(cx + (W - 2.45) / 2), I(2.4), I(2.45), I(2.45)
    fbox.left, fbox.top, fbox.width, fbox.height = I(cx + 0.1), I(4.98), I(W - 0.2), I(1.38)
    ftxt.left, ftxt.top, ftxt.width, ftxt.height = I(cx + 0.14), I(5.02), I(W - 0.28), I(1.3)
    ftxt.text_frame.word_wrap = True
    for para in ftxt.text_frame.paragraphs:
        for rr in para.runs:
            if rr.font.size and rr.font.size > Pt(11.5): rr.font.size = Pt(12)
            elif rr.font.size and rr.font.size >= Pt(10): rr.font.size = Pt(9.5)
add_clicks(s, [[(sh[n], 'fade') for n in g] for g in groups[1:]])
nt = s.notes_slide.notes_text_frame; t = nt.text
t = t.replace('Nhóm dùng ba độ đo, cũng là ba độ đo trong các slide kết quả.', 'Nhóm dùng bốn độ đo, ứng với bốn yêu cầu R1 đến R4.')
a = 'Vẽ 5 vật, tìm thấy đủ 5, nên 100 phần trăm.'
assert a in t
t = t.replace(a, a + ' Thứ hai, đúng số lượng, OCE: với từng loại, lấy số vật tìm thấy trừ số vật đã vẽ, lấy trị tuyệt đối rồi cộng lại; 0 là hoàn hảo. Ở đây vẽ 3 bò, 1 cây, 1 xe tải và tìm thấy đúng 3, 1, 1, nên OCE bằng 0. Ảnh nào có OCE bằng 0 thì tính là đếm đúng; tỉ lệ ảnh đếm đúng là độ đo đi kèm.')
t = t.replace('Thứ hai, vị trí và hình dạng', 'Thứ ba, vị trí và hình dạng').replace('Thứ ba, đúng tương quan', 'Thứ tư, đúng tương quan')
nt.text = t
p.save(sys.argv[2]); print('ok')
