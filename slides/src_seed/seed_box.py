"""v46 (Bobby 2026-11-26, page 23): 'Nhiễu ban đầu khác -> ảnh khác' box uses real photo-like SD 1.5 outputs that
differ clearly: notebook 11 text-only farm caption, seeds 0, 1, 6 (top row of src_s4farm/img/SHEET.jpg),
each with its own Gaussian-noise tile. Images enlarged. Usage: seed_box.py IN OUT"""
import sys, io, os
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.util import Inches as I
HERE = os.path.dirname(os.path.abspath(__file__))
sheet = Image.open(os.path.join(HERE, '..', 'src_s4farm', 'img', 'SHEET.jpg')).convert('RGB')
T = sheet.width // 8
p = Presentation(sys.argv[1]); s = p.slides[15]
sh = {x.name: x for x in s.shapes}
assert sh['TextBox 77'].text_frame.text.startswith('Nhiễu ban đầu khác')
def put(pic, img):
    buf = io.BytesIO(); img.save(buf, 'PNG'); buf.seek(0)
    _, rid = pic.part.get_or_add_image_part(buf)
    pic._element.blipFill.blip.rEmbed = rid
for k, (seed, nz, out) in enumerate(((0, 'Picture 78', 'Picture 80'), (1, 'Picture 81', 'Picture 83'), (6, 'Picture 84', 'Picture 86'))):
    put(sh[out], sheet.crop((seed * T, 0, seed * T + T, T)).resize((384, 384)))
    n = np.clip(np.random.default_rng(seed).normal(128, 60, (64, 64, 3)), 0, 255).astype('uint8')
    put(sh[nz], Image.fromarray(n).resize((192, 192), Image.NEAREST))
    x0 = 7.75 + k * 1.6
    sh[nz].left, sh[nz].top, sh[nz].width, sh[nz].height = I(x0), I(4.98), I(0.62), I(0.62)
    sh[out].left, sh[out].top, sh[out].width, sh[out].height = I(x0 + 0.38), I(5.3), I(1.12), I(1.12)
    ar = sh[('TextBox 79', 'TextBox 82', 'TextBox 85')[k]]; ar.left, ar.top = I(x0 + 0.64), I(4.98)
cap = sh['TextBox 87'].text_frame.paragraphs[0]
cap.runs[0].text = 'Cùng câu lệnh về nông trại, chỉ đổi nhiễu ban đầu: ảnh khác hẳn.'
for r in cap.runs[1:]: r._r.getparent().remove(r._r)
nt = s.notes_slide.notes_text_frame; t = nt.text
a = 'Stable Diffusion làm đúng như vậy nhưng thêm câu lệnh, để ảnh tạo ra khớp với câu mô tả.'
assert a in t
nt.text = t.replace(a, 'Ba ví dụ bên phải là ảnh thật của Stable Diffusion 1.5 với cùng một câu lệnh về nông trại, chỉ đổi nhiễu ban đầu: bố cục, góc chụp và cách đặt vật đều khác hẳn.')
p.save(sys.argv[2]); print('ok')
