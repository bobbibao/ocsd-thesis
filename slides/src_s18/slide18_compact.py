"""v38 (Bobby 2026-11-26, page 25): slide 18 drops the dimming overlay (Rectangle 33) and its fade transition;
the Diffusion Model part is compacted into a thin strip so the slide presents Stable Diffusion. Slide 17 untouched.
Usage: slide18_compact.py IN OUT"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
I = Inches
p = Presentation(sys.argv[1]); s = p.slides[17]
sh = {x.name: x for x in s.shapes}
assert 'Rectangle 33' in sh and sh['Text 3'].text_frame.text.startswith('18 /')
ns = '{http://schemas.openxmlformats.org/presentationml/2006/main}'
r = sh['Rectangle 33']._element; r.getparent().remove(r)
for tag in ('timing', 'transition'):
    for e in s._element.findall('.//' + ns + tag): e.getparent().remove(e)
for e in s._element.findall('.//{http://schemas.openxmlformats.org/markup-compatibility/2006}AlternateContent'):
    if e.find('.//' + ns + 'transition') is not None: e.getparent().remove(e)

def geo(n, x, y, w, h):
    o = sh[n]; o.left, o.top, o.width, o.height = I(x), I(y), I(w), I(h)
def size(n, pt):
    for pa in sh[n].text_frame.paragraphs:
        for rr in pa.runs: rr.font.size = Pt(pt)

# --- compact Diffusion strip: 1.45 .. 2.75
geo('TextBox 22', 0.6, 1.45, 7.6, 0.3)
geo('Connector 23', 0.6, 1.78, 7.6, 0)
tb = sh['TextBox 24'].text_frame.paragraphs[0]
tb.runs[0].text = 'Huấn luyện: thêm nhiễu dần vào ảnh thật, dạy mạng đoán phần nhiễu. Tạo ảnh: bắt đầu từ nhiễu ngẫu nhiên, khử nhiễu qua nhiều bước (Ho và cs., 2020).'
for rr in tb.runs[1:]: rr._r.getparent().remove(rr._r)
geo('TextBox 24', 0.6, 1.85, 7.6, 0.6); size('TextBox 24', 11.5)
sh['TextBox 24'].text_frame.word_wrap = True
o = sh['TextBox 31']._element; o.getparent().remove(o)   # long example list dropped in the compact strip
geo('Picture 21', 8.5, 1.45, 4.23, 1.37)
geo('Connector 25', 0.6, 2.95, 12.13, 0)

# --- Stable Diffusion part, moved up and enlarged: 3.05 .. 6.95
geo('TextBox 28', 0.6, 3.07, 12.1, 0.35); size('TextBox 28', 13)
geo('Connector 29', 0.6, 3.45, 12.1, 0)
geo('TextBox 30', 0.6, 3.6, 3.55, 1.9); size('TextBox 30', 13)
geo('SDWEAK idea', 0.6, 5.55, 3.55, 1.3); size('SDWEAK idea', 13)
for n, x, w in (('SDWEAK h1', 4.7, 3.1), ('SDWEAK h2', 7.95, 3.1), ('SDWEAK h3', 11.15, 1.65)):
    geo(n, x, 3.58, w, 0.3); size(n, 11)
RH, Y0, GAP = 0.95, 3.95, 0.1
for i in range(3):
    y = Y0 + i * (RH + GAP)
    geo('SDWEAK n%d' % i, 4.35, y + RH / 2 - 0.15, 0.3, 0.3)
    geo('SDWEAK wbg%d' % i, 4.7, y, 3.1, RH); geo('SDWEAK w%d' % i, 4.7, y, 3.1, RH); size('SDWEAK w%d' % i, 12)
    geo('SDWEAK arr%d' % i, 7.83, y + RH / 2 - 0.1, 0.1, 0.2)
    geo('SDWEAK sbg%d' % i, 7.95, y, 3.1, RH); geo('SDWEAK s%d' % i, 7.95, y, 3.1, RH); size('SDWEAK s%d' % i, 12)
    geo('SDWEAK l%d' % i, 11.15, y, 1.65, RH); size('SDWEAK l%d' % i, 11)
from pptx.enum.text import MSO_ANCHOR
for i in range(3):
    for n in ('SDWEAK w%d', 'SDWEAK s%d', 'SDWEAK l%d'): sh[n % i].text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
p.save(sys.argv[2]); print('ok')
