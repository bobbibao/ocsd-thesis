"""Bobby 2026-11-26: on the recap slide 18 drop the last animation, the orange frame around the Stable Diffusion part
(shape 'Rounded Rectangle 34', zoom entrance). The fade of the Diffusion part stays. Usage: slide18_noframe.py IN OUT"""
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); s = p.slides[17]
fr = [x for x in s.shapes if x.name == 'Rounded Rectangle 34'][0]; sid = str(fr.shape_id)
for el in s._element.xpath(f'.//p:par[p:cTn//p:spTgt[@spid="{sid}"]]') + s._element.xpath(f'.//p:bldP[@spid="{sid}"]'):
    if el.getparent() is not None: el.getparent().remove(el)
fr._element.getparent().remove(fr._element)
assert f'spid="{sid}"' not in __import__('lxml.etree', fromlist=['x']).tostring(s._element).decode()
p.save(sys.argv[2]); print('ok')
