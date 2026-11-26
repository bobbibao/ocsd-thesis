"""Bobby 2026-11-26: M3 identity on the farm truck is wrong (M2 truck = red cab + grey bed; <o2> gives a white box
truck, final truck grey). Make slide 25 say so honestly instead of 'gần với xe của M2'. Usage: slide25_honest.py IN OUT"""
import sys
from pptx import Presentation
from pptx.dml.color import RGBColor
p = Presentation(sys.argv[1]); s = p.slides[24]
sh = {x.name: x for x in s.shapes}
r = sh['TextBox 50'].text_frame.paragraphs[0].runs[0]
assert r.text.startswith('→ xe đầu kéo'), r.text
r.text = '✗ chỉ giữ dáng thùng dài, mất cabin đỏ của xe M2: M3 học sai màu ở cảnh này'
r.font.color.rgb = RGBColor.from_string('C0392B')
pa = [q for q in sh['TextBox 56'].text_frame.paragraphs if q.runs and q.runs[0].text.startswith('→ <o2>')][0]; runs = pa.runs
assert runs[0].text.startswith('→ <o2> gọi lại'), runs[0].text
runs[0].text = ('→ <o2> vẫn đặt đúng chỗ chiếc xe (bên trái), nhưng mang diện mạo M3 học sai nên xe ra màu xám, không đỏ '
                '(hạn chế của M3). Ba con bò (<o0>, <o3>, <o4>) mỗi con một từ riêng, nên không lẫn nhau.')
for extra in runs[1:]: extra._r.getparent().remove(extra._r)
n = s.notes_slide.notes_text_frame
old = 'thay bằng <o2> thì ra xe tải có thùng trắng dài, gần với xe của M2.'
assert old in n.text
n.text = n.text.replace(old, 'thay bằng <o2> thì ra xe tải có thùng trắng dài: giữ được dáng xe thùng nhưng mất cabin đỏ của xe M2. '
                        'Nói thẳng: ở cảnh này M3 học sai màu, độ lỗi huấn luyện gần như không giảm, nên xe trong ảnh cuối màu xám chứ không đỏ. Đây là hạn chế của M3 mà nhóm nêu ở phần hạn chế.')
n.text = n.text.replace('nên trong ảnh cuối nó là chiếc xe bên trái, chỗ khoanh trắng.', 'nên trong ảnh cuối nó vẫn nằm đúng chỗ, chiếc xe bên trái khoanh trắng, dù sai màu.')
p.save(sys.argv[2]); print('ok')
