"""Bobby chose (2026-11-26) to rerun the farm scene with seed 1 for the M3 slide. In the seed-1 run M3 learns the
truck correctly (<o2> -> red cab + long trailer, like the M2 truck). M2 is identical across seeds, so only slide 25
switches to seed 1: M3 check images + the seed-1 OCSD image on the farm background. Usage: m3_seed1.py IN OUT IMG_DIR"""
import sys, copy
from pptx import Presentation
from pptx.util import Inches as I
from pptx.dml.color import RGBColor
IN, OUT, IMG = sys.argv[1:4]
p = Presentation(IN); s = p.slides[24]
S = {x.name: x for x in s.shapes}
def pic(name, f):
    o = S[name]; q = s.shapes.add_picture(f'{IMG}/{f}', o.left, o.top, o.width, o.height)
    ln = o._element.spPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ln')
    if ln is not None: q._element.spPr.append(copy.deepcopy(ln))
    q.name = name; q._element.nvPicPr.cNvPr.id = o.shape_id
    o._element.addprevious(q._element); o._element.getparent().remove(o._element); S[name] = q
for n, f in (('Picture 58', 'm3_ref_s1.png'), ('Picture 59', 'm3_class_s1.png'), ('Picture 60', 'm3_token_s1.png'), ('Picture 61', 'final_s1.png')):
    pic(n, f)
r = S['TextBox 50'].text_frame.paragraphs[0].runs[0]
r.text = '→ <o2> ra đúng xe của M2: cabin đỏ, thùng dài phía sau'; r.font.color.rgb = RGBColor.from_string('2E8B57')
pa = [q for q in S['TextBox 56'].text_frame.paragraphs if q.runs and q.runs[0].text.startswith('→ <o2>')][0]
pa.runs[0].text = ('→ <o2> gọi lại đúng chiếc xe này (cabin đỏ, thùng trắng), ở bên trái. Ba con bò (<o0>, <o3>, <o4>) cùng loại '
                   'nhưng mỗi con một từ riêng, nên không lẫn nhau. (Ảnh M3 và ảnh cuối: lần chạy hạt giống 1.)')
# ring around the truck in the seed-1 image (box in 512 px, read off the image)
fp = S['Picture 61']; k = fp.width / 512; b = (33, 252, 112, 320); pad = 8
ring = S['Oval 53']; ring.left, ring.top = int(fp.left + (b[0] - pad) * k), int(fp.top + (b[1] - pad) * k)
ring.width, ring.height = int((b[2] - b[0] + 2 * pad) * k), int((b[3] - b[1] + 2 * pad) * k)
for nm in ('Rounded Rectangle 54', 'TextBox 55'):
    t = S[nm]; t.left, t.top = ring.left + I(0.06), ring.top + ring.height + I(0.02)
s.notes_slide.notes_text_frame.text = (
    'M3 dạy mô hình nhớ từng vật. Lấy ví dụ chiếc xe tải mà M2 đã vẽ: cabin đỏ, thùng xám phía sau. Nhóm tạo một từ mới, <o2>, rồi cho mô hình xem chiếc xe này khoảng 200 lần: '
    'mỗi lần phủ nhiễu và bắt mô hình khử lại. Lỗi chỉ được tính bên trong vùng của xe, nên mô hình học đúng chiếc xe chứ không học nền; '
    'đồng thời từ <o2> bị buộc chỉ chú ý vào vùng của xe này, để không lẫn với bò và cây bên cạnh. '
    'Kiểm tra: cùng hạt giống, câu lệnh chỉ ghi “a truck” thì ra một xe đầu kéo đỏ nhìn từ phía trước; thay bằng <o2> thì ra đúng kiểu xe của M2: cabin đỏ kéo một thùng dài. '
    'Trong M4, câu lệnh của cảnh dùng <o2> cho chiếc xe này, nên trong ảnh cuối nó là chiếc xe cabin đỏ thùng trắng bên trái, chỗ khoanh trắng. '
    'Ba con bò cùng loại nhưng mỗi con một từ riêng, <o0>, <o3> và <o4>, nên không bị gộp hay lẫn nhau; cây là <o1>. '
    'Lưu ý: ảnh trên slide này là lần chạy hạt giống 1 của cùng cảnh; M1, M2 giống hệt lần chạy hạt giống 0 ở các slide khác.')
p.save(OUT); print('ok')
