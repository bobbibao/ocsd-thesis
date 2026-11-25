"""New slide right after the M5 slide: sketch -> OCSD (M1-M5) -> image, then the same sketch with other backgrounds
(only the background words of the prompt change). Images: notebook 07 final OCSD (ocsd/bg_swap.py, harmonised M4; img_final/), SD 1.5 OCSD, scene
qd_3_medium_003, seed 0. Idempotent (replaces the slide if present). Usage: bg_slide.py in.pptx out.pptx (run here)."""
import copy, json, re, sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

TITLE = 'Kết quả OCSD: từ phác thảo đến ảnh, và khi đổi bối cảnh'
ORG, INK, MUT, BLU, LINE = C(0xD9, 0x69, 0x2B), C(0x1F, 0x24, 0x30), C(0x5A, 0x60, 0x6E), C(0x2E, 0x5A, 0x9C), C(0xD0, 0xD4, 0xDA)
PEACH, LBL = C(0xFD, 0xEC, 0xE1), C(0xE6, 0xEE, 0xFA)
p = Presentation(sys.argv[1])
title = lambda s: next((sh.text_frame.text.strip() for sh in s.shapes if sh.name == 'Text 1' and sh.has_text_frame), '')
lst = p.slides._sldIdLst
for i, s in enumerate(list(p.slides)):          # idempotent
    if title(s) == TITLE:
        sid = lst[i]; p.part.drop_rel(sid.rId); lst.remove(sid)
src_i = next(i for i, s in enumerate(p.slides) if title(s) == 'M5-Điều kiện hóa nhận biết đối tượng')
src = p.slides[src_i]
s = p.slides.add_slide(src.slide_layout)
for sh in list(s.placeholders): sh._element.getparent().remove(sh._element)
for sh in src.shapes:
    if sh.name in ('Text 0', 'Text 1', 'Text 2', 'Text 3') and sh.top > I(0.3):
        s.shapes._spTree.append(copy.deepcopy(sh._element))
for sh in s.shapes:
    if sh.name == 'Text 1':
        r = sh.text_frame.paragraphs[0].runs; r[0].text = TITLE
        for q in r[1:]: q.text = ''
new = lst[-1]; lst.remove(new); lst.insert(src_i + 1, new)

def tb(x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=2):
    t = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = I(0.04); tf.margin_top = tf.margin_bottom = I(0.01)
    for j, para in enumerate(paras):
        pa = tf.paragraphs[0] if j == 0 else tf.add_paragraph(); pa.alignment = align; pa.space_after = Pt(space)
        for text, sz, col, b, *it in para:
            r = pa.add_run(); r.text = text; f = r.font; f.size = Pt(sz); f.color.rgb = col; f.bold = b; f.name = 'Arial'
            f.italic = bool(it and it[0])
    return t

def pic(f, x, y, z, hl=False):
    q = s.shapes.add_picture(f, I(x), I(y), I(z), I(z)); q.line.color.rgb = ORG if hl else LINE; q.line.width = Pt(2.25 if hl else 0.75)

def box(x, y, w, h, fill):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(x), I(y), I(w), I(h)); b.fill.solid(); b.fill.fore_color.rgb = fill
    b.line.fill.background(); b.shadow.inherit = False; b.adjustments[0] = 0.08; return b

def arrow(x, y, w, col=ORG):
    a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, I(x), I(y), I(w), I(0.3)); a.fill.solid(); a.fill.fore_color.rgb = col
    a.line.fill.background(); a.shadow.inherit = False

M = json.load(open('img_final/bgswap.json'))
runs = M['runs']
VI = {'near a train station': 'gần nhà ga', 'on a city street': 'trên phố', 'on a snowy field': 'trên cánh đồng tuyết',
      'on a sandy beach': 'trên bãi cát'}
# ---- row 1: sketch -> OCSD -> image
z = 2.25; y1 = 1.5
pic('img_final/sketch.png', 0.6, y1, z)
tb(0.5, y1 + z + 0.03, z + 0.2, 0.4, [[('Đầu vào: phác thảo + “two orange buses and a wooden bench near a train station”', 9, MUT, False, True)]], align=PP_ALIGN.CENTER)
arrow(3.0, y1 + z / 2 - 0.15, 1.3)
tb(2.85, y1 + z / 2 - 0.62, 1.6, 0.45, [[('OCSD', 13, ORG, True)], [('M1 → M5', 10, INK, False)]], align=PP_ALIGN.CENTER, space=0)
pic('img_final/' + runs[0]['file'].split('/')[-1], 4.45, y1, z, hl=True)
tb(4.35, y1 + z + 0.03, z + 0.2, 0.25, [[('Ảnh OCSD tạo ra', 10, ORG, True)]], align=PP_ALIGN.CENTER)
box(7.0, y1, 5.7, z, PEACH)
tb(7.18, y1 + 0.1, 5.35, z - 0.2, [
    [('Đúng bài toán đặt ra từ đầu', 13, ORG, True)],
    [('✓ Đủ vật: ', 11.5, INK, True), ('cả 3 vật đều có, vật hòa vào nền, không viền dán.', 11.5, INK, False)],
    [('✓ Đúng số lượng: ', 11.5, INK, True), ('2 xe buýt + 1 ghế, không thừa, không thiếu.', 11.5, INK, False)],
    [('✓ Đúng chỗ, đúng dáng: ', 11.5, INK, True), ('mỗi vật nằm ở chỗ đã vẽ, theo hình nét vẽ.', 11.5, INK, False)],
    [('✓ Đúng tương quan: ', 11.5, INK, True), ('ghế ở trên, bên phải; 2 xe buýt bên trái, phía dưới.', 11.5, INK, False)],
    [('Chưa tốt: ', 11.5, MUT, True), ('ghế nhỏ, sát tường nhà; vị trí lệch nhẹ so với nét vẽ (xem phần đánh giá).', 11.5, MUT, False, True)]], space=7)
# ---- row 2: change the background
y2 = 4.25; z2 = 1.95
box(0.6, y2 - 0.05, 12.1, 0.34, LBL)
tb(0.7, y2 - 0.03, 11.9, 0.3, [[('Đổi bối cảnh: ', 11, BLU, True), ('cùng phác thảo, cùng vật của M2-M3, chỉ đổi phần nền trong câu mô tả', 11, INK, False)]], anchor=MSO_ANCHOR.MIDDLE)
xs = [0.6, 3.75, 6.9]
for k, r in enumerate(runs[1:4]):
    x = xs[k]
    pic('img_final/' + r['file'].split('/')[-1], x, y2 + 0.4, z2)
    tb(x + z2 + 0.08, y2 + 0.4, 1.05, z2, [[('“… ' + r['bg'] + '”', 9.5, BLU, True, True)], [('(' + VI[r['bg']] + ')', 9, MUT, False)]], anchor=MSO_ANCHOR.MIDDLE)
tb(10.05, y2 + 0.4, 2.65, z2, [
    [('Nền đổi theo câu mô tả, ', 10.5, INK, True), ('còn 3 vật giữ nguyên số lượng, chỗ và dáng như phác thảo.', 10.5, INK, False)],
    [('Ảnh thật, cùng hạt giống; M2-M3 chạy một lần, M4-M5 chạy lại cho mỗi nền.', 9, MUT, False, True)]], anchor=MSO_ANCHOR.MIDDLE, space=4)
ns = s.notes_slide.notes_text_frame
ns.text = ('Đây là toàn bộ OCSD trên một ví dụ thật, đúng bài toán đặt ra từ đầu. Đầu vào là phác thảo hai xe buýt và một ghế, cùng câu mô tả '
           '"two orange buses and a wooden bench near a train station". Qua năm mô-đun M1 đến M5, OCSD cho ra ảnh bên cạnh: đủ ba vật, đúng hai xe buýt '
           'và một ghế, mỗi vật nằm đúng chỗ và đúng dáng đã vẽ, ghế ở trên bên phải, hai xe buýt bên trái phía dưới. Với M4 mới, vật hòa vào nền, không còn viền dán; '
           'điểm chưa tốt là ghế nhỏ, sát tường nhà và vị trí lệch nhẹ so với nét vẽ, nhóm nói rõ ở phần đánh giá. Hàng dưới: giữ nguyên phác thảo và các vật đã tạo ở M2, M3, chỉ đổi phần nền '
           'trong câu mô tả thành trên phố, trên cánh đồng tuyết và trên bãi cát. Nền đổi theo câu, còn ba vật vẫn giữ đúng số lượng, vị trí và hình dạng. '
           'Đây là ảnh thật do code OCSD tạo ra, cùng hạt giống.')
# renumber
N = len(p.slides)
for i, sl in enumerate(p.slides):
    for sh in sl.shapes:
        if sh.has_text_frame and re.fullmatch(r'\d+ / \d+', sh.text_frame.text.strip()):
            r = sh.text_frame.paragraphs[0].runs; r[0].text = f'{i + 1} / {N}'
            for q in r[1:]: q.text = ''
p.save(sys.argv[2]); print('bg slide at', src_i + 2, 'of', N)
