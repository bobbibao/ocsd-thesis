"""Two slides after the OCSD architecture slide: one real image traced through M1-M5 (SD 1.5 OCSD, trace notebook
04_trace_ocsd, scene qd_3_medium_003, seed 0). Images in img/ come from the trace ZIP (paths taken from trace.json).
Usage: python build_trace_slides.py IN.pptx OUT.pptx"""
import copy, json, re, sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
SRC, OUT = sys.argv[1:3]
p = Presentation(SRC)
INK, MUT, ORG, BLU, GRY, LINE, WHT = C(0x1F,0x24,0x30), C(0x5B,0x62,0x70), C(0xD9,0x69,0x2B), C(0x2E,0x5A,0x9C), C(0xF3,0xF4,0xF6), C(0xD5,0xD8,0xDE), C(255,255,255)
M = json.load(open('trace.json'))
loc = lambda q: 'img/' + q.split('qd_3_medium_003_s0/')[1]
SL = list(p.slides)
arch = [s for s in SL if any(x.has_text_frame and x.text_frame.text.startswith('Kiến trúc OCSD: hai nhánh') for x in s.shapes)][0]
tmpl = [s for s in SL if any(x.has_text_frame and x.text_frame.text.startswith('Phân rã đối tượng và sinh ảnh') for x in s.shapes)][0]

def new_slide():
    s = p.slides.add_slide(tmpl.slide_layout)
    for ph in list(s.placeholders): ph._element.getparent().remove(ph._element)
    for x in tmpl.shapes:
        if x.name in ('Text 0', 'Text 1', 'Text 3') or (x.has_text_frame and x.text_frame.text.startswith('OCSD ·')):
            s.shapes._spTree.append(copy.deepcopy(x._element))
    return s
def title(s, t):
    x = [x for x in s.shapes if x.name == 'Text 1'][0]; r = x.text_frame.paragraphs[0].runs
    r[0].text = t
    for q in r[1:]: q.text = ''
def tb(s, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=2):
    t = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in 'margin_left', 'margin_right', 'margin_top', 'margin_bottom': setattr(tf, m, 0)
    for k, line in enumerate(lines):
        para = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); para.alignment = align; para.space_after = Pt(space)
        for seg in line:
            txt, sz, col, b = seg[:4]
            r = para.add_run(); r.text = txt; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b; r.font.name = 'Arial'
            if len(seg) > 4: r.font.italic = seg[4]
    return t
def pic(s, f, x, y, z, hl=False, cap=None, capcol=MUT, capb=False):
    q = s.shapes.add_picture(f, I(x), I(y), I(z), I(z)); q.line.color.rgb = ORG if hl else LINE; q.line.width = Pt(2.25 if hl else 0.75)
    if cap: tb(s, x - 0.08, y + z + 0.03, z + 0.16, 0.22, [[(cap, 9, ORG if hl else capcol, hl or capb)]], align=PP_ALIGN.CENTER)
    return q
def head(s, x, y, w, tag, text):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(x), I(y), I(0.5), I(0.26)); b.fill.solid(); b.fill.fore_color.rgb = ORG
    b.line.fill.background(); b.shadow.inherit = False
    tb(s, x, y, 0.5, 0.26, [[(tag, 10.5, WHT, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, x + 0.6, y, w - 0.6, 0.3, [[(text, 11.5, INK, True)]], anchor=MSO_ANCHOR.MIDDLE)
def notes(s, t): s.notes_slide.notes_text_frame.text = t

# ------------------------------------------------------------- slide 1: input, M1, M2, M3
s = new_slide(); A = s
title(s, 'Ví dụ thật: một ảnh đi qua năm mô-đun (1/2)')
z = 1.18
# input + M1
tb(s, 0.6, 1.42, 2.6, 0.3, [[('Đầu vào', 11.5, INK, True)]], anchor=MSO_ANCHOR.MIDDLE)
pic(s, 'img/00_input/sketch.png', 0.6, 1.8, 1.55)
tb(s, 0.6, 3.4, 2.55, 0.6, [[('“two orange buses and a wooden bench near a train station”', 9.5, MUT, False, True)]])
head(s, 0.6, 4.08, 2.6, 'M1', 'Phân rã')
pic(s, 'img/M1/parsed_masks_boxes.png', 0.6, 4.45, 1.55)
tb(s, 0.6, 6.05, 2.55, 0.7, [[('3 vật: xe buýt 1, ghế, xe buýt 2; mỗi vật có vùng, hộp và quan hệ trái/phải, trên/dưới.', 9.5, MUT, False)]])
# M2
x2 = 3.45
head(s, x2, 1.42, 4.2, 'M2', 'Sinh từng vật: 2 ứng viên, chọn cái khớp nét')
names = ['xe buýt 1', 'ghế gỗ', 'xe buýt 2']
for i, o in enumerate(M['m2']):
    y = 1.82 + i * 1.62
    pic(s, loc(o['sketch_crop']), x2, y, z, cap=f'nét: {names[i]}')
    for k, c in enumerate(o['candidates']):
        pic(s, loc(c['file']), x2 + 1.3 + k * 1.3, y, z, hl=c['selected'], cap='được chọn' if c['selected'] else f'ứng viên {k + 1}')
# M3
x3 = 7.6
head(s, x3, 1.42, 5.1, 'M3', 'Học định danh: token ⟨oᵢ⟩ nhớ đúng vật của M2')
for k, h in enumerate(['vật từ M2', '“a bus”', '“⟨oᵢ⟩ bus”']):
    pass
for i, r in enumerate(M['m3']['check']):
    y = 1.82 + i * 1.62; cls = r['cls']
    pic(s, loc(r['reference']), x3, y, z, cap='vật từ M2')
    pic(s, loc(r['class_word']), x3 + 1.3, y, z, cap=f'trước: “a {cls}”')
    pic(s, loc(r['learned_token']), x3 + 2.6, y, z, hl=True, cap=f'sau: “{r["token"]} {cls}”')
tb(s, x3 + 3.95, 1.85, 1.15, 4.6, [[('Cùng hạt giống, chỉ đổi từ trong câu lệnh.', 9.5, INK, True)],
                                   [('Xe buýt 2: đỏ hai tầng → cam một tầng, giống vật của M2.', 9.5, MUT, False)],
                                   [('Ghế: chuyển sang gỗ tối như M2.', 9.5, MUT, False)],
                                   [('Xe buýt 1: chưa thấy khác biệt.', 9.5, MUT, False)]], space=8)
notes(s, 'Đây là một ảnh thật do OCSD tạo ra, nhóm ghi lại kết quả của từng mô-đun. Đầu vào là phác thảo ba vật và câu mô tả: hai xe buýt cam và một ghế gỗ gần nhà ga. '
         'M1 tách phác thảo thành ba vật, mỗi vật có vùng, hộp và quan hệ vị trí. M2 sinh riêng từng vật theo nét vẽ của nó, mỗi vật hai ứng viên, rồi chọn ứng viên khớp nét nhất, khung cam là cái được chọn. '
         'M3 dạy cho mỗi vật một từ mới, ví dụ o2. Cột giữa là ảnh khi câu lệnh chỉ ghi “a bus”, cột phải là khi ghi “o2 bus” với cùng hạt giống: xe buýt đỏ hai tầng chuyển thành xe buýt cam một tầng giống vật của M2; ghế chuyển sang gỗ tối. '
         'Nhóm cũng nói thật: với xe buýt 1 thì chưa thấy khác biệt.')

# ------------------------------------------------------------- slide 2: M4, denoising, M5
s = new_slide(); B = s
title(s, 'Ví dụ thật: một ảnh đi qua năm mô-đun (2/2)')
tr = [r for r in M['m5']['trajectory'] if r['phase'] == 'scene']
z = 1.2
head(s, 0.6, 1.42, 12.1, 'M4', 'Tổng hợp cảnh: đặt vật của M2 đúng chỗ phác thảo, rồi trộn vào nền khi khử nhiễu')
pic(s, 'img/M4/composed_foreground.jpg', 0.6, 1.78, 0.98, cap='vật đặt đúng chỗ')
pic(s, 'img/M4/blend_mask.png', 1.72, 1.78, 0.98, cap='mặt nạ trộn')
tb(s, 2.95, 1.85, 9.75, 1.0, [[('27 trên 30 bước đầu (α = 0,1): nền được khử nhiễu riêng theo câu lệnh nền, ', 11, INK, False),
                             ('vật của M2 được dán lại vào đúng vùng ở mỗi bước', 11, INK, True), ('.', 11, INK, False)],
                            [('3 bước cuối: cả cảnh khử nhiễu chung với câu lệnh có token ⟨oᵢ⟩, để vật hòa vào nền (ánh sáng, bóng).', 11, INK, False)]], space=6)
y = 3.45
tb(s, 0.6, y - 0.36, 12.1, 0.3, [[('Từ nhiễu đến ảnh: 30 bước DDIM của chính lần chạy này', 11.5, INK, True)]], anchor=MSO_ANCHOR.MIDDLE)
steps = [0, 5, 10, 15, 20, 26, 29]
for k, st in enumerate(steps):
    r = tr[st]; f = loc(r.get('composite') or r['unet_input'])
    pic(s, f, 0.6 + k * 1.37, y, z, cap=f'bước {st + 1}' + (' (trộn)' if r['blend'] else ''))
pic(s, 'img/final.png', 0.6 + 7 * 1.37 + 0.15, y, z, hl=True, cap='ảnh cuối')
y = 5.32
head(s, 0.6, y - 0.4, 12.1, 'M5', 'Điều kiện hóa nhận biết đối tượng và kiểm tra sau khi sinh')
st = 27
for i, nm in enumerate(['xe buýt 1', 'ghế', 'xe buýt 2']):
    pic(s, loc(tr[st]['attn'][i]), 0.6 + i * 1.37, y, z, cap=f'chú ý “{nm}”')
pic(s, 'img/M5/scene_controlnet_input.png', 0.6 + 3 * 1.37, y, z, cap='ControlNet cảnh')
pic(s, loc(M['m5d']['checks'][0]['file']), 0.6 + 4 * 1.37, y, z, hl=True, cap='kiểm tra: đủ 3/3')
tb(s, 0.6 + 5 * 1.37 + 0.1, y, 4.6, 1.4, [[('(a) Mỗi từ chỉ “vẽ” trong vùng của vật đó (bản đồ chú ý đỏ đúng chỗ).', 10.5, INK, False)],
                                         [('(b) Năng lượng chú ý kéo mỗi vật về đúng vùng ở 3 bước cuối.', 10.5, INK, False)],
                                         [('(c) ControlNet cấp cảnh giữ hình dạng theo phác thảo.', 10.5, INK, False)],
                                         [('(d) Bộ phát hiện tìm đủ 2 xe buýt và 1 ghế → đạt, không phải sinh lại.', 10.5, INK, False)]], space=5)
tb(s, 0.6, 6.82, 12.1, 0.22, [[('Ảnh thật từ notebook 04_trace_ocsd: OCSD (SD 1.5, đủ M1–M5, cấu hình như các bảng kết quả), cảnh qd_3_medium_003, hạt giống 0; chạy trên GPU L4, mất khoảng 2 phút.', 8.5, MUT, False, True)]])
notes(s, 'Tiếp theo là phần dựng cảnh. M4 đặt các vật của M2 vào đúng chỗ đã phác thảo. Trong 27 trên 30 bước đầu, nền được khử nhiễu riêng, còn các vật được dán lại vào đúng vùng ở mỗi bước, nên hàng giữa thấy rõ: nền đi từ nhiễu thành nhà ga, còn hai xe buýt và ghế nằm đúng chỗ. '
         'Ba bước cuối cả cảnh được khử nhiễu chung để vật hòa vào nền. M5 hoạt động trong lúc này: bản đồ chú ý cho thấy từ “bus” và “bench” chỉ tác động đúng vùng của vật; ControlNet cấp cảnh giữ hình dạng theo phác thảo. '
         'Cuối cùng M5 kiểm tra bằng bộ phát hiện: tìm đủ hai xe buýt và một ghế, nên ảnh đạt và không cần sinh lại.')

# ------------------------------------------------------------- move the two new slides right after the architecture slide
lst = p.slides._sldIdLst; ids = list(lst)
new = ids[-2:]
for e in new: lst.remove(e)
anchor = [e for e in ids if p.part.related_part(e.get(qn('r:id'))) is arch.part][0]
for e in reversed(new): anchor.addnext(e)
tot = len(p.slides)
for i, sl in enumerate(p.slides, 1):
    for t in sl._element.iter(qn('a:t')):
        if t.text and re.fullmatch(r'\s*\d+ / \d+\s*', t.text): t.text = f'{i} / {tot}'
p.save(OUT); print('slides', tot)
