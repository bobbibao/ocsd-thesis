"""Chain step 12 (after final_results.py): put the FINAL OCSD (harmonised M4, notebook 07, Bobby 2026-11-25) images in
the deck. Rebuilds the M4 and M5 slides, swaps the qualitative images, adds the slide 'M4 mới so với M4 cũ' after the
comparison slide and replaces the limitation picture. Needs img/ from figs_final.py. Usage (run here): final_images.py IN OUT"""
import copy, json, re, sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
p = Presentation(sys.argv[1])
INK, MUT, ORG, BLU, PUR, LINE, WHT = C(0x1F,0x24,0x30), C(0x5B,0x62,0x70), C(0xD9,0x69,0x2B), C(0x2E,0x5A,0x9C), C(0x6B,0x3F,0xA0), C(0xD5,0xD8,0xDE), C(255,255,255)
PEA, GRY, LBL, LPU, GRN = C(0xFB,0xE9,0xDD), C(0xF3,0xF4,0xF6), C(0xE3,0xEC,0xF7), C(0xEE,0xE6,0xF5), C(0x2E,0x7D,0x32)
P = qn('p:cNvPr'); A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
M = json.load(open('img/bgswap.json')); W = M['runs'][0]['m4']['weights']; T = len(W)
KH, KE = M['runs'][0]['m4']['k_hold'], M['runs'][0]['m4']['k_end']
pc = lambda v: f'{round(100 * v)}%'
f2 = lambda v: f'{v:.2f}'.replace('.', ',')

def title(s): return next((x.text_frame.text.strip() for x in s.shapes if x.name == 'Text 1' and x.has_text_frame), '')
def find(prefix): return next(s for s in p.slides if title(s).startswith(prefix))
def tb(s, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=3, bullet=False):
    t = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in 'margin_left', 'margin_right', 'margin_top', 'margin_bottom': setattr(tf, m, 0)
    for k, line in enumerate(lines):
        para = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); para.alignment = align; para.space_after = Pt(space)
        if bullet: line = [('•  ', line[0][1], line[0][2], False)] + line
        for seg in line:
            txt, sz, col, b = seg[:4]
            r = para.add_run(); r.text = txt; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b; r.font.name = 'Arial'
            if len(seg) > 4: r.font.italic = seg[4]
    return t
def pic(s, f, x, y, z, hl=False, cap=None, capcol=None, cw=0.3, csz=9, bold=None):
    q = s.shapes.add_picture(f, I(x), I(y), I(z), I(z)); q.line.color.rgb = ORG if hl else LINE; q.line.width = Pt(2.25 if hl else 0.75)
    if cap: tb(s, x - cw / 2, y + z + 0.03, z + cw, 0.22, [[(cap, csz, capcol or (ORG if hl else MUT), hl if bold is None else bold)]], align=PP_ALIGN.CENTER)
    return q
def shape(s, kind, x, y, w, h, fill=None, line=None, lw=None):
    b = s.shapes.add_shape(kind, I(x), I(y), I(w), I(h)); b.shadow.inherit = False
    if fill is None: b.fill.background()
    else: b.fill.solid(); b.fill.fore_color.rgb = fill
    if line is None: b.line.fill.background()
    else: b.line.color.rgb = line; b.line.width = Pt(lw or 1)
    return b
def rbox(s, x, y, w, h, fill): b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill); b.adjustments[0] = 0.07; return b
def arrow(s, x, y, w=0.28, col=ORG): shape(s, MSO_SHAPE.RIGHT_ARROW, x, y, w, 0.24, col)
def band(s, x, y, w, text, col):
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, 0.27, col)
    tb(s, x, y, w, 0.27, [[(text, 9.5, WHT, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
def head(s, x, y, w, letter, text, col):
    shape(s, MSO_SHAPE.OVAL, x, y, 0.42, 0.42, col)
    tb(s, x, y, 0.42, 0.42, [[(letter, 13, WHT, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, x + 0.52, y - 0.02, w - 0.52, 0.48, [[(text, 12.5, col, True)]], anchor=MSO_ANCHOR.MIDDLE)
def ring(s, img_x, img_y, z, box, pad=14, col=ORG):
    k = z / 512; b = box
    shape(s, MSO_SHAPE.OVAL, img_x + (b[0] - pad) * k, img_y + (b[1] - pad) * k, (b[2] - b[0] + 2 * pad) * k, (b[3] - b[1] + 2 * pad) * k, None, col, 2.25)
def clear_body(s):
    for el in list(s.shapes._spTree.iterchildren()):
        if el.tag.endswith('AlternateContent'): el.getparent().remove(el); continue
        c = el.find('.//' + P)
        if c is None: continue
        off = el.find('.//' + A + 'off'); y = int(off.get('y')) if off is not None else 0; x = int(off.get('x')) if off is not None else 0
        if c.get('name') in ('Text 0', 'Text 1', 'Text 2', 'Text 3'): continue
        if y < I(0.7) and x > I(4.5): continue           # module tracker bar
        el.getparent().remove(el)
def notes(s, t): s.notes_slide.notes_text_frame.text = t

# ================================================================ M4 slide
s = find('M4-Tổng hợp cảnh')
clear_body(s)
tb(s, 0.6, 1.42, 12.1, 0.3, [[('Ví dụ thật (OCSD cuối cùng): cùng cảnh 2 xe buýt + 1 ghế, 30 bước khử nhiễu.', 12.5, MUT, True)]])
z = 1.6; y = 2.08
tb(s, 0.6, y - 0.3, z, 0.26, [[('Đầu vào', 11, INK, True)]], align=PP_ALIGN.CENTER)
pic(s, 'img/placed.png', 0.6, y, z, cap='vật M2 đặt khớp nét vẽ (xanh)', capcol=INK, cw=0.5, csz=8.5)
arrow(s, 2.3, y + z / 2 - 0.12, 0.36)
xs = [2.8, 4.5, 6.2, 7.9]; xf = 9.85
band(s, xs[0], y - 0.33, xs[-1] + z - xs[0], 'Ảnh thật đưa vào U-Net ở từng bước (viền cam = chỗ của vật M2)', BLU)
for k, st in enumerate((1, 12, 18, 24)):
    pic(s, f'img/st{st:02d}.png', xs[k], y, z, cap=f'bước {st}: giữ {pc(W[st - 1])}', capcol=INK, csz=9.5, bold=True)
arrow(s, xs[-1] + z + 0.06, y + z / 2 - 0.12, xf - xs[-1] - z - 0.12, PUR)
band(s, xf, y - 0.33, z, 'Ảnh sau M4', PUR)
pic(s, 'img/m4_out.png', xf, y, z, hl=True, cap='bước 30: tự do', csz=9.5)
tb(s, xf + z + 0.12, y, 12.75 - xf - z - 0.12, z, [[('2 xe buýt + 1 ghế, đúng chỗ đã vẽ', 10, ORG, True)], [('(ghế còn mờ → M5 sửa, slide sau)', 8.5, MUT, False, True)]], anchor=MSO_ANCHOR.MIDDLE, space=3)
# 30-cell strip of the real anchor weight w per step
x0, x1, ys = xs[0], xf + z, 4.1; cw = (x1 - x0) / T
def mix(c, w): return C(*[round(255 - (255 - v) * w) for v in c])
for i, w in enumerate(W):
    col = mix((0x2E, 0x5A, 0x9C), max(w, 0.0)) if w > 0 else C(0xDD, 0xCF, 0xEC)
    shape(s, MSO_SHAPE.RECTANGLE, x0 + i * cw, ys, cw - 0.015, 0.26, col)
    if i + 1 in (1, 12, 18, 24, 30):
        tb(s, x0 + i * cw - 0.2, ys + 0.27, cw + 0.4, 0.18, [[(str(i + 1), 8, INK, True)]], align=PP_ALIGN.CENTER, space=0)
tb(s, 0.6, ys - 0.02, 2.1, 0.3, [[('Lực giữ vật M2 (w)', 9.5, INK, True)], [('mỗi ô = một bước', 8, MUT, False, True)]], space=0)
lab = [(0, KH, f'Giữ chặt (bước 1–{KH}): w = 100%', BLU), (KH, KE, f'Nhả dần ({KH + 1}–{KE}): w giảm từ {pc(W[KH])} về {pc(W[KE - 1])}', BLU),
       (KE, T, f'Tự do ({KE + 1}–{T}): w = 0', PUR)]
for a, b, t, col in lab:
    tb(s, x0 + a * cw, ys + 0.47, (b - a) * cw, 0.22, [[(t, 8.5, col, True)]], align=PP_ALIGN.CENTER, space=0)
# worked calculation + what changed
rbox(s, 0.6, 5.0, 6.35, 1.6, LBL)
tb(s, 0.78, 5.08, 6.0, 1.5, [
    [('Cách trộn ở mỗi bước, trong vùng của vật:', 10.5, BLU, True)],
    [('ảnh mới = w × vật M2 (phủ nhiễu cho khớp bước) + (1 − w) × ảnh mô hình đang vẽ', 10.5, INK, False, True)],
    [('Ví dụ bước 18: w = ' + f2(W[17]) + f' → {pc(W[17])} vật M2 + {pc(1 - W[17])} mô hình tự vẽ; khi nhả chỉ giữ màu và dáng thô, chi tiết để mô hình vẽ.', 10, INK, False)],
    [('Ngoài vùng vật: nền khử nhiễu theo “… near a train station”.', 10, INK, False)]], space=3)
rbox(s, 7.1, 5.0, 5.6, 1.6, PEA)
tb(s, 7.28, 5.08, 5.25, 1.5, [
    [('Khác M4 cũ (dán cứng 27 bước, chỉ thả 3 bước cuối):', 10.5, ORG, True)],
    [('nhả dần thay vì dán cứng → vật hòa vào nền, hết vẻ “dán vào”;', 10, INK, False)],
    [('vật và nền chung một lần khử nhiễu, cùng một nhiễu;', 10, INK, False)],
    [('mặt nạ co 2 px, viền làm mềm 4 px, tô kín nền quanh vật → hết viền trắng;', 10, INK, False)],
    [('từ mới của M3 bật từ bước 14; ControlNet phác thảo 0,6 cả 30 bước.', 10, INK, False)]], space=2, bullet=True)
tb(s, 0.6, 6.68, 12.1, 0.3, [[('Lịch giữ/nhả được chọn từ 10 cấu hình thử trên tập phát triển, trước khi chạy tập kiểm thử (xem phần thực nghiệm).', 9.5, MUT, False, True)]])
notes(s, 'M4 dựng cả cảnh. Đầu vào là các vật M2 đã vẽ, đặt đúng chỗ của nét phác thảo, nét xanh trên hình. '
    'Bốn hình giữa là ảnh thật đưa vào U-Net ở bước 1, 12, 18 và 24 của chính lần chạy này, còn nguyên nhiễu; viền cam là chỗ của vật M2. '
    f'Thanh 30 ô bên dưới là lực giữ vật ở từng bước. Từ bước 1 đến {KH}, vật M2 được giữ chặt, w bằng 100 phần trăm, nên bố cục và số lượng được khóa sớm. '
    f'Từ bước {KH + 1} đến {KE}, lực giữ giảm dần; ví dụ ở bước 18, w bằng {f2(W[17])}, tức {pc(W[17])} lấy từ vật M2 và {pc(1 - W[17])} do mô hình tự vẽ, và lúc này chỉ giữ màu và dáng thô. '
    f'Sáu bước cuối mô hình vẽ tự do, nên viền, bóng và ánh sáng của vật khớp với nền. '
    'So với M4 cũ: M4 cũ dán cứng vật suốt 27 bước rồi chỉ thả 3 bước cuối, nên vật trông như dán vào và lộ viền trắng. M4 mới nhả dần, vật và nền khử nhiễu chung, mặt nạ được co và làm mềm. '
    'Trên 72 cảnh, độ có mặt của vật tăng từ 71 lên 79 phần trăm và ảnh tự nhiên hơn rõ rệt. Ở ảnh này ghế còn mờ; M5 phần d sẽ sửa, ở slide sau.')

# ================================================================ M5 slide
s = find('M5-Điều kiện hóa nhận biết đối tượng')
clear_body(s)
tb(s, 0.6, 1.42, 12.1, 0.35, [[('Trong lúc dựng cảnh, liên tục nhắc mô hình: vật nào, ở chỗ nào, mấy cái. Ví dụ thật: cảnh 2 xe buýt + 1 ghế.', 12.5, MUT, True)]])
rbox(s, 0.6, 1.95, 6.0, 4.55, PEA)
head(s, 0.78, 2.07, 5.7, 'a', 'Mỗi từ chỉ được vẽ trong khung của vật đó', ORG)
head(s, 0.78, 2.55, 5.7, 'b', 'Vật bị bỏ quên hoặc tràn ra ngoài → sửa ngay', BLU)
for i, nm in enumerate(['“orange bus” (dưới)', '“wooden bench”', '“orange bus” (trên)']):
    pic(s, f'img/attn_{i}.png', 0.8 + i * 1.92, 3.12, 1.72, cap=nm, capcol=INK, cw=0.2, csz=9)
tb(s, 0.8, 5.12, 5.65, 1.3, [[('Bản đồ chú ý thật ở bước 12. Màu đỏ = chỗ từ đó đang “vẽ”; khung trắng = chỗ của vật trong phác thảo.', 10.5, INK, False)],
                             [('(a) chặn từ vẽ ra ngoài khung; (b) nếu đỏ yếu trong khung hoặc lan ra ngoài, chỉnh ảnh ở chính bước đó.', 10.5, INK, False)],
                             [('→ đủ vật, đúng chỗ, không rò màu sang vật khác', 10.5, ORG, True)]], space=4)
rbox(s, 6.75, 1.95, 2.85, 4.55, C(0xE6, 0xF1, 0xE6))
head(s, 6.9, 2.07, 2.6, 'c', 'Nhắc lại phác thảo', GRN)
pic(s, 'img/sketch_on_final.png', 7.1, 2.6, 2.15, cap='nét trắng = phác thảo, đè lên ảnh cuối', capcol=INK, cw=0.5, csz=9)
om = M['cfg']['omega']; cwi = 2.5 / T
for k in range(T):
    shape(s, MSO_SHAPE.RECTANGLE, 6.92 + k * cwi, 5.08, cwi - 0.01, 0.17, GRN)
tb(s, 6.88, 5.26, 2.6, 0.2, [[(f'bước 1–{T}: {f2(om)[:3]} ở mọi bước', 8, GRN, True)]], space=0)
tb(s, 6.88, 5.5, 2.6, 0.95, [[(f'Lực nhắc = {f2(om)[:3]} × lực đầy đủ ', 10, INK, True), ('(M2 vẽ từng vật dùng 1,0)', 9, MUT, False)],
                            [('Đủ giữ dáng vật, nền vẫn tự do.', 10, INK, False)],
                            [('→ đúng hình dạng', 10.5, GRN, True)]], space=2)
rbox(s, 9.75, 1.95, 2.95, 4.55, LPU)
head(s, 9.9, 2.07, 2.7, 'd', 'Kiểm tra lần cuối', PUR)
bench = M['steps']['region_masks_boxes'][1]; zz = 1.3
for k, (f, cap) in enumerate((('img/m4_out.png', 'sau M4: ghế mờ'), ('img/final.png', 'sau khi sửa'))):
    x = 9.87 + k * 1.42
    pic(s, f, x, 2.6, zz, hl=k == 1, cap=cap, capcol=PUR if k == 1 else MUT, cw=0.1, csz=8.5)
    ring(s, x, 2.6, zz, bench, pad=10, col=PUR if k == 0 else ORG)
tb(s, 9.92, 4.2, 2.65, 2.2, [[('Đếm lại vật trong ảnh; thiếu hay lệch thì vẽ lại riêng vùng lỗi (tối đa 2 lần), giữ ảnh điểm cao nhất.', 10, INK, False)],
                             [('Ở đây ghế chưa rõ → vẽ lại vùng ghế; xe buýt giữ nguyên.', 10, INK, False)],
                             [('→ đúng số lượng', 10.5, PUR, True)]], space=4)
tb(s, 0.6, 6.62, 12.1, 0.3, [[('Bộ đếm ở (d) là Grounding DINO, khác OWLv2 dùng để chấm điểm, để phương pháp không “học tủ” theo công cụ chấm.', 9.5, MUT, False, True)]])
notes(s, 'M5 hoạt động trong lúc dựng cảnh, và luôn nhắc mô hình: vật nào, ở chỗ nào, mấy cái. Có bốn phần, minh họa bằng chính cảnh hai xe buýt và một ghế của OCSD cuối cùng. '
    'Phần a và b nhìn vào bản đồ chú ý thật ở bước 12: màu đỏ là chỗ một từ đang vẽ, khung trắng là chỗ của vật trong phác thảo. Phần a chặn từ vẽ ra ngoài khung; phần b, nếu đỏ yếu trong khung hoặc lan ra ngoài, thì chỉnh ảnh ngay ở bước đó. Ở ảnh thật, đỏ của mỗi cụm từ nằm gọn trong khung của vật đó. '
    f'Phần c đưa phác thảo vào lần nữa qua ControlNet, nhẹ tay và ở cả 30 bước: lực {f2(om)[:3]}, tức {round(om * 100)} phần trăm so với lực đầy đủ 1,0 mà M2 dùng khi vẽ từng vật. Đủ giữ dáng vật mà không ép cả nền theo nét. '
    'Phần d đếm lại vật trong ảnh bằng Grounding DINO. Ở ví dụ này, sau M4 ghế còn mờ, gần như lẫn vào tường, nên M5 vẽ lại riêng vùng ghế; ảnh bên phải là ảnh sau khi sửa, hai xe buýt giữ nguyên.')

# ================================================================ qualitative slide: all columns from the m4_study_v1 run
s = find('Cùng một phác thảo, các phương pháp vẽ ra sao?')
rows = [('Picture 47', 'Picture 48', 'Picture 49', 'Picture 50', 'qd_1_medium_003'),
        ('Picture 52', 'Picture 53', 'Picture 54', 'Picture 55', 'qd_3_medium_003'),
        ('Picture 57', 'Picture 58', 'Picture 59', 'Picture 60', 'qd_8p_complex_003')]
byname = {x.name: x for x in s.shapes}
for *names, sid in rows:
    for nm, f in zip(names, (f'img/q_sketch_{sid}.png', f'img/q_controlnet_{sid}.jpg', f'img/q_gligen_{sid}.jpg', f'img/q_ocsd_final_{sid}.jpg')):
        old = byname[nm]; new = s.shapes.add_picture(f, old.left, old.top, old.width, old.height)
        old._element.addprevious(new._element); old._element.getparent().remove(old._element)
for x in s.shapes:
    if x.name == 'Text 27':
        r = x.text_frame.paragraphs[0].runs; r[0].text = 'Ảnh thật (hạt giống 0) của lần chạy cuối, cùng cấu hình OCSD cuối (SD 1.5, đủ năm mô-đun, M4 nhả dần) với bảng kết quả. OCSD cũng có ca sai (phần hạn chế).'
        for q in r[1:]: q.text = ''
notes(s, 'Đây là ảnh thật của lần chạy cuối; mỗi hàng là một phác thảo, mỗi cột là một phương pháp. Hàng đầu, con bò xám bên hồ: mô hình nền không vẽ ra con bò, chỉ có một cái bè; GLIGEN vẽ bò nhưng màu đen; OCSD vẽ con bò đứng đúng chỗ đã phác thảo, liền khối với nền. '
    'Hàng hai, hai xe buýt cam và một ghế ở nhà ga: mô hình nền phủ cả ảnh một mảng cam; GLIGEN có đủ hai xe và ghế nhưng không có nhà ga; OCSD có hai xe buýt đúng chỗ trước nhà ga, nhưng ghế dính vào tường nhà. '
    'Hàng ba, cảnh rất đông: mọi phương pháp đều còn sai; OCSD có chó, gấu và nhà trong rừng, nhưng chó sai màu so với câu mô tả. Nhóm trình bày ca khó này ở phần hạn chế.')

# ================================================================ new slide: old M4 vs new M4 (after the comparison slide)
NT = 'M4 mới so với M4 cũ: cùng phác thảo, vật hòa vào nền'
lst = p.slides._sldIdLst
for i, sl in enumerate(list(p.slides)):
    if title(sl) == NT: sid_ = lst[i]; p.part.drop_rel(sid_.rId); lst.remove(sid_)
src_i = next(i for i, sl in enumerate(p.slides) if title(sl) == 'OCSD so với các phương pháp khác')
src = p.slides[src_i]
s = p.slides.add_slide(src.slide_layout)
for sh in list(s.placeholders): sh._element.getparent().remove(sh._element)
for sh in src.shapes:
    if sh.name in ('Text 0', 'Text 1', 'Text 2', 'Text 3'): s.shapes._spTree.append(copy.deepcopy(sh._element))
for sh in s.shapes:
    if sh.name == 'Text 1':
        r = sh.text_frame.paragraphs[0].runs; r[0].text = NT
        for q in r[1:]: q.text = ''
new = lst[-1]; lst.remove(new); lst.insert(src_i + 1, new)
S4 = ['qd_1_medium_004', 'qd_3_simple_005', 'qd_5_simple_003', 'qd_5_medium_003']
z = 1.45; x0 = 1.75; dx = 1.6; ys = [1.55, 3.12, 4.69]
for r, (lab, col) in enumerate((('Phác thảo', INK), ('M4 cũ', MUT), ('M4 mới\n(OCSD cuối)', ORG))):
    tb(s, 0.6, ys[r], 1.05, z, [[(t, 11.5, col, True)] for t in lab.split('\n')], anchor=MSO_ANCHOR.MIDDLE, space=0)
for c, sid in enumerate(S4):
    x = x0 + c * dx; sc = json.load(open(f'img/q_scene_{sid}.json'))
    pic(s, f'img/q_sketch_{sid}.png', x, ys[0], z)
    pic(s, f'img/q_ocsd_{sid}.jpg', x, ys[1], z)
    pic(s, f'img/q_ocsd_final_{sid}.jpg', x, ys[2], z, hl=True)
    tb(s, x - 0.05, ys[2] + z + 0.04, z + 0.1, 0.4, [[('“' + sc['caption'] + '”', 7.5, MUT, False, True)]], align=PP_ALIGN.CENTER, space=0)
xr = 8.3
rbox(s, xr, 1.55, 4.4, 4.6, GRY)
tb(s, xr + 0.18, 1.63, 4.05, 0.3, [[('72 cảnh × 2 hạt giống, cùng M1–M3, M5', 11, INK, True)]])
rows = [('Độ tự nhiên, chấm mù (0–2)', '0,44', '1,36', True), ('PickScore (ảnh đẹp, khớp câu)', '21,16', '21,89', True),
        ('Có mặt đối tượng (OPR)', '71%', '79%', True), ('Đúng tương quan (RA)', '45%', '51%', False), ('Đúng vị trí (mIoU)', '0,53', '0,48', False)]
yy = 2.05
tb(s, xr + 2.55, yy, 0.8, 0.22, [[('M4 cũ', 9.5, MUT, True)]], align=PP_ALIGN.CENTER, space=0)
tb(s, xr + 3.35, yy, 0.9, 0.22, [[('M4 mới', 9.5, ORG, True)]], align=PP_ALIGN.CENTER, space=0)
for k, (nm, a, b, sig) in enumerate(rows):
    y = yy + 0.3 + k * 0.42
    shape(s, MSO_SHAPE.RECTANGLE, xr + 0.15, y - 0.04, 4.1, 0.38, WHT)
    tb(s, xr + 0.22, y, 2.35, 0.3, [[(nm, 9.5, INK, False)]], anchor=MSO_ANCHOR.MIDDLE, space=0)
    tb(s, xr + 2.55, y, 0.8, 0.3, [[(a, 10.5, MUT, False)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0)
    tb(s, xr + 3.35, y, 0.9, 0.3, [[(b + (' *' if sig else ''), 10.5, ORG if sig else INK, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0)
tb(s, xr + 0.18, 4.5, 4.05, 1.6, [
    [('* khác biệt có ý nghĩa thống kê; RA, mIoU: khác biệt không có ý nghĩa.', 8.5, MUT, False, True)],
    [('Vật hết viền trắng, hết vẻ “dán vào”, ánh sáng và bóng khớp nền.', 10, INK, True)],
    [('Đánh đổi: khi nhả vật, vật có thể xê dịch nhẹ (mIoU 0,53 → 0,48). GLIGEN vẫn tự nhiên hơn khi chấm mù (1,81).', 9.5, INK, False)]], space=4)
tb(s, 0.6, 6.62, 12.1, 0.3, [[('Ảnh thật, hạt giống 0. Chấm mù: 36 phác thảo, 8 phương pháp xáo trộn, một người chấm, chấm xong mới mở khóa.', 9, MUT, False, True)]])
notes(s, 'Slide này so sánh trực tiếp M4 cũ trong khóa luận với M4 mới của OCSD cuối cùng; M1, M2, M3 và M5 giữ nguyên. Hàng giữa là M4 cũ: con chó, con bò, con chim trông như được cắt dán lên nền, có viền và lệch ánh sáng. '
    'Hàng dưới là M4 mới, cùng phác thảo và cùng hạt giống: vật nằm trong cảnh, có bóng, ánh sáng khớp nền. Trên 72 cảnh và 2 hạt giống: điểm tự nhiên khi chấm mù tăng từ 0,44 lên 1,36 trên thang 0 đến 2, PickScore tăng từ 21,16 lên 21,89, '
    'và độ có mặt của vật tăng từ 71 lên 79 phần trăm; ba khác biệt này có ý nghĩa thống kê. Đúng tương quan tăng và đúng vị trí giảm nhẹ, nhưng hai khác biệt này không có ý nghĩa thống kê. '
    'Nhóm nói rõ: khi chấm mù, GLIGEN vẫn tự nhiên hơn OCSD, và việc chấm mù chỉ có một người chấm.')

# ================================================================ limitation picture
s = find('Hạn chế và hướng phát triển')
LS = 'qd_8p_simple_005'
byname = {x.name: x for x in s.shapes}
for nm, f in (('Picture 30', f'img/q_sketch_{LS}.png'), ('Picture 31', f'img/q_ocsd_final_{LS}.jpg')):
    old = byname[nm]; new = s.shapes.add_picture(f, old.left, old.top, old.width, old.height)
    old._element.addprevious(new._element); old._element.getparent().remove(old._element)
for nm, t in (('Text 6', 'Phác thảo 9 đối tượng'), ('Text 7', 'OCSD: vật như đồ chơi, nổi rời trên nền')):
    r = byname[nm].text_frame.paragraphs[0].runs; r[0].text = t
    for q in r[1:]: q.text = ''

# ================================================================ M3 final image + architecture notes
s = find('M3-Học định danh')
for x in list(s.shapes):
    if x.shape_type == 13 and abs(x.left - I(0.8)) < I(0.05) and abs(x.top - I(5.1)) < I(0.05):
        new = s.shapes.add_picture('img/final.png', x.left, x.top, x.width, x.height)
        new.line.color.rgb = LINE; new.line.width = Pt(0.75)
        x._element.addprevious(new._element); x._element.getparent().remove(x._element)
for x in s.shapes:     # the orange ring vanishes on the orange station of the new image: make it white
    if x.name.startswith('Oval') and x.top > I(5.0) and x.left < I(2.6): x.line.color.rgb = WHT; x.line.width = Pt(3)
    if x.has_text_frame:
        for para in x.text_frame.paragraphs:
            for r in para.runs: r.text = r.text.replace('(khoanh cam)', '(khoanh trắng)')
tf = s.notes_slide.notes_text_frame; tf.text = tf.text.replace('chỗ khoanh cam', 'chỗ khoanh trắng')
s = find('Kiến trúc OCSD')
tf = s.notes_slide.notes_text_frame
tf.text = tf.text.replace('M4 trộn tiềm ẩn rồi suy luận tùy biến', 'M4 dựng cảnh: giữ vật của M2 ở các bước đầu rồi nhả dần để vật hòa vào nền')

# ================================================================ renumber
n = len(p.slides)
for k, sl in enumerate(p.slides, 1):
    for x in sl.shapes:
        if x.has_text_frame and re.fullmatch(r'\s*\d+ / \d+\s*', x.text_frame.text):
            r = x.text_frame.paragraphs[0].runs; r[0].text = f'{k} / {n}'
            for q in r[1:]: q.text = ''
p.save(sys.argv[2]); print('slides', n)
