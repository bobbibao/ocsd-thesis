"""24-11 update of the defense deck:
(1) write only "OCSD" (no "-XL") everywhere; (2) seven diffusion-model explainer slides after
slide 13, then slide 13 again with the Stable Diffusion half highlighted; (3) renumber pages.
Usage: python build_2411.py SRC.pptx FIGDIR CHARTDIR OUT_raw.pptx   (then omml.py for equations)"""
import copy, re, sys
from lxml import etree
from pptx import Presentation
from pptx.util import Inches as I, Pt, Emu
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

SRC, FIG, CH, OUT = sys.argv[1:5]
p = Presentation(SRC)
INK, MUT, ORG, BLU = C(0x1F, 0x24, 0x30), C(0x5B, 0x62, 0x70), C(0xD9, 0x69, 0x2B), C(0x2E, 0x5A, 0x9C)
GRY, PEA, LBL, YEL, WHT, LINE = C(0xF3, 0xF4, 0xF6), C(0xFB, 0xE9, 0xDD), C(0xE3, 0xEC, 0xF7), C(0xFF, 0xF3, 0xCD), C(255, 255, 255), C(0xD5, 0xD8, 0xDE)
slides = list(p.slides)
S13 = slides[12]

# ------------------------------------------------------------------ (1) OCSD-XL -> OCSD
REPL = [
    ('Bản cải tiến OCSD-XL chạy cùng năm mô-đun trên SDXL.', 'Bản cải tiến của OCSD chạy cùng năm mô-đun trên SDXL.'),
    ('(OCSD-XL: 1024 × 1024)', '(OCSD trên SDXL: 1024 × 1024)'),
    ('bản cải tiến OCSD-XL dùng SDXL', 'bản cải tiến của OCSD dùng SDXL'),
    ('OCSD-XL: cùng năm mô-đun, trên mô hình ảnh mạnh hơn', 'OCSD cải tiến: cùng năm mô-đun, trên mô hình ảnh mạnh hơn'),
    ('đặt trên SDXL (OCSD-XL) thì tăng rõ', 'đặt trên SDXL thì tăng rõ'),
    ('OCSD-XL', 'OCSD'),
]
def walk(shapes):
    for sh in shapes:
        if sh.shape_type == 6: yield from walk(sh.shapes)
        else: yield sh
def fix_text(el):
    for t in el.iter(qn('a:t')):
        if t.text and 'XL' in t.text:
            for a, b in REPL: t.text = t.text.replace(a, b)
for s in slides:
    fix_text(s._element)
    if s.has_notes_slide: fix_text(s.notes_slide._element)
# split-run check
for i, s in enumerate(slides, 1):
    for sh in walk(s.shapes):
        if sh.has_text_frame and re.search('OCSD.?-.?XL', sh.text_frame.text): print('LEFT', i, sh.name)

# charts with the old legend/labels (slide 32: by count, slide 33: three bar charts)
def repic(s, old, path):
    new = s.shapes.add_picture(path, old.left, old.top, old.width, old.height)
    old._element.addprevious(new._element); old._element.getparent().remove(old._element)
def pics(s): return [x for x in s.shapes if x.shape_type == 13]
s32, s33 = slides[31], slides[32]
pc = pics(s32); assert len(pc) == 1, pc; repic(s32, pc[0], f'{CH}/c_count.png')
pc = sorted(pics(s33), key=lambda x: x.left); assert len(pc) == 3, pc
for x, f in zip(pc, ['c_opr', 'c_miou', 'c_nat']): repic(s33, x, f'{CH}/{f}.png')

# ------------------------------------------------------------------ helpers
def run(par, txt, sz, col=INK, b=False, it=False):
    r = par.add_run(); r.text = txt; f = r.font; f.size = Pt(sz); f.color.rgb = col; f.bold = b; f.italic = it; f.name = 'Arial'; return r
def tb(s, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=4, bullet=False):
    """paras: list of lines; line = list of (text, size, color, bold) or a str"""
    t = s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in 'margin_left', 'margin_right', 'margin_top', 'margin_bottom': setattr(tf, m, 0)
    for k, line in enumerate(paras):
        par = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); par.alignment = align; par.space_after = Pt(space)
        for seg in line:
            txt, sz, col, b = (seg + (False,))[:4] if len(seg) == 3 else seg
            run(par, txt, sz, col, b)
    return t
def box(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, adj=None):
    b = s.shapes.add_shape(shape, I(x), I(y), I(w), I(h)); b.fill.solid(); b.fill.fore_color.rgb = fill
    if line is None: b.line.fill.background()
    else: b.line.color.rgb = line; b.line.width = Pt(1.25)
    b.shadow.inherit = False
    if adj is not None: b.adjustments[0] = adj
    return b
def pic(s, f, x, y, w, h=None, border=None):
    im = s.shapes.add_picture(f, I(x), I(y), I(w), I(h) if h else None)
    if border: im.line.color.rgb = border; im.line.width = Pt(0.75)
    return im
def arrow(s, x1, y1, x2, y2, col, w=2.0):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(x1), I(y1), I(x2), I(y2))
    c.line.color.rgb = col; c.line.width = Pt(w)
    ln = c.line._get_or_add_ln(); te = etree.SubElement(ln, qn('a:tailEnd')); te.set('type', 'triangle'); te.set('w', 'med'); te.set('len', 'med')
    return c
def card(s, x, y, w, h, fill, head, hcol, body, hs=14, bs=12, pad=0.18):
    box(s, x, y, w, h, fill)
    return tb(s, x + pad, y + 0.12, w - 2 * pad, h - 0.2, [[(head, hs, hcol, True)]] + [[seg if isinstance(seg, tuple) else (seg, bs, INK, False) for seg in (b if isinstance(b, list) else [b])] for b in body], space=5)
def label(s, x, y, w, txt, sz=10, col=MUT, b=False, align=PP_ALIGN.CENTER):
    return tb(s, x, y, w, 0.3, [[(txt, sz, col, b)]], align=align)
def notes(s, txt): s.notes_slide.notes_text_frame.text = txt

HEAD = {sh.name: sh for sh in S13.shapes}
def new_slide(title):
    s = p.slides.add_slide(S13.slide_layout)
    for ph in list(s.placeholders): ph._element.getparent().remove(ph._element)
    for nm in ('Text 0', 'Text 1', 'TextBox 32', 'Text 3'):
        s.shapes._spTree.append(copy.deepcopy(HEAD[nm]._element))
    for sh in s.shapes:
        if sh.name == 'Text 1': sh.text_frame.paragraphs[0].runs[0].text = title
    return s

def dup_slide(src):
    s = p.slides.add_slide(src.slide_layout)
    for ph in list(s.placeholders): ph._element.getparent().remove(ph._element)
    rid = {}
    for r in src.part.rels.values():
        if 'image' in r.reltype: rid[r.rId] = s.part.relate_to(r._target, r.reltype)
    for el in src.shapes._spTree.iterchildren():
        if el.tag in (qn('p:nvGrpSpPr'), qn('p:grpSpPr')): continue
        e = copy.deepcopy(el)
        for node in e.iter():
            for att in (qn('r:embed'), qn('r:link'), qn('r:id')):
                if node.get(att) in rid: node.set(att, rid[node.get(att)])
        s.shapes._spTree.append(e)
    return s

F = lambda n: f'{FIG}/{n}.png'
SZ = 12  # body text size

# ------------------------------------------------------------------ A. idea + ink analogy
s = new_slide('Ý tưởng: làm “tan” dần bức ảnh, rồi học cách làm ngược lại')
X0, IW, GAP = 2.75, 1.42, 0.24
rows = [('Giọt mực trong cốc nước', 'hiện tượng vật lý', [F('ink_0'), F('ink_1'), F('ink_2'), F('ink_3')], ['vừa nhỏ mực', 'vài giây', 'một phút', 'vài giờ']),
        ('Bức ảnh + nhiễu', 'mô hình khuếch tán', [F('fwd_0'), F('fwd_200'), F('fwd_300'), F('fwd_1000')], ['ảnh gốc', 'thêm ít nhiễu', 'thêm nhiều nhiễu', 'chỉ còn nhiễu'])]
for r, (h1, h2, ims, caps) in enumerate(rows):
    y = 1.55 + r * 1.95
    tb(s, 0.6, y + 0.45, 2.0, 0.8, [[(h1, 14, INK, True)], [(h2, 11, MUT, False)]])
    for k, (f, c) in enumerate(zip(ims, caps)):
        x = X0 + k * (IW + GAP); pic(s, f, x, y, IW, IW, border=LINE if r else None); label(s, x - 0.1, y + IW + 0.04, IW + 0.2, c, 10)
xe = X0 + 4 * IW + 3 * GAP
tb(s, xe + 0.35, 1.75, 12.73 - xe - 0.35, 1.3, [[('Chỉ đi một chiều: ', SZ, INK, True), ('mực tự tan đều, không bao giờ tự gom lại thành giọt.', SZ, INK)]])
tb(s, xe + 0.35, 3.7, 12.73 - xe - 0.35, 1.3, [[('Giống hệt: ', SZ, INK, True), ('cộng nhiễu ngẫu nhiên nhiều lần thì ảnh “tan” thành nhiễu, mất hết nội dung.', SZ, INK)]])
ya = 5.5
arrow(s, X0, ya, xe, ya, ORG, 2.5); arrow(s, xe, ya + 0.42, X0, ya + 0.42, BLU, 2.5)
tb(s, xe + 0.35, ya - 0.15, 12.73 - xe - 0.35, 0.3, [[('Quá trình thuận: ', SZ, ORG, True), ('thêm nhiễu, dễ, không cần học', SZ, INK)]])
tb(s, xe + 0.35, ya + 0.27, 12.73 - xe - 0.35, 0.3, [[('Quá trình ngược: ', SZ, BLU, True), ('khử nhiễu, khó, mạng nơ-ron phải học', SZ, INK)]])
box(s, 0.6, 6.25, 12.13, 0.7, YEL)
tb(s, 0.8, 6.3, 11.73, 0.6, [[('Ý tưởng cốt lõi: ', SZ, INK, True), ('nếu học được cách đi ngược từng bước nhỏ, ta bắt đầu từ một nhiễu ngẫu nhiên bất kỳ và “gom” nó lại thành một ảnh mới. Tên “khuếch tán” lấy từ hiện tượng vật lý này (Sohl-Dickstein và cs., ICML 2015).', SZ, INK)]], anchor=MSO_ANCHOR.MIDDLE)
notes(s, 'Trước khi nói Stable Diffusion, nhóm em giải thích kỹ mô hình khuếch tán qua một ví dụ đời thường. Nhỏ một giọt mực vào cốc nước: mực tự tan đều ra cả cốc, nhưng không bao giờ tự gom lại thành giọt. Mô hình khuếch tán làm điều tương tự với ảnh: cộng nhiễu ngẫu nhiên từng chút một, đến cuối ảnh chỉ còn nhiễu. Chiều thêm nhiễu này dễ, không cần học gì. Chiều ngược lại, từ nhiễu về ảnh, mới khó, và đó chính là việc mạng nơ-ron phải học. Nếu học được cách đi ngược từng bước nhỏ, ta có thể bắt đầu từ một nhiễu bất kỳ và gom nó lại thành một ảnh mới. Cái tên khuếch tán lấy từ hiện tượng vật lý này, do Sohl-Dickstein và cộng sự đề xuất năm 2015.')
A = s

# ------------------------------------------------------------------ B. forward process
s = new_slide('Quá trình thuận: thêm nhiễu Gauss từng chút một')
ts = [0, 100, 200, 300, 500, 1000]; keep = {0: '100%', 100: '95%', 200: '81%', 300: '63%', 500: '28%', 1000: '≈ 0%'}
W6, G6 = 1.85, (12.13 - 6 * 1.85) / 5
for k, t in enumerate(ts):
    x = 0.6 + k * (W6 + G6); pic(s, F(f'fwd_{t}'), x, 1.5, W6, W6, border=LINE)
    tb(s, x, 1.5 + W6 + 0.05, W6, 0.5, [[(f't = {t}', 11, INK, True)], [(f'ảnh còn {keep[t]}', 10, MUT)]], align=PP_ALIGN.CENTER, space=0)
box(s, 0.6, 3.95, 6.0, 1.55, GRY)
tb(s, 0.8, 4.03, 5.6, 1.45, [
    [('Mỗi bước thêm một ít nhiễu: ', SZ, INK, True), ('⟦q(x_t|x_{t-1})=\\mathcal{N}(\\sqrt{1-\\beta_t}\\,x_{t-1},\\,\\beta_t I)⟧', 13, INK)],
    [('Nhảy thẳng tới bước t bất kỳ: ', SZ, INK, True)],
    [('⟦x_t=\\sqrt{\\bar{\\alpha}_t}\\,x_0+\\sqrt{1-\\bar{\\alpha}_t}\\,\\varepsilon,\\quad \\varepsilon\\sim\\mathcal{N}(0,I)⟧', 15, BLU)],
    [('⟦\\beta_t⟧: lượng nhiễu ở bước t (0,0001 → 0,02);  ⟦\\bar{\\alpha}_t=(1-\\beta_1)\\cdots(1-\\beta_t)⟧;  T = 1000.', 10.5, MUT)]], space=4)
box(s, 0.6, 5.62, 6.0, 1.33, PEA)
tb(s, 0.8, 5.68, 5.6, 1.25, [
    [('Ví dụ: ', SZ, ORG, True), ('một điểm ảnh sáng 0,8, ở bước t = 500: ⟦x_{500}=0{,}28\\times 0{,}8+0{,}96\\times\\varepsilon⟧, tức chỉ còn 0,22 của ảnh, phần lớn là nhiễu.', SZ, INK)],
    [('Như pha loãng cà phê: ', SZ, INK, True), ('mỗi bước rót bớt một ít cà phê, thêm một ít nước; sau 1000 bước chỉ còn nước.', SZ, INK)]], space=4)
pic(s, F('curve'), 7.0, 3.92, 5.45)
notes(s, 'Quá trình thuận: ở mỗi bước, ta làm mờ ảnh đi một chút và cộng thêm một ít nhiễu Gauss, với lượng nhiễu beta t tăng dần từ 0,0001 đến 0,02, tổng cộng T bằng 1000 bước. Điểm hay là không cần lặp từng bước: công thức màu xanh cho phép nhảy thẳng tới bước t bất kỳ, với căn alpha ngang t là phần ảnh gốc còn lại. Dãy ảnh phía trên tính đúng bằng công thức này: ở bước 300 ảnh còn 63%, bước 500 còn 28%, bước 1000 gần như chỉ còn nhiễu. Ví dụ một điểm ảnh sáng 0,8, ở bước 500 chỉ còn 0,28 nhân 0,8, còn lại là nhiễu. Giống như pha loãng cà phê: mỗi bước rót bớt cà phê, thêm nước. Biểu đồ bên phải cho thấy phần ảnh giảm dần, phần nhiễu tăng dần.')
B = s

# ------------------------------------------------------------------ C. training
s = new_slide('Huấn luyện: dạy mạng nơ-ron đoán phần nhiễu đã thêm vào')
y0, IW = 1.6, 1.45
pic(s, F('tr_x0'), 0.6, y0, IW, IW, border=LINE); label(s, 0.45, y0 + IW + 0.05, IW + 0.3, 'ảnh thật ⟦x_0⟧', 11, INK)
tb(s, 2.1, y0 + 0.5, 0.35, 0.5, [[('+', 24, INK, True)]], align=PP_ALIGN.CENTER)
pic(s, F('tr_eps'), 2.5, y0, IW, IW, border=LINE); label(s, 2.2, y0 + IW + 0.05, IW + 0.6, 'nhiễu ⟦\\varepsilon⟧ (biết trước)', 11, INK)
arrow(s, 4.0, y0 + IW / 2, 4.45, y0 + IW / 2, MUT, 2)
pic(s, F('tr_xt'), 4.5, y0, IW, IW, border=LINE); label(s, 4.2, y0 + IW + 0.05, IW + 0.6, 'ảnh nhiễu ⟦x_t⟧', 11, INK)
arrow(s, 6.0, y0 + IW / 2, 6.45, y0 + IW / 2, MUT, 2)
u = box(s, 6.5, y0 + 0.1, 1.9, IW - 0.2, LBL, line=BLU, shape=MSO_SHAPE.ROUNDED_RECTANGLE, adj=0.12)
tb(s, 6.55, y0 + 0.2, 1.8, IW - 0.4, [[('U-Net khử nhiễu', 12, BLU, True)], [('⟦\\varepsilon_\\theta(x_t,t)⟧', 13, BLU)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
arrow(s, 8.45, y0 + IW / 2, 8.9, y0 + IW / 2, MUT, 2)
pic(s, F('tr_epshat'), 8.95, y0, IW, IW, border=LINE); label(s, 8.65, y0 + IW + 0.05, IW + 0.6, 'nhiễu mạng đoán ⟦\\hat{\\varepsilon}⟧', 11, INK)
box(s, 10.65, y0 + 0.05, 2.08, IW - 0.1, YEL)
tb(s, 10.75, y0 + 0.1, 1.88, IW - 0.2, [[('So sánh với ⟦\\varepsilon⟧', 12, INK, True)], [('⟦\\mathcal{L}=\\|\\varepsilon-\\varepsilon_\\theta(x_t,t)\\|^2⟧', 13, INK)], [('sai càng ít càng tốt', 10, MUT)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=3)
tb(s, 0.6, 3.75, 6.2, 0.35, [[('Một bước huấn luyện (Ho và cs., NeurIPS 2020)', 14, BLU, True)]])
steps = ['Lấy một ảnh thật ⟦x_0⟧ trong tập dữ liệu.',
         'Chọn ngẫu nhiên một bước t (từ 1 đến 1000) và một nhiễu ⟦\\varepsilon⟧.',
         'Tạo ảnh nhiễu ⟦x_t⟧ bằng công thức nhảy thẳng ở slide trước.',
         'Mạng đoán nhiễu; đoán sai bao nhiêu thì chỉnh trọng số bấy nhiêu. Lặp lại hàng trăm triệu lần trên hàng trăm triệu ảnh.']
for k, st in enumerate(steps):
    yy = 4.2 + k * 0.62
    c = box(s, 0.6, yy, 0.38, 0.38, BLU, shape=MSO_SHAPE.OVAL)
    tb(s, 0.6, yy + 0.02, 0.38, 0.34, [[(str(k + 1), 12, WHT, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, 1.15, yy + 0.02, 5.65, 0.6, [[(st, SZ, INK)]])
card(s, 7.1, 3.75, 5.63, 1.55, PEA, 'Ví dụ dễ hiểu: đề bài có sẵn đáp án', ORG,
     ['Thầy giáo tự làm bẩn một bức ảnh bằng vết bẩn mình đã biết, rồi hỏi học trò: “vết bẩn trông thế nào?”. Vì thầy biết đáp án nên không cần ai gán nhãn, chỉ cần thật nhiều ảnh.'], hs=13)
card(s, 7.1, 5.45, 5.63, 1.15, GRY, 'Vì sao chọn t ngẫu nhiên?', INK,
     ['Để mạng giỏi ở mọi mức nhiễu: nhiễu nặng thì đoán bố cục, nhiễu nhẹ thì sửa chi tiết.'], hs=13)
notes(s, 'Huấn luyện mô hình khuếch tán rất đơn giản. Lấy một ảnh thật x0, chọn ngẫu nhiên một bước t và một nhiễu epsilon, dùng công thức nhảy thẳng để tạo ảnh nhiễu xt. Đưa xt và t vào mạng U-Net, mạng đoán xem nhiễu đã thêm vào là gì. Vì ta tự thêm nhiễu nên ta biết đáp án đúng, và hàm mất mát chỉ là bình phương sai khác giữa nhiễu thật và nhiễu mạng đoán. Giống như thầy giáo tự làm bẩn một bức ảnh bằng vết bẩn mình biết rồi hỏi học trò vết bẩn trông thế nào: không cần ai gán nhãn, chỉ cần nhiều ảnh. Chọn t ngẫu nhiên để mạng giỏi ở mọi mức nhiễu, nhiễu nặng thì học đoán bố cục, nhiễu nhẹ thì học sửa chi tiết. Đề tài không huấn luyện lại mạng này từ đầu, mà dùng U-Net đã huấn luyện sẵn của Stable Diffusion.')
Cc = s

# ------------------------------------------------------------------ D. reverse process
s = new_slide('Quá trình ngược: mỗi bước trừ bớt phần nhiễu mạng đoán')
rt = [1000, 500, 300, 200, 100, 0]
for k, t in enumerate(rt):
    x = 0.6 + k * (W6 + G6); pic(s, F(f'rev_0_{t}'), x, 1.5, W6, W6, border=LINE)
    label(s, x, 1.5 + W6 + 0.05, W6, f't = {t}', 11, INK, True)
tb(s, 0.6, 3.68, 12.13, 0.25, [[('Dãy ảnh tính thật bằng công thức bên dưới (T = 1000), dùng bộ khử nhiễu lý tưởng trên 12 ảnh mẫu thay cho U-Net để minh họa; ảnh 128 × 128.', 9, MUT)]])
box(s, 0.6, 4.05, 6.2, 2.75, GRY)
tb(s, 0.8, 4.13, 5.8, 2.8, [
    [('Từ ⟦x_t⟧ về ⟦x_{t-1}⟧ (một bước khử nhiễu):', SZ, INK, True)],
    [('⟦x_{t-1}=\\frac{1}{\\sqrt{\\alpha_t}}\\left(x_t-\\frac{\\beta_t}{\\sqrt{1-\\bar{\\alpha}_t}}\\,\\varepsilon_\\theta(x_t,t)\\right)+\\sigma_t z⟧', 15, BLU)],
    [('Ký hiệu', 11, INK, True)],
    [('⟦\\varepsilon_\\theta(x_t,t)⟧: nhiễu mạng đoán (đã học ở slide trước)', 11, INK)],
    [('⟦\\alpha_t=1-\\beta_t⟧; ⟦\\bar{\\alpha}_t⟧ như quá trình thuận', 11, INK)],
    [('⟦\\sigma_t z⟧, ⟦z\\sim\\mathcal{N}(0,I)⟧: thêm lại một chút nhiễu mới để ảnh đa dạng', 11, INK)],
    [('Nghĩa: trừ bớt một phần nhiễu đã đoán, không trừ hết một lần.', 11, MUT)]], space=5)
card(s, 7.05, 4.05, 5.68, 2.75, PEA, 'Ví dụ dễ hiểu: nhà điêu khắc', ORG,
     ['“Bức tượng nằm sẵn trong khối đá, chỉ cần đẽo bỏ phần thừa.”',
      'Khối đá là nhiễu. Mỗi bước, mạng chỉ ra đâu là phần thừa (nhiễu) để đẽo bớt một chút.',
      [('Đẽo từ thô đến tinh: ', SZ, INK, True), ('các bước đầu định ra bố cục (vị trí, hình dạng đối tượng), các bước cuối mới thêm chi tiết. Trong dãy ảnh trên, đối tượng hiện ra từ khoảng t = 300.', SZ, INK, False)]], hs=13)
notes(s, 'Quá trình ngược đi từ nhiễu về ảnh. Ở mỗi bước, mạng đoán nhiễu trong ảnh hiện tại, rồi ta trừ bớt một phần nhiễu đó theo công thức màu xanh, cộng lại một chút nhiễu mới nhỏ để ảnh đa dạng. Không trừ hết một lần mà trừ dần qua 1000 bước. Dãy ảnh phía trên được tính thật bằng đúng công thức này; để minh họa trên máy thường, nhóm em thay U-Net bằng bộ khử nhiễu lý tưởng trên 12 ảnh mẫu. Có thể hình dung như nhà điêu khắc: bức tượng nằm sẵn trong khối đá, chỉ cần đẽo bỏ phần thừa. Khối đá là nhiễu, mạng chỉ ra phần thừa để đẽo dần. Quan trọng: đẽo từ thô đến tinh, các bước đầu quyết định bố cục, vị trí và hình dạng đối tượng, các bước sau mới thêm chi tiết. Trong dãy ảnh, đối tượng bắt đầu hiện ra từ khoảng bước 300.')
D_ = s

# ------------------------------------------------------------------ E. sampling
s = new_slide('Sinh ảnh mới: bắt đầu từ một nhiễu ngẫu nhiên')
box(s, 0.6, 1.5, 5.1, 2.75, GRY)
tb(s, 0.8, 1.6, 4.7, 2.9, [
    [('Thuật toán lấy mẫu (DDPM)', 14, BLU, True)],
    [('1.  ', SZ, BLU, True), ('Rút nhiễu ngẫu nhiên ⟦x_T\\sim\\mathcal{N}(0,I)⟧, theo một hạt giống (seed).', SZ, INK)],
    [('2.  ', SZ, BLU, True), ('Với t = T, T − 1, …, 1: mạng đoán nhiễu ⟦\\varepsilon_\\theta(x_t,t)⟧, tính ⟦x_{t-1}⟧ theo công thức slide trước.', SZ, INK)],
    [('3.  ', SZ, BLU, True), ('Trả về ⟦x_0⟧: một ảnh mới.', SZ, INK)]], space=8)
card(s, 0.6, 4.4, 5.1, 1.0, PEA, 'Hạt giống quyết định ảnh', ORG,
     ['Khác hạt giống thì khác ảnh; cùng hạt giống thì ra cùng ảnh.'], hs=13)
card(s, 0.6, 5.55, 5.1, 1.15, GRY, 'Nhược điểm: chậm', INK,
     ['1000 lần chạy mạng cho một ảnh. DDIM (phần sau) rút còn 30 bước.'], hs=13)
rt5 = [1000, 300, 200, 100, 0]; W5 = 1.18; G5 = 0.14; xs0 = 6.95
for k, t in enumerate(rt5): label(s, xs0 + k * (W5 + G5), 1.45, W5, f't = {t}', 10, INK, True)
for r in range(3):
    y = 1.78 + r * (W5 + 0.17)
    tb(s, 5.95, y + 0.3, 0.95, 0.6, [[('hạt giống', 9.5, MUT)], [(str(r + 1), 14, ORG, True)]], align=PP_ALIGN.CENTER, space=0)
    for k, t in enumerate(rt5): pic(s, F(f'rev_{r}_{t}'), xs0 + k * (W5 + G5), y, W5, W5, border=LINE)
tb(s, 5.95, 5.85, 6.78, 1.1, [[('Ba hạt giống, ba ảnh khác nhau. Cùng bộ khử nhiễu lý tưởng trên 12 ảnh mẫu như slide trước nên ảnh cuối trùng một ảnh mẫu; mô hình thật học từ hàng trăm triệu ảnh nên tạo ra ảnh mới chưa từng có.', 10, MUT)]])
notes(s, 'Ghép lại, đây là cách sinh một ảnh mới. Bước 1: rút một nhiễu ngẫu nhiên theo hạt giống. Bước 2: lặp từ bước 1000 về bước 1, mỗi bước mạng đoán nhiễu và trừ bớt. Bước 3: thu được ảnh. Bên phải là ba hạt giống khác nhau, cho ba ảnh khác nhau. Vì vậy trong thực nghiệm, nhóm em cố định hạt giống để mọi phương pháp xuất phát từ cùng một nhiễu, so sánh mới công bằng. Nhược điểm là chậm, 1000 lần chạy mạng cho một ảnh; phần sau sẽ nói DDIM rút xuống còn 30 bước.')
E_ = s

# ------------------------------------------------------------------ F. 2D toy
s = new_slide('Ví dụ trực quan: mô hình học “hình dạng” của dữ liệu')
pic(s, F('toy2d'), 1.35, 1.5, 7.6)
tb(s, 0.6, 2.05, 0.75, 0.9, [[('Thuận', 12, BLU, True)], [('thêm nhiễu', 9.5, MUT)]], space=0)
tb(s, 0.6, 3.85, 0.75, 0.9, [[('Ngược', 12, ORG, True)], [('khử nhiễu', 9.5, MUT)]], space=0)
card(s, 9.25, 1.5, 3.48, 1.55, LBL, 'Hàng trên', BLU, ['1.500 điểm dữ liệu xếp thành hình trái tim. Thêm nhiễu dần, trái tim tan thành một đám mây nhiễu tròn.'], hs=13, bs=11)
card(s, 9.25, 3.17, 3.48, 1.55, PEA, 'Hàng dưới', ORG, ['Bắt đầu từ 1.500 điểm nhiễu mới, khử nhiễu 1000 bước: các điểm tự xếp lại thành trái tim.'], hs=13, bs=11)
box(s, 0.6, 5.5, 12.13, 1.3, YEL)
tb(s, 0.8, 5.6, 11.73, 1.15, [
    [('Với ảnh thật cũng vậy: ', SZ, INK, True), ('mỗi ảnh 512 × 512 màu là một điểm trong không gian 786.432 chiều. Ảnh “trông như thật” chỉ nằm trong một vùng rất nhỏ của không gian đó, giống đường viền trái tim.', SZ, INK)],
    [('Mô hình khuếch tán học “hình dạng” của vùng này và kéo một điểm nhiễu bất kỳ về đó, nên ra một ảnh hợp lý. Mỗi điểm trong hình là một “ảnh”.', SZ, INK)]], space=6)
tb(s, 9.25, 4.85, 3.48, 0.4, [[('Tính thật: DDPM, T = 1000, bộ khử nhiễu lý tưởng.', 9, MUT)]])
notes(s, 'Ví dụ này giúp hình dung trực quan nhất. Mỗi chấm là một dữ liệu, ở đây là điểm hai chiều xếp thành hình trái tim. Hàng trên là quá trình thuận: thêm nhiễu dần, trái tim tan thành một đám mây tròn. Hàng dưới là quá trình ngược: bắt đầu từ các điểm nhiễu hoàn toàn mới, khử nhiễu 1000 bước, các điểm tự xếp lại thành trái tim. Ảnh thật cũng vậy: mỗi ảnh 512 nhân 512 màu là một điểm trong không gian gần 800 nghìn chiều, và ảnh trông như thật chỉ nằm trong một vùng nhỏ, giống đường viền trái tim. Mô hình khuếch tán học hình dạng của vùng đó và kéo một nhiễu bất kỳ về vùng đó, nên sinh ra ảnh hợp lý.')
F_ = s

# ------------------------------------------------------------------ G. bridge to SD
s = new_slide('Từ mô hình khuếch tán đến Stable Diffusion')
tb(s, 0.6, 1.5, 12.13, 0.4, [[('Mô hình khuếch tán cơ bản (vừa trình bày) còn ba hạn chế. Stable Diffusion giải quyết cả ba:', 14, MUT)]])
cards = [('1', 'Chậm và tốn bộ nhớ', 'Khử nhiễu thẳng trên ảnh 512 × 512 × 3 = 786.432 số.',
          'Nén ảnh vào không gian tiềm ẩn 64 × 64 × 4 = 16.384 số (nhỏ hơn 48 lần) bằng bộ mã hóa VAE, khử nhiễu trên ảnh nén.'),
         ('2', 'Không điều khiển được nội dung', 'Sinh ảnh ngẫu nhiên, không biết người dùng muốn vẽ gì.',
          'Đưa câu mô tả vào U-Net qua chú ý chéo: ⟦\\varepsilon_\\theta(z_t,t,\\tau(y))⟧; CFG giúp ảnh bám câu lệnh hơn.'),
         ('3', 'Quá nhiều bước', '1000 lần chạy mạng cho một ảnh.',
          'Lấy mẫu DDIM: chỉ 30 bước nhảy, ảnh vẫn tốt.')]
CW = (12.13 - 2 * 0.25) / 3
for k, (n, h, prob, fix) in enumerate(cards):
    x = 0.6 + k * (CW + 0.25)
    box(s, x, 2.05, CW, 1.7, GRY)
    tb(s, x + 0.2, 2.15, CW - 0.4, 1.7, [[(f'Hạn chế {n}', 11, MUT, True)], [(h, 15, INK, True)], [(prob, 13, INK)]], space=5)
    arrow(s, x + CW / 2, 3.8, x + CW / 2, 4.2, ORG, 2.5)
    box(s, x, 4.25, CW, 1.65, PEA)
    tb(s, x + 0.2, 4.35, CW - 0.4, 1.45, [[('Stable Diffusion', 11, ORG, True)], [(fix, 13, INK)]], space=5)
box(s, 0.6, 6.1, 12.13, 0.75, LBL)
tb(s, 0.8, 6.15, 11.73, 0.65, [[('Stable Diffusion = mô hình khuếch tán + không gian tiềm ẩn + câu mô tả. ', 14, BLU, True), ('Quay lại slide 13 để xem kiến trúc cụ thể.', 14, INK)]], anchor=MSO_ANCHOR.MIDDLE)
notes(s, 'Mô hình khuếch tán cơ bản vừa trình bày còn ba hạn chế, và Stable Diffusion giải quyết cả ba. Một, khử nhiễu thẳng trên ảnh 512 nhân 512 rất tốn kém; Stable Diffusion nén ảnh vào không gian tiềm ẩn nhỏ hơn 48 lần và khử nhiễu trên ảnh nén. Hai, mô hình cơ bản sinh ảnh ngẫu nhiên; Stable Diffusion đưa câu mô tả vào U-Net qua chú ý chéo, nên vẽ theo ý người dùng. Ba, 1000 bước là quá nhiều; dùng DDIM chỉ còn 30 bước. Tóm lại, Stable Diffusion bằng mô hình khuếch tán cộng không gian tiềm ẩn cộng câu mô tả. Giờ quay lại slide 13 để xem kiến trúc cụ thể.')
G_ = s

# ------------------------------------------------------------------ H. slide 13 again, SD half highlighted
s = dup_slide(S13)
ov = box(s, 0.45, 1.42, 12.43, 2.47, WHT)
sf = ov.fill._xPr.find(qn('a:solidFill'))[0]; al = etree.SubElement(sf, qn('a:alpha')); al.set('val', '72000')
fr = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(0.45), I(3.98), I(12.43), I(3.02))
fr.fill.background(); fr.line.color.rgb = ORG; fr.line.width = Pt(2.25); fr.adjustments[0] = 0.04; fr.shadow.inherit = False
tag = box(s, 10.13, 0.42, 2.6, 0.36, ORG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, adj=0.3)
tb(s, 10.13, 0.44, 2.6, 0.32, [[('Quay lại slide 13', 12, WHT, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
notes(s, 'Quay lại slide 13, giờ là phần Stable Diffusion. Stable Diffusion là một mô hình khuếch tán cụ thể, của Rombach và cộng sự thuộc nhóm CompVis, Đại học LMU München, cùng Runway, Stability AI tài trợ; bài báo ở CVPR 2022. Hình bên phải là kiến trúc gốc, ứng đúng ba điểm vừa nói: bộ mã hóa nén ảnh nhỏ hơn 48 lần, U-Net khử nhiễu trên ảnh nén, câu lệnh được chuyển thành vector và đưa vào U-Net qua chú ý chéo, cuối cùng bộ giải mã trả lại ảnh. Slide sau vẽ lại quá trình sinh ảnh này cho dễ theo dõi.')
H_ = s

# slide 13 notes: diffusion only, then hand over to the explainer slides
notes(S13, 'Mô hình khuếch tán là tên một loại mô hình sinh ảnh, do Sohl-Dickstein và cộng sự đề xuất năm 2015, năm 2020 được Ho, Jain và Abbeel cải tiến, lần đầu sinh được ảnh chất lượng cao. Nhiều mô hình nổi tiếng thuộc loại này: DALL·E 2 và DALL·E 3 của OpenAI, Imagen của Google, Stable Diffusion, và cả Sora sinh video. Đây là phần trọng tâm của kiến thức nền, nên nhóm em giải thích kỹ mô hình khuếch tán trong các slide sau, bằng ví dụ dễ hiểu, rồi quay lại slide này để nói về Stable Diffusion ở nửa dưới.')

# ------------------------------------------------------------------ reorder: new slides right after 13
lst = p.slides._sldIdLst; ids = list(lst)
new = ids[-8:]
for e in new: lst.remove(e)
anchor = ids[12]
for e in reversed(new): anchor.addnext(e)

# ------------------------------------------------------------------ renumber "N / 40"
tot = len(p.slides)
for i, s in enumerate(p.slides, 1):
    for t in s._element.iter(qn('a:t')):
        if t.text and re.fullmatch(r'\s*\d+ / \d+\s*', t.text): t.text = f'{i} / {tot}'
p.save(OUT)
print('slides', tot)
