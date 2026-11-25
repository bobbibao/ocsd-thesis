"""Bobby 2026-11-26: qualitative slide - add a Collage column and show cases where OCSD does well (failures stay on
the limitations slide). Images from the m4_study_v1 slide-image ZIP (seed 0), in img33/. Usage: slide33_qual.py IN OUT"""
import copy, sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
p = Presentation(sys.argv[1])
s = next(x for x in p.slides if any(q.name == 'Text 1' and q.text_frame.text.startswith('Cùng một phác thảo') for q in x.shapes))
E = lambda v: round(v / 914400, 2)
OLDX = {2.75: 0, 5.35: 1, 7.95: 2, 10.55: 4}
X0, DX, Z, YS = 2.45, 2.05, 1.55, (1.85, 3.47, 5.09)
ROWS = ['qd_1_simple_005', 'qd_3_simple_004', 'qd_5_medium_003']
ARMS = ['sketch', 'controlnet', 'gligen', 'collage', 'ocsd_final']
# collage column = copy of the GLIGEN column (header, pictures, frames)
for sh in list(s.shapes):
    if E(sh.left) == 7.95:
        e = copy.deepcopy(sh._element); sh._element.addnext(e)
        new = s.shapes[-1] if False else None
for sh in s.shapes:   # mark copies: the second shape found at x=7.95 with the same name goes to column 3
    pass
seen = {}
for sh in list(s.shapes):
    x = E(sh.left)
    if x not in OLDX: continue
    col = OLDX[x]
    if col == 2:
        seen[sh.name] = seen.get(sh.name, 0) + 1
        if seen[sh.name] == 2: col = 3
    row = None if sh.top < I(1.8) else min(range(3), key=lambda r: abs(E(sh.top) - (1.85, 3.45, 5.05)[r]))
    sh.left = I(X0 + col * DX)
    if row is None:
        sh.width = I(Z); sh.text_frame.paragraphs[0].runs[0].text = ['Phác thảo', 'Mô hình nền', 'GLIGEN', 'Collage', 'OCSD'][col]
        for r in sh.text_frame.paragraphs[0].runs[1:]: r.text = ''
        continue
    sh.top, sh.width, sh.height = I(YS[row]), I(Z), I(Z)
    if sh.shape_type == 13:
        sid = ROWS[row]; f = f'img33/sketch_{sid}.png' if col == 0 else f'img33/{ARMS[col]}_{sid}.jpg'
        n = s.shapes.add_picture(f, sh.left, sh.top, sh.width, sh.height)
        sh._element.addprevious(n._element); sh._element.getparent().remove(sh._element)
caps = {'Text 9': '“a gray cat in a living room”', 'Text 15': '“a black rabbit, an old car and an orange car on a snowy field”',
        'Text 21': '“two white cows, a brown cow, a big house and a white dog in a green meadow”'}
for sh in s.shapes:
    if sh.name in caps:
        k = list(caps).index(sh.name); sh.left, sh.top, sh.width, sh.height = I(0.6), I(YS[k]), I(1.7), I(Z)
        r = sh.text_frame.paragraphs[0].runs; r[0].text = caps[sh.name]; r[0].font.size = Pt(12)
        for q in r[1:]: q.text = ''
    if sh.name == 'Text 27':
        sh.top = I(6.72); r = sh.text_frame.paragraphs[0].runs
        r[0].text = 'Ảnh thật (hạt giống 0) của lần chạy cuối, cùng cấu hình với bảng kết quả. Chọn các ca OCSD làm tốt; ca OCSD còn sai ở phần hạn chế.'
        for q in r[1:]: q.text = ''
s.notes_slide.notes_text_frame.text = (
    'Đây là ảnh thật của lần chạy cuối; mỗi hàng là một phác thảo, mỗi cột là một phương pháp. Nhóm chọn các ca OCSD làm tốt; ca OCSD còn sai được trình bày ở phần hạn chế. '
    'Hàng đầu, một con mèo xám trong phòng khách: mọi phương pháp đều có mèo, nhưng ở Collage con mèo như bị cắt dán lên sàn; OCSD vẽ con mèo nằm trên ghế, liền khối với cảnh. '
    'Hàng hai, một thỏ đen, một ô tô cũ và một ô tô cam trên cánh đồng tuyết: OCSD có đủ ba vật, đúng chỗ, đúng tương quan; '
    'GLIGEN vẽ thỏ thành một con vật khác; Collage chỉ được máy chấm nhận ra một trong ba vật, các vật trông như dán lên nền. '
    'Hàng ba, năm vật trên đồng cỏ: OCSD giữ đủ năm vật và đúng mọi cặp tương quan; GLIGEN vẽ ngôi nhà thành một khối hộp trắng; Collage gần đủ nhưng thiếu một vật.')
p.save(sys.argv[2]); print('ok')
