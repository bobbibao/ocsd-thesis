"""Bobby 2026-11-26: one example scene through the deck: the farm sketch qd_5_medium_005 ('three small cows, a tall
tree and a red truck on a farm', final OCSD, seed 0, notebook 10) on the background slides (diffusion figure, add
noise, remove noise, SD prompt), M1-M5, 'Kết quả OCSD' and the metric slide. Targeted edits: same layout, pictures,
example texts and notes change; replaced pictures keep their shape names (slides13_16_anim.py targets them by name).
Needs img/ from figs_farm.py. Run on the v21 layout BEFORE slides13_16_anim.py. Usage (run here): farm_slides.py IN OUT"""
import copy, json, sys
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches as I
p = Presentation(sys.argv[1])
def title(s): return next((x.text_frame.text.strip() for x in s.shapes if x.name == 'Text 1' and x.has_text_frame), '')
def find(prefix): return next(s for s in p.slides if title(s).startswith(prefix))
def sh(s, name): return next(x for x in s.shapes if x.name == name)
def pic(s, name, f):
    old = sh(s, name); new = s.shapes.add_picture('img/' + f, old.left, old.top, old.width, old.height)
    ln = old._element.spPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ln')
    if ln is not None: new._element.spPr.append(copy.deepcopy(ln))
    new.name = name
    old._element.addprevious(new._element); old._element.getparent().remove(old._element)
def txt(s, name, *paras):
    tf = sh(s, name).text_frame
    for para, t in zip(tf.paragraphs, paras):
        runs = para.runs
        if isinstance(t, str): t = [t]
        if len(runs) > len(t) and runs[0].text.strip() == '•': t = ['•  '] + list(t)
        for k, r in enumerate(runs): r.text = t[k] if k < len(t) else ''
def note(s, t): s.notes_slide.notes_text_frame.text = t
def rep_note(s, a, b):
    t = s.notes_slide.notes_text_frame.text; assert a in t, a; note(s, t.replace(a, b))
CAP = '“three small cows, a tall tree and a red truck on a farm”'
M = json.load(open('img/trace.json'))

# ================================================================ background
s = find('Diffusion Model và Stable Diffusion'); pic(s, 'Picture 21', 'fig13.png')
s = find('Mô hình khuếch tán: thêm nhiễu')
for nm, f in (('Picture 34', 'fw_t0.png'), ('Picture 37', 'fw_eps0.png'), ('Picture 40', 'fw_t150.png'), ('Picture 43', 'fw_eps1.png'),
              ('Picture 46', 'fw_t400.png'), ('Picture 49', 'fw_eps2.png'), ('Picture 52', 'fw_t1000.png'),
              ('Picture 55', 'tr_noisy.png'), ('Picture 58', 'tr_eps.png'), ('Picture 61', 'fw_t0.png'),
              ('Picture 66', 'tr_noisy.png'), ('Picture 72', 'tr_eps.png')):
    pic(s, nm, f)
rep_note(s, 'lấy ảnh con bò', 'lấy ảnh cảnh nông trại, chính ảnh OCSD tạo ra ở phần phương pháp')
s = find('Tạo ảnh: bắt đầu từ nhiễu')
for nm, f in (('Picture 34', 'rv_x0.png'), ('Picture 37', 'rv_eps0.png'), ('Picture 40', 'rv_x1.png'), ('Picture 43', 'rv_eps1.png'),
              ('Picture 46', 'rv_x2.png'), ('Picture 49', 'rv_eps2.png'), ('Picture 52', 'final.png'),
              ('Picture 63', 'rs_0.png'), ('Picture 65', 'rs_1.png'), ('Picture 67', 'rs_2.png'), ('Picture 69', 'rs_3.png'),
              ('Picture 71', 'rs_4.png'), ('Picture 73', 'final.png')):
    pic(s, nm, f)
txt(s, 'TextBox 35', 'nhiễu ngẫu nhiên', 't = 958'); txt(s, 'TextBox 41', 'thấy dáng bò, cây', 't = 331')
txt(s, 'TextBox 47', 'rõ dần', 't = 133'); txt(s, 'TextBox 53', 'ảnh mới', 't = 0')
for nm, t in zip(('TextBox 64', 'TextBox 66', 'TextBox 68', 'TextBox 70', 'TextBox 72', 'TextBox 74'), (958, 793, 595, 397, 199, 0)):
    txt(s, nm, f't = {t}')
txt(s, 'TextBox 60', 'Ảnh thật: các bước khử nhiễu của OCSD trên cảnh nông trại (30 bước, giải mã về ảnh). '
    'Ảnh “U-Net đoán nhiễu” được tính lại từ ảnh hiện tại và ảnh sạch U-Net dự đoán.')
note(s, 'Khi mạng đã biết tìm nhiễu, tạo ảnh mới là đi ngược lại. Bắt đầu từ một nhiễu hoàn toàn ngẫu nhiên. Ở mỗi bước, U-Net nhìn ảnh đang nhiễu, '
     'đoán phần nhiễu, rồi chỉ xóa bớt một ít theo hướng đó. Đây là ảnh thật của chính lần chạy OCSD trên cảnh nông trại: ở bước đầu chưa thấy gì, '
     'khoảng t bằng 331 đã thấy dáng ba con bò và cái cây, đến cuối là ảnh rõ nét. Hàng dưới là các bước trung gian của cùng lần chạy. '
     'Nhiễu ban đầu khác thì ra ảnh khác, như ba ví dụ bên phải. Stable Diffusion làm đúng như vậy nhưng thêm câu lệnh, để ảnh tạo ra khớp với câu mô tả.')
s = find('Bức tranh tổng thể về Stable Diffusion'); txt(s, 'Text 5', 'Câu lệnh (prompt)', CAP)

# ================================================================ M1 + M2
s = find('M1-Phân rã vật thể và M2')
pic(s, 'Picture 60', 'sketch.png'); pic(s, 'Picture 61', 'parsed.png')
txt(s, 'TextBox 27', CAP)
txt(s, 'TextBox 31', ['Làm sạch nét, gom nét theo từng vật → 5 vật, mỗi vật một mặt nạ và một hộp.'],
    ['Suy ra tương quan từng cặp, ví dụ: xe tải ở dưới bò 1, cây bên phải xe tải.'],
    ['Tách câu: cụm từ cho từng vật (“small cow”, “tall tree”, “red truck”) và cho nền (“on a farm”).'])
# (label, crop, left pic, right pic, left score boxes, right score boxes, name, obj, selected side, sel scores, other scores)
cols = [('TextBox 33', 'Picture 62', 'Picture 63', 'Picture 64', ('TextBox 37', 'TextBox 38'), ('TextBox 40', 'TextBox 41'), 'bò 1 (trên trái)', 1, 'L',
         ('0,24+0,48+0,88', '1,60'), ('0,27+0,40+0,88', '1,55')),
        ('TextBox 42', 'Picture 65', 'Picture 67', 'Picture 66', ('TextBox 46', 'TextBox 47'), ('TextBox 49', 'TextBox 50'), 'cây', 2, 'R',
         ('0,20+0,23+0,74', '1,18'), ('0,21+0,54+0,42', '1,17')),
        ('TextBox 51', 'Picture 68', 'Picture 70', 'Picture 69', ('TextBox 55', 'TextBox 56'), ('TextBox 58', 'TextBox 59'), 'xe tải', 3, 'R',
         ('0,21+0,54+0,88', '1,63'), ('0,22+0,41+0,62', '1,25'))]
for lab, crop, pl, pr, bl, br, name, o, side, vsel, voth in cols:
    txt(s, lab, name); pic(s, crop, f'crop{o}.png')
    (psel, poth, bsel, both) = (pl, pr, bl, br) if side == 'L' else (pr, pl, br, bl)
    pic(s, psel, f'cand{o}_sel.png'); pic(s, poth, f'cand{o}_oth.png')
    txt(s, bsel[0], vsel[0]); txt(s, bsel[1], f'= {vsel[1]} ✓'); txt(s, both[0], voth[0]); txt(s, both[1], f'= {voth[1]}')
note(s, 'M1 làm sạch phác thảo, gom nét theo từng vật thể để có mặt nạ và hộp giới hạn, suy ra tương quan vị trí giữa các cặp vật thể, '
     'và tách câu mô tả thành cụm từ cho từng vật thể và cho nền. M2 vẽ riêng từng vật thể bằng ControlNet, giữ nguyên cụm từ người dùng viết để đúng thuộc tính như màu sắc, '
     'tạo hai ứng viên rồi chọn ảnh tốt nhất. Ảnh trong slide là ví dụ thật từ lần chạy OCSD cuối cùng, cũng là phác thảo ở slide đặt vấn đề: ba con bò nhỏ, một cây cao và một xe tải đỏ trên nông trại. '
     'M1 tách thành năm vật, mỗi vật có mặt nạ và hộp; M2 lấy nét của từng vật, vẽ hai ứng viên, rồi chấm điểm mỗi ứng viên bằng tổng ba số từ 0 đến 1: '
     'CLIP đo ảnh giống câu mô tả; nét đo vật trùng nét vẽ, tính bằng IoU; DINO là độ chắc chắn khi Grounding DINO nhận ra vật. '
     'Ví dụ xe tải: ứng viên được chọn 0,21 cộng 0,54 cộng 0,88 bằng 1,63, ứng viên còn lại 0,22 cộng 0,41 cộng 0,62 bằng 1,25; '
     'ứng viên được chọn trùng nét vẽ hơn và máy nhận ra là xe tải chắc hơn, nên được giữ, đóng khung cam. Ở cây, hai ứng viên gần bằng điểm: '
     'ứng viên còn lại trùng nét hơn nhưng máy ít chắc đó là cây, nên tổng thấp hơn một chút. Slide trình bày ba trong năm vật; hai con bò còn lại làm giống bò 1.')

# ================================================================ M3 (follow the truck, <o2>)
s = find('M3-Học định danh')
fp = sh(s, 'Picture 61'); FL, FT, FW = fp.left, fp.top, fp.width
pic(s, 'Picture 58', 'm3_ref.png'); pic(s, 'Picture 59', 'm3_class.png'); pic(s, 'Picture 60', 'm3_token.png'); pic(s, 'Picture 61', 'final.png')
txt(s, 'TextBox 35', 'xe tải do M2 vẽ ra')
txt(s, 'TextBox 39', 'Dạy mô hình một từ mới <o2>')
txt(s, 'TextBox 41', ['Cho mô hình xem chiếc xe này khoảng 200 lần: mỗi lần phủ nhiễu lên rồi bắt khử lại.'],
    ['Chỉ chấm lỗi bên trong vùng của xe → học đúng chiếc xe, không học nền.'],
    ['Bắt từ <o2> chỉ “nhìn” vào vùng của xe này → không lẫn với bò và cây bên cạnh.'])
txt(s, 'TextBox 47', '“a photo of a truck”'); txt(s, 'TextBox 49', '“a photo of <o2> truck”')
txt(s, 'TextBox 50', '→ xe đầu kéo đỏ thành xe thùng trắng, gần với xe của M2')
txt(s, 'TextBox 55', '<o2>')
txt(s, 'TextBox 56', 'Chiếc xe này trong ảnh cuối của M4-Tổng hợp cảnh (khoanh trắng):', 'câu lệnh cảnh ở M4 thay tên loại bằng các từ mới:',
    ['“a photo of <o0> small cow, <o1> tall tree, ', '<o2> red truck', ', <o3> small cow and <o4> small cow on a farm”'],
    '→ <o2> gọi lại đúng chiếc xe này, ở bên trái. Ba con bò (<o0>, <o3>, <o4>) cùng loại nhưng mỗi con một từ riêng, nên không lẫn nhau.')
truck = next(d for d in M['m5d']['checks'][0]['detections'] if d['cls'] == 'truck')['box']
k = FW / 512; b = [max(0, truck[0]), truck[1], truck[2], truck[3]]; pad = 10
ring = sh(s, 'Oval 53'); ring.left, ring.top = int(FL + max(0, b[0] - pad) * k), int(FT + (b[1] - pad) * k)
ring.width, ring.height = int((b[2] - max(0, b[0] - pad) + pad) * k), int((b[3] - b[1] + 2 * pad) * k)
for nm in ('Rounded Rectangle 54', 'TextBox 55'):
    t = sh(s, nm); t.left, t.top = ring.left + I(0.06), ring.top + ring.height + I(0.02)
note(s, 'M3 dạy mô hình nhớ từng vật. Lấy ví dụ chiếc xe tải mà M2 đã vẽ. Nhóm tạo một từ mới, <o2>, rồi cho mô hình xem chiếc xe này khoảng 200 lần: '
     'mỗi lần phủ nhiễu và bắt mô hình khử lại. Lỗi chỉ được tính bên trong vùng của xe, nên mô hình học đúng chiếc xe chứ không học nền; '
     'đồng thời từ <o2> bị buộc chỉ chú ý vào vùng của xe này, để không lẫn với bò và cây bên cạnh. '
     'Kiểm tra: cùng hạt giống, câu lệnh chỉ ghi “a truck” thì ra một xe đầu kéo màu đỏ nhìn thẳng; thay bằng <o2> thì ra xe tải có thùng trắng dài, gần với xe của M2. '
     'Trong M4, câu lệnh của cảnh dùng <o2> cho chiếc xe này, nên trong ảnh cuối nó là chiếc xe bên trái, chỗ khoanh trắng. '
     'Ba con bò cùng loại nhưng mỗi con một từ riêng, <o0>, <o3> và <o4>, nên không bị gộp hay lẫn nhau; cây là <o1>.')

# ================================================================ M4
s = find('M4-Tổng hợp cảnh')
pic(s, 'Picture 91', 'placed.png')
for nm, st in (('Picture 92', 1), ('Picture 93', 12), ('Picture 94', 18), ('Picture 95', 24)): pic(s, nm, f'st{st:02d}.png')
pic(s, 'Picture 96', 'm4_out.png')
txt(s, 'TextBox 26', 'Ví dụ thật (OCSD cuối cùng): cảnh 3 bò + 1 cây + 1 xe tải trên nông trại, 30 bước khử nhiễu.')
txt(s, 'TextBox 46', '3 bò + 1 cây + 1 xe, đúng chỗ đã vẽ', '(M5 kiểm tra lại, slide sau)')
tf = sh(s, 'TextBox 87').text_frame
for r in tf.paragraphs[3].runs: r.text = r.text.replace('“… by a lake”', '“… on a farm”')
rep_note(s, 'Ở ảnh này đã có đủ hai chim và một cây, đúng chỗ đã vẽ', 'Ở ảnh này đã có đủ ba con bò, một cây và một xe tải, đúng chỗ đã vẽ')

# ================================================================ M5
s = find('M5-Điều kiện hóa nhận biết')
txt(s, 'TextBox 41', 'Trong lúc dựng cảnh, liên tục nhắc mô hình: vật nào, ở chỗ nào, mấy cái. Ví dụ thật: cảnh 3 bò + 1 cây + 1 xe tải.')
for nm, i in (('Picture 106', 0), ('Picture 107', 1), ('Picture 108', 2)): pic(s, nm, f'attn_{i}.png')
txt(s, 'TextBox 50', '“small cow” (bò 1)'); txt(s, 'TextBox 52', '“tall tree”'); txt(s, 'TextBox 54', '“red truck”')
for r in sh(s, 'TextBox 55').text_frame.paragraphs[0].runs: r.text = r.text.replace('ở bước 11', 'ở bước 21')
pic(s, 'Picture 109', 'sketch_on_final.png'); pic(s, 'Picture 110', 'check1.png'); pic(s, 'Picture 111', 'final.png')
txt(s, 'TextBox 99', 'đếm: 3 bò + 1 cây + 1 xe')
txt(s, 'TextBox 104', 'Đếm lại vật trong ảnh; thiếu hay lệch thì vẽ lại riêng vùng lỗi (tối đa 2 lần), giữ ảnh điểm cao nhất.',
    'Ở đây đủ 3 bò + 1 cây + 1 xe, đúng chỗ → đạt ngay lần đầu, không cần vẽ lại.', '→ đúng số lượng')
note(s, 'M5 hoạt động trong lúc dựng cảnh, và luôn nhắc mô hình: vật nào, ở chỗ nào, mấy cái. Có bốn phần, minh họa bằng chính cảnh nông trại của OCSD cuối cùng. '
     'Phần a và b nhìn vào bản đồ chú ý thật ở bước 21: màu đỏ là chỗ một cụm từ đang vẽ, khung trắng là chỗ của vật trong phác thảo. '
     'Phần a chặn từ vẽ ra ngoài khung; phần b, nếu đỏ yếu trong khung hoặc lan ra ngoài, thì chỉnh ảnh ngay ở bước đó. '
     'Ở ảnh thật, đỏ của “small cow” nằm ở khung bò 1, “tall tree” nằm trong khung cây, “red truck” nằm trong khung xe, nên các vật không rò sang nhau. '
     'Phần c đưa phác thảo vào lần nữa qua ControlNet, nhẹ tay và ở cả 30 bước: lực 0,6, tức 60 phần trăm so với lực đầy đủ 1,0 mà M2 dùng khi vẽ từng vật. Đủ giữ dáng vật mà không ép cả nền theo nét. '
     'Phần d đếm lại vật trong ảnh bằng Grounding DINO: ở ví dụ này máy thấy đủ ba con bò, một cây và một xe tải, đúng chỗ, nên ảnh đạt ngay lần đầu và được giữ nguyên. '
     'Nếu thiếu hoặc lệch, M5 sẽ vẽ lại riêng vùng lỗi, tối đa hai lần.')

# ================================================================ result + background swap
s = find('Kết quả OCSD: từ phác thảo đến ảnh')
pic(s, 'Picture 50', 'sketch.png'); pic(s, 'Picture 51', 'final.png')
pic(s, 'Picture 52', '1_in_a_city_park.png'); pic(s, 'Picture 53', '2_on_a_snowy_field.png'); pic(s, 'Picture 54', '3_on_a_sandy_beach.png')
txt(s, 'TextBox 34', 'Đầu vào: phác thảo + ' + CAP)
txt(s, 'TextBox 40', 'Đúng bài toán đặt ra từ đầu',
    ['✓ Đủ vật: ', 'cả 5 vật đều có, vật hòa vào nền, không viền dán.'],
    ['✓ Đúng số lượng: ', '3 bò + 1 cây + 1 xe tải, không thừa, không thiếu.'],
    ['✓ Đúng chỗ, đúng dáng: ', 'mỗi vật nằm ở chỗ đã vẽ, theo hình nét vẽ.'],
    ['✓ Đúng tương quan: ', 'xe dưới bò 1, cây bên phải xe, bò lớn dưới bên phải.'],
    ['Chưa tốt: ', 'xe tải màu xám chứ không đỏ; nền trông như mô hình đồ chơi.'])
txt(s, 'TextBox 44', '“… in a city park”', '(trong công viên)')
txt(s, 'TextBox 46', '“… on a snowy field”', '(trên cánh đồng tuyết)')
txt(s, 'TextBox 48', '“… on a sandy beach”', '(trên bãi cát)')
txt(s, 'TextBox 49', ['Nền đổi theo câu mô tả, ', '3 bò và xe giữ chỗ như phác thảo. Chưa tốt: cây ở công viên như ảnh dán; nền tuyết thừa vài cây thông.'],
    'Ảnh thật, cùng hạt giống; M2-M3 chạy một lần, M4-M5 chạy lại cho mỗi nền.')
note(s, 'Đây là toàn bộ OCSD trên chính phác thảo ở slide đặt vấn đề, nơi ControlNet, T2I-Adapter và GLIGEN đều lỗi. Đầu vào là phác thảo ba con bò nhỏ, một cây cao và một xe tải, '
     'cùng câu mô tả “three small cows, a tall tree and a red truck on a farm”. Qua năm mô-đun M1 đến M5, OCSD cho ra ảnh bên cạnh: đủ năm vật, đúng ba bò, một cây, một xe, '
     'mỗi vật nằm đúng chỗ đã vẽ; xe ở dưới con bò bên trái, cây bên phải xe, con bò lớn ở dưới bên phải. Vật hòa vào nền, không có viền dán. '
     'Điểm chưa tốt, nhóm nói rõ: xe tải ra màu xám chứ không đỏ như câu mô tả, và cả cảnh trông giống mô hình đồ chơi. '
     'Hàng dưới: giữ nguyên phác thảo và các vật đã tạo ở M2, M3, chỉ đổi phần nền trong câu mô tả thành trong công viên, trên cánh đồng tuyết và trên bãi cát. '
     'Nền đổi theo câu, ba con bò và chiếc xe vẫn giữ đúng chỗ. Chưa tốt: ở công viên, cây trông như một tấm ảnh dán vào; ở nền tuyết có thêm vài cây thông nhỏ, '
     'tức là khi đổi nền, số lượng chưa phải lúc nào cũng đúng. Đây là ảnh thật do code OCSD tạo ra, cùng hạt giống. '
     'Đây là một ví dụ minh họa; so sánh trên toàn bộ dữ liệu nằm ở phần thực nghiệm.')

# ================================================================ metric slide
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
base = Image.open('img/final.png').convert('RGB').resize((512, 512))
dets = M['m5d']['checks'][0]['detections']; objs = M['m1']['objects']
GRN, BLU, ORG, WHT = (40, 200, 90), (60, 120, 255), (235, 110, 40), (255, 255, 255)
NM = {'cow': 'bò', 'tree': 'cây', 'truck': 'xe tải'}
def lab(d, x, y, t, col):
    w = d.textlength(t, font=F); x = min(max(0, x), 512 - w - 10); y = min(max(0, y), 512 - 28)
    d.rectangle([x, y, x + w + 10, y + 28], fill=col); d.text((x + 5, y + 2), t, font=F, fill=WHT)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)
pairs = sorted(((iou(o['box'], x['box']), i, j) for i, o in enumerate(objs) for j, x in enumerate(dets) if x['cls'] == o['cls']), reverse=True)
match, used = {}, set()
for v, i, j in pairs:
    if i not in match and j not in used: match[i] = (j, v); used.add(j)
assert len(match) == len(objs)
im = base.copy(); d = ImageDraw.Draw(im)
for det in dets:
    b = det['box']; d.rectangle(b, outline=GRN, width=5); lab(d, b[0], b[1] - 30, NM[det['cls']] + ' ✓', GRN)
im.save('img/m_opr.png')
im = base.copy(); d = ImageDraw.Draw(im)
def dash(b, col):
    for k in range(0, int(b[2] - b[0]), 16): d.line([b[0] + k, b[1], min(b[0] + k + 8, b[2]), b[1]], fill=col, width=4); d.line([b[0] + k, b[3], min(b[0] + k + 8, b[2]), b[3]], fill=col, width=4)
    for k in range(0, int(b[3] - b[1]), 16): d.line([b[0], b[1] + k, b[0], min(b[1] + k + 8, b[3])], fill=col, width=4); d.line([b[2], b[1] + k, b[2], min(b[1] + k + 8, b[3])], fill=col, width=4)
for o in objs: dash(o['box'], BLU)
for det in dets: d.rectangle(det['box'], outline=GRN, width=5)
IOU = [match[i][1] for i in range(len(objs))]
f2 = lambda v: f'{v:.2f}'.replace('.', ',')
for i, o in enumerate(objs): b = o['box']; lab(d, b[0] + 4, b[1] + 4, f2(IOU[i]), ORG)
lab(d, 8, 470, 'xanh dương: hộp đã vẽ', BLU); lab(d, 8, 436, 'xanh lá: hộp tìm thấy', GRN)
im.save('img/m_miou.png')
# relation accuracy over all pairs: left/right and above/below as in M1 relations, checked on the found boxes
cen = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
C = [cen(dets[match[i][0]]['box']) for i in range(len(objs))]
rel = {}
for a, b_, r in M['m1']['relations']:
    if r in ('left_of', 'right_of', 'above', 'below'): rel.setdefault((a, b_), []).append(r)
ok = lambda a, b_, r: {'left_of': C[a][0] < C[b_][0], 'right_of': C[a][0] > C[b_][0], 'above': C[a][1] < C[b_][1], 'below': C[a][1] > C[b_][1]}[r]
good = sum(all(ok(a, b_, r) for r in rs) for (a, b_), rs in rel.items()); npair = len(rel)
im = base.copy(); d = ImageDraw.Draw(im)
for a, b_, t in [(2, 1, 'cây bên phải ✓'), (3, 4, 'ở dưới ✓')]:
    d.line([C[a], C[b_]], fill=ORG, width=6)
    mx, my = (C[a][0] + C[b_][0]) / 2, (C[a][1] + C[b_][1]) / 2; lab(d, mx - 60, my - 14, t, ORG)
for q in C: d.ellipse([q[0] - 9, q[1] - 9, q[0] + 9, q[1] + 9], fill=ORG)
im.save('img/m_ra.png')
s = find('Cách đo')
for nm, f in (('Picture 41', 'm_opr.png'), ('Picture 42', 'm_miou.png'), ('Picture 43', 'm_ra.png')): pic(s, nm, f)
n = len(objs); mi = sum(IOU) / n
txt(s, 'TextBox 24', 'OPR = số vật tìm thấy ÷ số vật đã vẽ', f'= {len(match)} ÷ {n} = 100%')
txt(s, 'TextBox 31', 'IoU mỗi vật = phần chồng nhau ÷ phần gộp của 2 hộp', 'mIoU = (' + ' + '.join(f2(v) for v in IOU) + f') ÷ {n} = {f2(mi)}')
txt(s, 'TextBox 38', 'RA = số cặp giữ đúng ÷ tổng số cặp', f'= {good} ÷ {npair} = {round(100 * good / npair)}%')
note(s, 'Nhóm dùng ba độ đo, cũng là ba độ đo trong các slide kết quả. Mọi ảnh được chấm tự động bằng OWLv2, một bộ phát hiện độc lập, khác bộ dùng bên trong OCSD, '
     'để phương pháp không “học tủ” theo công cụ chấm. Hình minh họa dùng chính cảnh nông trại ở phần phương pháp: ba con bò, một cây và một xe tải. '
     f'Thứ nhất, giữ vật thể, OPR: trong các vật đã vẽ, bao nhiêu vật được tìm thấy đúng loại và gần đúng chỗ. Vẽ {n} vật, tìm thấy đủ {n}, nên 100 phần trăm. '
     'Thứ hai, vị trí và hình dạng, mIoU. Với mỗi vật, lấy phần hai hộp chồng lên nhau chia cho phần gộp của hai hộp, gọi là IoU, từ 0 đến 1. '
     f'Ở cảnh ví dụ, năm vật được ' + ', '.join(f2(v) for v in IOU) + f'; trung bình là {f2(mi)}. Xe tải thấp nhất, {f2(IOU[2])}, vì xe vẽ ra lệch sang trái và nhỏ hơn nét phác thảo. '
     f'Thứ ba, đúng tương quan, RA: các cặp vật có giữ đúng trái phải, trên dưới như phác thảo không; ví dụ cây bên phải xe tải, con bò lớn ở dưới con bò nhỏ bên phải. '
     f'{good} trên {npair} cặp đúng. Khi so sánh các phương pháp, nhóm dùng kiểm định thống kê cặp, hiệu chỉnh Holm.')
p.save(sys.argv[2]); print('ok', [round(v, 2) for v in IOU], round(mi, 3), good, npair)
