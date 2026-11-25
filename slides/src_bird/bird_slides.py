"""Bobby 2026-11-26: replace the bus/bench example on the M1+M2, M3, M4, M5 and 'Kết quả OCSD' slides with the
natural scene qd_3_simple_003 (a tall tree and two gray birds by a lake, final OCSD, notebook 08). Targeted edits on
Bobby's deck: same layout, only pictures, example texts and notes change. Needs img/ from figs_bird.py.
Usage (run here): bird_slides.py IN OUT"""
import copy, sys
from pptx import Presentation
from pptx.util import Inches as I, Pt
p = Presentation(sys.argv[1])
def title(s): return next((x.text_frame.text.strip() for x in s.shapes if x.name == 'Text 1' and x.has_text_frame), '')
def find(prefix): return next(s for s in p.slides if title(s).startswith(prefix))
def sh(s, name): return next(x for x in s.shapes if x.name == name)
def pic(s, name, f):
    old = sh(s, name); new = s.shapes.add_picture('img/' + f, old.left, old.top, old.width, old.height)
    ln = old._element.spPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ln')
    if ln is not None: new._element.spPr.append(copy.deepcopy(ln))
    old._element.addprevious(new._element); old._element.getparent().remove(old._element)
def txt(s, name, *paras):
    """each para: a string (first run gets it, others emptied) or a list of run texts (bullet runs kept by position)"""
    tf = sh(s, name).text_frame
    for para, t in zip(tf.paragraphs, paras):
        runs = para.runs
        if isinstance(t, str): t = [t]
        if len(runs) > len(t) and runs[0].text.strip() == '•': t = ['•  '] + list(t)
        for k, r in enumerate(runs): r.text = t[k] if k < len(t) else ''
def note(s, t): s.notes_slide.notes_text_frame.text = t
def drop(s, name): e = sh(s, name)._element; e.getparent().remove(e)

# ---------------------------------------------------------------- M1 + M2
s = find('M1-Phân rã vật thể và M2')
pic(s, 'Picture 26', 'sketch.png'); pic(s, 'Picture 29', 'parsed.png')
txt(s, 'TextBox 27', '“a tall tree and two gray birds by a lake”')
txt(s, 'TextBox 31', ['Làm sạch nét, gom nét theo từng vật → 3 vật, mỗi vật một mặt nạ và một hộp.'],
    ['Suy ra tương quan từng cặp, ví dụ: cây bên phải chim 1, chim 1 ở dưới chim 2.'],
    ['Tách câu: cụm từ cho từng vật (“tall tree”, “gray bird”) và cho nền (“by a lake”).'])
# column = (label box, crop pic, selected pic, other pic, sel score boxes, other score boxes, values)
cols = [('TextBox 33', 'Picture 34', 'Picture 36', 'Picture 39', ('TextBox 37', 'TextBox 38'), ('TextBox 40', 'TextBox 41'), 'cây', 1,
         ('0,20+0,57+0,86', '1,63'), ('0,22+0,59+0,80', '1,61')),
        ('TextBox 42', 'Picture 43', 'Picture 48', 'Picture 45', ('TextBox 49', 'TextBox 50'), ('TextBox 46', 'TextBox 47'), 'chim 1 (dưới)', 2,
         ('0,28+0,48+0,92', '1,68'), ('0,27+0,34+0,91', '1,52')),
        ('TextBox 51', 'Picture 52', 'Picture 57', 'Picture 54', ('TextBox 58', 'TextBox 59'), ('TextBox 55', 'TextBox 56'), 'chim 2 (trên)', 3,
         ('0,23+0,43+0,86', '1,52'), ('0,26+0,32+0,78', '1,36'))]
for lab, crop, psel, poth, bsel, both, name, o, vsel, voth in cols:
    txt(s, lab, name); pic(s, crop, f'crop{o}.png'); pic(s, psel, f'cand{o}_1.png'); pic(s, poth, f'cand{o}_2.png')
    txt(s, bsel[0], vsel[0]); txt(s, bsel[1], f'= {vsel[1]} ✓'); txt(s, both[0], voth[0]); txt(s, both[1], f'= {voth[1]}')
note(s, 'M1 làm sạch phác thảo, gom nét theo từng vật thể để có mặt nạ và hộp giới hạn, suy ra tương quan vị trí giữa các cặp vật thể, '
     'và tách câu mô tả thành cụm từ cho từng vật thể và cho nền. M2 vẽ riêng từng vật thể bằng ControlNet, giữ nguyên cụm từ người dùng viết để đúng thuộc tính như màu sắc, '
     'sinh hai ứng viên rồi chọn ảnh tốt nhất. Ảnh trong slide là ví dụ thật từ lần chạy OCSD cuối cùng: phác thảo một cây cao và hai con chim xám bên hồ. '
     'M1 tách thành ba vật, mỗi vật có mặt nạ và hộp; M2 lấy nét của từng vật, vẽ hai ứng viên, rồi chấm điểm mỗi ứng viên bằng tổng ba số từ 0 đến 1: '
     'CLIP đo ảnh giống câu mô tả; nét đo vật trùng nét vẽ, tính bằng IoU; DINO là độ chắc chắn khi Grounding DINO nhận ra vật. '
     'Ví dụ chim 1: ứng viên được chọn 0,28 cộng 0,48 cộng 0,92 bằng 1,68, ứng viên còn lại 0,27 cộng 0,34 cộng 0,91 bằng 1,52; '
     'hai ảnh giống câu và máy nhận ra ngang nhau, nhưng ứng viên được chọn trùng nét vẽ hơn hẳn, nên được giữ, đóng khung cam. Ứng viên còn lại chỉ là điểm thấp hơn, không phải ảnh lỗi.')

# ---------------------------------------------------------------- M3
s = find('M3-Học định danh')
fp = sh(s, 'Picture 57'); FL, FT, FW = fp.left, fp.top, fp.width
pic(s, 'Picture 34', 'm3_ref.png'); pic(s, 'Picture 46', 'm3_class.png'); pic(s, 'Picture 48', 'm3_token.png'); pic(s, 'Picture 57', 'final.png')
txt(s, 'TextBox 35', 'chim xám do M2 vẽ ra')
txt(s, 'TextBox 39', 'Dạy mô hình một từ mới <o1>')
txt(s, 'TextBox 41', ['Cho mô hình xem con chim này khoảng 200 lần: mỗi lần phủ nhiễu lên rồi bắt khử lại.'],
    ['Chỉ chấm lỗi bên trong vùng của chim → học đúng con chim, không học nền.'],
    ['Bắt từ <o1> chỉ “nhìn” vào vùng của chim này → không lẫn với con chim còn lại.'])
txt(s, 'TextBox 47', '“a photo of a bird”'); txt(s, 'TextBox 49', '“a photo of <o1> bird”')
txt(s, 'TextBox 50', '→ chim sẻ đen trắng thành chim xám có mào, mỏ vàng, giống chim của M2')
txt(s, 'TextBox 55', '<o1>')
txt(s, 'TextBox 56', 'Con chim này trong ảnh cuối của M4-Tổng hợp cảnh (khoanh trắng):', 'câu lệnh cảnh ở M4 thay tên loại bằng các từ mới:',
    ['“a photo of <o0> tall tree, ', '<o1> gray bird', ' and <o2> gray bird by a lake”'],
    '→ <o1> gọi lại đúng con chim xám này, ở phía dưới bên trái. Cây (<o0>) và chim còn lại (<o2>) được học theo cùng cách.')
k = FW / 512; b = (59.4, 249.3, 266.3, 385.7); pad = 10
ring = sh(s, 'Oval 53'); ring.left, ring.top = int(FL + (b[0] - pad) * k), int(FT + (b[1] - pad) * k)
ring.width, ring.height = int((b[2] - b[0] + 2 * pad) * k), int((b[3] - b[1] + 2 * pad) * k)
for nm in ('Rounded Rectangle 54', 'TextBox 55'):
    t = sh(s, nm); t.left, t.top = ring.left + I(0.06), ring.top + ring.height + I(0.02)
note(s, 'M3 dạy mô hình nhớ từng vật. Lấy ví dụ con chim xám mà M2 đã vẽ. Nhóm tạo một từ mới, <o1>, rồi cho mô hình xem con chim này khoảng 200 lần: '
     'mỗi lần phủ nhiễu và bắt mô hình khử lại. Lỗi chỉ được tính bên trong vùng của chim, nên mô hình học đúng con chim chứ không học nền; '
     'đồng thời từ <o1> bị buộc chỉ chú ý vào vùng của chim này, để không lẫn với con chim còn lại, vì hai con cùng loại. '
     'Kiểm tra: cùng hạt giống, câu lệnh chỉ ghi “a bird” thì ra một con chim sẻ đen trắng; thay bằng <o1> thì ra chim xám có mào, mỏ vàng, giống chim của M2. '
     'Trong M4, câu lệnh của cảnh dùng <o1> cho con chim này, nên trong ảnh cuối nó là con chim lớn phía dưới bên trái, chỗ khoanh trắng. Cây và con chim còn lại được học theo cùng cách.')

# ---------------------------------------------------------------- M4
s = find('M4-Tổng hợp cảnh')
pic(s, 'Picture 28', 'placed.png')
for nm, st in (('Picture 33', 1), ('Picture 35', 12), ('Picture 37', 18), ('Picture 39', 24)): pic(s, nm, f'st{st:02d}.png')
pic(s, 'Picture 44', 'm4_out.png')
txt(s, 'TextBox 26', 'Ví dụ thật (OCSD cuối cùng): cảnh 2 chim + 1 cây bên hồ, 30 bước khử nhiễu.')
txt(s, 'TextBox 46', '2 chim + 1 cây, đúng chỗ đã vẽ', '(M5 kiểm tra lại, slide sau)')
tf = sh(s, 'TextBox 87').text_frame
for r in tf.paragraphs[3].runs: r.text = r.text.replace('“… near a train station”', '“… by a lake”')
sh(s, 'Rounded Rectangle 86').height = I(1.12); sh(s, 'TextBox 87').height = I(1.0); sh(s, 'TextBox 90').top = I(6.22)   # box was sized for the removed old-M4 card
t = s.notes_slide.notes_text_frame.text
note(s, t.replace('Ở ảnh này ghế còn mờ; M5 phần d sẽ sửa, ở slide sau.', 'Ở ảnh này đã có đủ hai chim và một cây, đúng chỗ đã vẽ; M5 phần d kiểm tra lại, ở slide sau.'))

# ---------------------------------------------------------------- M5
s = find('M5-Điều kiện hóa nhận biết')
txt(s, 'TextBox 41', 'Trong lúc dựng cảnh, liên tục nhắc mô hình: vật nào, ở chỗ nào, mấy cái. Ví dụ thật: cảnh 2 chim + 1 cây.')
for nm, i in (('Picture 49', 0), ('Picture 51', 1), ('Picture 53', 2)): pic(s, nm, f'attn_{i}.png')
txt(s, 'TextBox 50', '“tall tree”'); txt(s, 'TextBox 52', '“gray bird” (dưới)'); txt(s, 'TextBox 54', '“gray bird” (trên)')
tf = sh(s, 'TextBox 55').text_frame
for r in tf.paragraphs[0].runs: r.text = r.text.replace('ở bước 12', 'ở bước 11')
pic(s, 'Picture 60', 'sketch_on_final.png')
pic(s, 'Picture 98', 'check1.png'); pic(s, 'Picture 101', 'final.png')
drop(s, 'Oval 100'); drop(s, 'Oval 103')
txt(s, 'TextBox 99', 'đếm: 2 chim + 1 cây'); txt(s, 'TextBox 102', 'đạt → giữ ảnh')
txt(s, 'TextBox 104', 'Đếm lại vật trong ảnh; thiếu hay lệch thì vẽ lại riêng vùng lỗi (tối đa 2 lần), giữ ảnh điểm cao nhất.',
    'Ở đây đủ 2 chim + 1 cây, đúng chỗ → đạt ngay lần đầu, không cần vẽ lại.', '→ đúng số lượng')
note(s, 'M5 hoạt động trong lúc dựng cảnh, và luôn nhắc mô hình: vật nào, ở chỗ nào, mấy cái. Có bốn phần, minh họa bằng chính cảnh một cây và hai chim của OCSD cuối cùng. '
     'Phần a và b nhìn vào bản đồ chú ý thật ở bước 11: màu đỏ là chỗ một cụm từ đang vẽ, khung trắng là chỗ của vật trong phác thảo. '
     'Phần a chặn từ vẽ ra ngoài khung; phần b, nếu đỏ yếu trong khung hoặc lan ra ngoài, thì chỉnh ảnh ngay ở bước đó. '
     'Ở ảnh thật, đỏ của “tall tree” nằm gọn trong khung cây, và mỗi “gray bird” nằm trong khung của con chim tương ứng, nên hai con chim không bị gộp làm một. '
     'Phần c đưa phác thảo vào lần nữa qua ControlNet, nhẹ tay và ở cả 30 bước: lực 0,6, tức 60 phần trăm so với lực đầy đủ 1,0 mà M2 dùng khi vẽ từng vật. Đủ giữ dáng vật mà không ép cả nền theo nét. '
     'Phần d đếm lại vật trong ảnh bằng Grounding DINO: ở ví dụ này máy thấy đủ hai chim và một cây, đúng chỗ, nên ảnh đạt ngay lần đầu và được giữ nguyên. '
     'Nếu thiếu hoặc lệch, M5 sẽ vẽ lại riêng vùng lỗi, tối đa hai lần.')

# ---------------------------------------------------------------- result + background swap
s = find('Kết quả OCSD: từ phác thảo đến ảnh')
pic(s, 'Picture 33', 'sketch.png'); pic(s, 'Picture 37', 'final.png')
pic(s, 'Picture 43', '1_in_a_city_park.png'); pic(s, 'Picture 45', '2_on_a_snowy_field.png'); pic(s, 'Picture 47', '3_on_a_sandy_beach.png')
txt(s, 'TextBox 34', 'Đầu vào: phác thảo + “a tall tree and two gray birds by a lake”')
txt(s, 'TextBox 40', 'Đúng bài toán đặt ra từ đầu',
    ['✓ Đủ vật: ', 'cả 3 vật đều có, vật hòa vào nền, không viền dán.'],
    ['✓ Đúng số lượng: ', '2 chim + 1 cây, không thừa, không thiếu.'],
    ['✓ Đúng chỗ, đúng dáng: ', 'mỗi vật nằm ở chỗ đã vẽ, theo hình nét vẽ.'],
    ['✓ Đúng tương quan: ', 'cây bên phải; chim lớn ở dưới, chim nhỏ ở trên bên trái.'],
    ['Chưa tốt: ', 'chim trên sát mép trái ảnh; cây thấp hơn nét vẽ một chút.'])
txt(s, 'TextBox 44', '“… in a city park”', '(trong công viên)')
txt(s, 'TextBox 46', '“… on a snowy field”', '(trên cánh đồng tuyết)')
txt(s, 'TextBox 48', '“… on a sandy beach”', '(trên bãi cát)')
txt(s, 'TextBox 49', ['Nền đổi theo câu mô tả, ', 'cây và 2 chim giữ chỗ và dáng như phác thảo. Nền tuyết thừa 1 chim nhỏ bên phải.'],
    'Ảnh thật, cùng hạt giống; M2-M3 chạy một lần, M4-M5 chạy lại cho mỗi nền.')
note(s, 'Đây là toàn bộ OCSD trên một ví dụ thật, đúng bài toán đặt ra từ đầu. Đầu vào là phác thảo một cây và hai con chim, cùng câu mô tả “a tall tree and two gray birds by a lake”. '
     'Qua năm mô-đun M1 đến M5, OCSD cho ra ảnh bên cạnh: đủ ba vật, đúng hai chim và một cây, mỗi vật nằm đúng chỗ và đúng dáng đã vẽ; cây bên phải, chim lớn ở dưới, chim nhỏ ở trên bên trái. '
     'Vật hòa vào nền, không có viền dán. Điểm chưa tốt: con chim phía trên sát mép trái ảnh, và cây thấp hơn nét vẽ một chút. '
     'Hàng dưới: giữ nguyên phác thảo và các vật đã tạo ở M2, M3, chỉ đổi phần nền trong câu mô tả thành trong công viên, trên cánh đồng tuyết và trên bãi cát. '
     'Nền đổi theo câu, cây và hai chim vẫn giữ đúng chỗ và hình dạng. Nhóm nói rõ: ở nền tuyết có thêm một con chim nhỏ bên phải, tức là khi đổi nền, số lượng chưa phải lúc nào cũng đúng. '
     'Đây là ảnh thật do code OCSD tạo ra, cùng hạt giống.')
p.save(sys.argv[2]); print('ok')
