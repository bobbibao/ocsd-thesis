"""v37: rewrite slide 9 questions + notes with rq_text.QS, then rebuild slide 38 and fix its notes.
Usage: rq_clear.py IN OUT"""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(__file__))
from pptx import Presentation
import rq_text
p = Presentation(sys.argv[1]); s = p.slides[8]
for name, q in zip(['Text 18', 'Text 21', 'Text 24'], rq_text.QS):
    sh = [x for x in s.shapes if x.name == name][0]
    pa = sh.text_frame.paragraphs[0]; pa.runs[0].text = q
    for r in pa.runs[1:]: r._r.getparent().remove(r._r)
nt = s.notes_slide.notes_text_frame
t = nt.text; i = t.index('Từ đó có ba mục tiêu')
nt.text = t[:i] + ('Từ đó nhóm đặt ba câu hỏi. Q1: phác thảo nhiều vật thể hơn hay nét vẽ trừu tượng hơn làm mô hình tạo ảnh mất vật nhiều hơn; '
    'thí nghiệm E1 và E2. Q2: tạo riêng từng vật và học định danh cho mỗi vật, tức M1 đến M3, có giữ đủ vật và đúng chỗ hơn cách chỉ dùng nét vẽ không; '
    'thí nghiệm E3. Q3: hợp thành cảnh hài hòa ở M4 có giữ nhiều vật hơn và cho ảnh tự nhiên hơn cách trộn cứng không; thí nghiệm E4.')
s38 = p.slides[37].notes_slide.notes_text_frame
t = s38.text
for a, b in [('Q1, mục tiêu xác định nguyên nhân:', 'Q1:'), ('Q2, mục tiêu đề xuất và so sánh:', 'Q2:'),
             ('Q3, mục tiêu hợp thành cảnh hài hòa:', 'Q3:'), ('độ phức tạp phác thảo', 'độ trừu tượng của nét vẽ'),
             ('cả hai giả thuyết đăng ký trước đều đúng', 'cả hai giả thuyết nhóm đặt trước khi chạy đều đúng')]:
    assert a in t, a; t = t.replace(a, b)
s38.text = t
tmp = sys.argv[2] + '.tmp.pptx'; p.save(tmp)
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), 'slide38_rq_visual.py'), tmp, sys.argv[2]])
os.remove(tmp)
