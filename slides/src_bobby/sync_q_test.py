"""Bobby 2026-11-26 'sửa luôn': slide 36 uses the short Q1-Q3 of slide 9; slide 29 footer names the test actually
used (sign test, Holm; tests.csv p_sign). Usage: sync_q_test.py IN OUT"""
import sys
from pptx import Presentation
p = Presentation(sys.argv[1])
Q = {'TextBox 7': 'Số lượng vật thể hay độ phức tạp của phác thảo mới là nguyên nhân chính khiến mô hình nền vi phạm R1–R4? Lỗi chủ yếu là gì?',
     'TextBox 20': 'Tách cảnh thành từng vật thể có định danh riêng có giúp đạt R1, R2, R4 tốt hơn các phương pháp chỉ dùng nét vẽ không, và phải đánh đổi gì?',
     'TextBox 33': 'Hợp thành cảnh hài hòa có giữ được nhiều vật thể hơn, đồng thời tự nhiên hơn cách trộn cứng không?'}
s = p.slides[35]
for sh in s.shapes:
    if sh.name in Q:
        r = sh.text_frame.paragraphs[0].runs; r[1].text = Q[sh.name]
        for q in r[2:]: q.text = ''
for sh in s.shapes:
    if sh.has_text_frame:
        for para in sh.text_frame.paragraphs:
            for r in para.runs: r.text = r.text.replace('đúng quan hệ', 'đúng tương quan')
s = p.slides[28]
for sh in s.shapes:
    if sh.name == 'TextBox 40':
        for r in sh.text_frame.paragraphs[0].runs: r.text = r.text.replace('kiểm định Wilcoxon cặp', 'kiểm định dấu cặp')
p.save(sys.argv[2]); print('ok')
