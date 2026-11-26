"""v43 (Bobby 2026-11-26, page 17 again): 'Phương pháp' box must say plainly WHAT OCSD is and how it works,
as an overview (no module names). Usage: slide10_overview.py IN OUT"""
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); s = p.slides[9]
x = [x for x in s.shapes if x.shape_id == 8][0]
assert x.text_frame.text.startswith('•  Xử lý riêng')
ROWS = [('OCSD là gì: ', 'phương pháp tạo ảnh từ bản phác thảo và câu mô tả, giữ đúng từng vật thể.'),
        ('Cách làm: ', 'tách phác thảo thành từng vật, tạo riêng từng vật, rồi đặt tất cả vào một cảnh.'),
        ('Kết quả: ', 'ảnh đủ vật, đúng chỗ mà vẫn tự nhiên.')]
for pa, (b, t) in zip(x.text_frame.paragraphs, ROWS):
    r = pa.runs; r[1].text = b; r[2].text = t
p.save(sys.argv[2]); print('ok')
