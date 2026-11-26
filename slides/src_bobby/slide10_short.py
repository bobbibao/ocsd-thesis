"""Bobby 2026-11-26: slide 10 'Đóng góp đề tài' shorter and more concise. Rewrites only the bullet runs of the
4 cards (same bold-lead + plain-tail pattern) and enlarges them. Usage: slide10_short.py IN OUT"""
import sys
from pptx import Presentation
from pptx.util import Pt
p = Presentation(sys.argv[1]); s = p.slides[9]
NEW = [
    [('Khung OCSD 5 mô-đun: ', 'xử lý từng vật thể.'),
     ('Vẽ, học định danh riêng ', '(M2, M3): không lẫn vật.'),
     ('Hợp thành cảnh ', '(M4, M5): đủ vật, tự nhiên.')],
    [('QuickDraw-Scenes: ', 'phác thảo tay, 1 đến 8+ vật.'),
     ('COCO-Sketch: ', 'phác thảo dựng từ ảnh thực.'),
     ('Đủ chú thích: ', 'câu mô tả, mặt nạ, hộp từng vật.')],
    [('Độ đo R1–R4: ', 'OPR, sai số số lượng, mIoU, RA.'),
     ('Chấm bằng OWLv2, ', 'độc lập với phương pháp.'),
     ('Kiểm định thống kê ', 'trên cùng SD 1.5.')],
    [('Mã nguồn trên GitHub, ', 'chạy bằng Colab.'),
     ('Ảnh trung gian M1–M5 ', 'để kiểm tra.'),
     ('Ứng dụng web minh họa ', 'và kiểm thử tự động.')]]
first = lambda x: x.text_frame.paragraphs[0].runs[1].text if x.has_text_frame and len(x.text_frame.paragraphs[0].runs) > 1 else ''
leads = {'Khung OCSD': 0, 'QuickDraw': 1, 'Bộ độ đo': 2, 'Toàn bộ mã': 3}
done = set()
for x in s.shapes:
    k = next((v for key, v in leads.items() if first(x).startswith(key)), None)
    if k is None: continue
    for para, (b, t) in zip(x.text_frame.paragraphs, NEW[k]):
        r = para.runs
        r[1].text, r[2].text = b, t
        for run in r: run.font.size = Pt(16)
        para.space_after = Pt(14)
    done.add(k)
assert done == {0, 1, 2, 3}, done
p.save(sys.argv[2]); print('ok')
