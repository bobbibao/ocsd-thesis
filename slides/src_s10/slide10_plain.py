"""v42 (Bobby 2026-11-26, page 17): slide 10 contributions in plain overview words, no module names (M..);
2nd bullet of 'Mã nguồn, ứng dụng' removed. Notes follow.
Usage: slide10_plain.py IN OUT"""
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); s = p.slides[9]
assert s.shapes[1].text_frame.text.startswith('Đóng góp') or 'Đóng góp' in ''.join(x.text_frame.text for x in s.shapes if x.has_text_frame)
NEW = {8:  [('Xử lý riêng từng vật thể: ', 'từ phác thảo đến ảnh.'),
            ('Ghi nhớ từng vật: ', 'vật cùng loại không bị lẫn.'),
            ('Hợp thành cảnh hài hòa: ', 'đủ vật, ảnh tự nhiên.')],
       2:  [('Đo theo R1–R4: ', 'vật, số lượng, vị trí, tương quan.'),
            ('Đếm vật độc lập: ', 'dùng OWLv2, khác phương pháp.'),
            ('So sánh công bằng: ', 'cùng SD 1.5, có kiểm định.')],
       17: [('Mã nguồn trên GitHub, ', 'chạy bằng Colab.'),
            ('Ứng dụng web minh họa ', 'và kiểm thử tự động.')]}
for x in s.shapes:
    if x.shape_id not in NEW: continue
    paras = x.text_frame.paragraphs; rows = NEW[x.shape_id]
    for pa, (b, t) in zip(paras, rows):
        r = pa.runs; r[1].text = b; r[2].text = t
        for extra in r[3:]: extra._r.getparent().remove(extra._r)
    for pa in paras[len(rows):]: pa._p.getparent().remove(pa._p)
nt = s.notes_slide.notes_text_frame; t = nt.text
for a, b in [('độ phức tạp của nét', 'độ trừu tượng của nét'),
             (', ảnh trung gian của từng mô-đun', '')]:
    assert a in t, a; t = t.replace(a, b)
nt.text = t
assert not any(f'M{i}' in x.text_frame.text for x in s.shapes if x.has_text_frame for i in range(1, 6))
p.save(sys.argv[2]); print('ok')
