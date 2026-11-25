"""Bobby 2026-11-26: slide 9 questions shorter; R4 without the word 'quan hệ' (-> 'Đúng tương quan', same as the RA
metric name on the charts). Usage: slide9_rq.py IN OUT"""
import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); s = p.slides[8]; S = {x.name: x for x in s.shapes}
def put(nm, *texts):
    for para, t in zip(S[nm].text_frame.paragraphs, texts):
        r = para.runs; r[0].text = t
        for q in r[1:]: q.text = ''
put('Text 15', 'Đúng tương quan', 'trái/phải, trên/dưới, trước/sau giữa các vật')
put('Text 18', 'Số lượng vật thể hay độ phức tạp của phác thảo mới là nguyên nhân chính khiến mô hình nền vi phạm R1–R4? Lỗi chủ yếu là gì?')
put('Text 21', 'Tách cảnh thành từng vật thể có định danh riêng có giúp đạt R1, R2, R4 tốt hơn các phương pháp chỉ dùng nét vẽ không, và phải đánh đổi gì?')
put('Text 24', 'Hợp thành cảnh hài hòa có giữ được nhiều vật thể hơn, đồng thời tự nhiên hơn cách trộn cứng không?')
for nm in ('Text 18', 'Text 21', 'Text 24'):
    S[nm].text_frame.vertical_anchor = 3  # middle, so each question sits level with its circle
    S[nm].top = S[nm.replace('18', '17').replace('21', '20').replace('24', '23')].top - (S[nm].height - S['Text 17'].height) // 2
tf = s.notes_slide.notes_text_frame; tf.text = tf.text.replace('R4 đúng quan hệ', 'R4 đúng tương quan')
p.save(sys.argv[2]); print('ok')
