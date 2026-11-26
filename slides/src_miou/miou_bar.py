"""v50 (Bobby 2026-11-26, slide 34): mIoU chart gets an extra bar for the old hard-blend M4 ('M4 cũ', 72 scenes x 2
seeds, mIoU 0.529, docs/M4_STUDY_RESULTS.md); the v48/v49 text callout is dropped (built from v47).
Same style as src_bobby/charts_noold.py. Usage: miou_bar.py IN OUT"""
import sys, io
from decimal import Decimal, ROUND_HALF_UP
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
from pptx import Presentation
G, O, LO, INK = '#9AA1AD', '#D9692B', '#F2B48C', '#1F2430'
names = ['Mô hình nền\n(ControlNet)', 'T2I-\nAdapter', 'Zhang và cs.\n(2025)', 'GLIGEN', 'Collage', 'OCSD\n(M4 cũ)', 'OCSD']
v = [0.492, 0.437, 0.435, 0.653, 0.607, 0.529, 0.484]
cols = [G, G, G, G, G, LO, O]
fig, ax = plt.subplots(figsize=(3.1, 2.75), dpi=250)
ax.bar(range(7), v, color=cols, width=0.66)
top = max(v)
for i, x in enumerate(v):
    ax.text(i, x + top * 0.02, str(Decimal(str(x)).quantize(Decimal('0.01'), ROUND_HALF_UP)).replace('.', ','), ha='center', va='bottom', fontsize=6.2,
            fontweight='bold', color=O if i >= 5 else INK)
ax.set_xticks(range(7)); ax.set_xticklabels(names, fontsize=4.8, color=INK)
ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ['top', 'right', 'left']]
ax.set_ylim(0, top * 1.2); ax.set_title('Đúng vị trí, hình dạng (mIoU)', fontsize=8.5, color=INK, pad=5)
fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(); buf.seek(0)
p = Presentation(sys.argv[1]); s = p.slides[33]
pic = [x for x in s.shapes if x.name == 'Picture 19'][0]
_, rid = pic.part.get_or_add_image_part(buf); pic._element.blipFill.blip.rEmbed = rid
nt = s.notes_slide.notes_text_frame; t = nt.text; a = 'đây là điểm yếu nhóm nói rõ.'
assert a in t
nt.text = t.replace(a, a + ' Cột cam nhạt là OCSD với M4 cũ (trộn cứng) trên cùng 72 cảnh: mIoU 0,53, cao hơn bản cuối 0,48, nhưng giữ vật thể chỉ 71 phần trăm và kém tự nhiên hơn, nên nhóm chọn M4 hài hòa.')
p.save(sys.argv[2]); print('ok')
