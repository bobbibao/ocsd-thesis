"""Bobby 2026-11-26: slide 6 chart title = only 'Tỉ lệ vật thể còn giữ được' (no 'Mô hình nền:'). Same style as
src_vatthe/figs_vatthe.py. Usage: slide6_chart.py IN OUT"""
import sys, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pptx import Presentation
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(6, 4.4))
v = [97, 64, 62, 39]; b = ax.bar(['1', '3', '5', '8+'], v, color=['#c9ced4', '#b0b7bf', '#8f98a2', '#e0662a'], width=.6)
for r, x_ in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, x_ + 2, f'{x_}%', ha='center', fontsize=18, fontweight='bold')
ax.set_ylim(0, 112); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlabel('Số vật thể trong phác thảo')
ax.set_title('Tỉ lệ vật thể còn giữ được', fontsize=15)
fig.tight_layout(); fig.savefig('img/prob6.png', dpi=200); plt.close(fig)
p = Presentation(sys.argv[1]); s = p.slides[5]
old = next(x for x in s.shapes if x.name == 'Picture 11')
new = s.shapes.add_picture('img/prob6.png', old.left, old.top, old.width, old.height)
old._element.addprevious(new._element); old._element.getparent().remove(old._element)
p.save(sys.argv[2]); print('ok')
