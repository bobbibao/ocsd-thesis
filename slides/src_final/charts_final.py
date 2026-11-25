"""Charts for the result slides of the FINAL OCSD (SD 1.5, M1-M5, M3 on, harmonised M4), numbers from
results/m4_study_v1 (branch claude/project-thread-5zyzae; 72 QuickDraw-Scenes x seeds 0,1). Run inside charts/."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
def fmt(f, x):
    q = Decimal(str(x)).quantize(Decimal('1') if f == '{:.0f}%' else Decimal('0.01'), ROUND_HALF_UP)
    return (f.format(float(q))).replace('.', ',')
G, O, LO, INK = '#9AA1AD', '#D9692B', '#F2B48C', '#1F2430'
names = ['Mô hình nền\n(ControlNet)', 'T2I-\nAdapter', 'Zhang và cs.\n(2025)', 'GLIGEN', 'Collage', 'OCSD\n(M4 cũ)', 'OCSD']
cols = [G, G, G, G, G, LO, O]
data = {'opr': ([65.5, 62.4, 63.3, 80.1, 82.4, 71.1, 78.5], 'Giữ đủ vật thể (OPR)', '{:.0f}%', 0),
        'miou': ([0.492, 0.437, 0.435, 0.653, 0.607, 0.529, 0.484], 'Đúng vị trí, hình dạng (mIoU)', '{:.2f}', 0),
        'ra': ([32.1, 28.2, 33.7, 56.3, 61.6, 44.5, 50.8], 'Đúng tương quan (RA)', '{:.0f}%', 0),
        'pick': ([21.67, 21.53, 20.73, 21.41, 21.52, 21.16, 21.89], 'Độ tự nhiên (PickScore)', '{:.2f}', 20.3)}
for k, (v, t, f, lo) in data.items():
    fig, ax = plt.subplots(figsize=(3.1, 2.75), dpi=250)
    ax.bar(range(7), [x - lo for x in v], bottom=lo, color=cols, width=0.66)
    top = max(v)
    for i, x in enumerate(v):
        ax.text(i, x + (top - lo) * 0.02, fmt(f, x), ha='center', va='bottom', fontsize=6.6,
                fontweight='bold', color=O if i == 6 else INK)
    ax.set_xticks(range(7)); ax.set_xticklabels(names, fontsize=5.0, color=INK)
    ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ['top', 'right', 'left']]
    ax.set_ylim(lo, lo + (top - lo) * 1.2); ax.set_title(t, fontsize=8.5, color=INK, pad=5)
    fig.tight_layout(); fig.savefig(f'f_{k}.png'); plt.close()
# by object count (summary.md "By object count"), OPR %
E1 = {'Mô hình nền (ControlNet)': ([97.2, 63.9, 61.7, 39.2], G, '--', 2),
      'GLIGEN': ([100.0, 89.8, 75.6, 55.2], '#6F8CC0', ':', 2),
      'Collage': ([97.2, 83.3, 80.6, 68.6], '#3E9474', '-.', 2),
      'OCSD (M4 cũ)': ([88.9, 79.6, 64.4, 51.5], LO, '-', 2),
      'OCSD': ([100.0, 81.5, 70.0, 62.5], O, '-', 3.5)}
fig, ax = plt.subplots(figsize=(7.6, 4.93), dpi=220)
for lab, (y, c, ls, lw) in E1.items():
    ax.plot(range(4), y, ls, color=c, lw=lw, marker='o', ms=6, label=lab)
ax.set_xticks(range(4)); ax.set_xticklabels(['1', '3', '5', '8+'], fontsize=12); ax.tick_params(axis='y', labelsize=12)
ax.set_xlabel('Số vật thể trong phác thảo', fontsize=13); ax.set_ylabel('Giữ đủ vật thể (%)', fontsize=13)
ax.set_ylim(30, 103); ax.grid(axis='y', color='#E5E7EB'); [ax.spines[s].set_visible(False) for s in ['top', 'right']]
ax.legend(fontsize=10.5, frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig('f_count.png'); plt.close()
# development (tuning split, 36 scenes x 2 seeds): SELECTION.md
arms = ['M4 cũ', 'M4 mới\n(tâm)', 'α 0,2', 'nhả\n40%', 'ControlNet\n0,6', 'LoRA\nmuộn', 'Kết hợp\n(chọn)']
opr = [.722, .703, .708, .729, .754, .728, .782]; miou = [.530, .405, .400, .418, .492, .424, .511]
ra = [.491, .377, .413, .410, .478, .405, .531]; pick = [21.14, 21.85, 21.78, 21.91, 21.74, 21.84, 21.74]
x = np.arange(7); w = 0.26
fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=230)
for j, (v, c, lab) in enumerate([(opr, O, 'OPR'), (miou, '#2E5A9C', 'mIoU'), (ra, '#8FB0DA', 'RA')]):
    ax.bar(x + (j - 1) * w, v, w, color=c, label=lab)
for i in range(7):
    ax.text(x[i], 0.83, f'Pick {pick[i]:.2f}'.replace('.', ','), ha='center', fontsize=6.5, color=INK if i else '#777')
ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=7.5, color=INK)
ax.set_ylim(0.3, 0.88); ax.tick_params(axis='y', labelsize=7); ax.grid(axis='y', color='#E5E7EB')
[ax.spines[s].set_visible(False) for s in ['top', 'right']]
ax.axvspan(5.55, 6.45, color='#FDECE1', zorder=0)
ax.legend(fontsize=7.5, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig('f_dev.png'); plt.close()
print('ok')
