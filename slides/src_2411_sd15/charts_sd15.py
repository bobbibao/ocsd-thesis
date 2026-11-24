"""Charts for the results slides, one OCSD (SD 1.5, M1-M5), numbers from results/paper_v2 (QuickDraw-Scenes,
E3: 36 scenes x seeds 0,1; E1: the same images by object count). Run inside charts/."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
G, O, INK = '#9AA1AD', '#D9692B', '#1F2430'
names = ['Mô hình nền\n(ControlNet)', 'Zhang và cs.\n(2025)', 'GLIGEN', 'Ghép ảnh\n(Collage)', 'OCSD']
cols = [G, G, G, G, O]
data = {'opr': ([67.4, 61.6, 82.2, 84.2, 74.0], 'Giữ đủ đối tượng (OPR)', '{:.0f}%'),
        'miou': ([0.510, 0.436, 0.669, 0.616, 0.541], 'Đúng vị trí, hình dạng (mIoU)', '{:.2f}'),
        'ra': ([36.6, 35.4, 58.3, 65.5, 50.7], 'Đúng tương quan (RA)', '{:.0f}%')}
for k, (v, t, f) in data.items():
    fig, ax = plt.subplots(figsize=(4.05, 3.0), dpi=250)
    ax.bar(range(5), v, color=cols, width=0.62)
    for i, x in enumerate(v):
        ax.text(i, x + max(v) * 0.02, f.format(x).replace('.', ','), ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=6.6, color=INK)
    ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ['top', 'right', 'left']]
    ax.set_ylim(0, max(v) * 1.18); ax.set_title(t, fontsize=9, color=INK, pad=6)
    fig.tight_layout(); fig.savefig(f'c_{k}.png'); plt.close()
# E1 QuickDraw, OPR % by object count (paper_v2 summary.md)
E1 = {'Mô hình nền (ControlNet)': ([94.4, 63.0, 72.2, 40.1], G, '--', 2),
      'GLIGEN': ([100.0, 85.2, 80.0, 63.5], '#6F8CC0', ':', 2),
      'Ghép ảnh (Collage)': ([94.4, 77.8, 92.2, 72.4], '#3E9474', '-.', 2),
      'OCSD': ([88.9, 75.9, 72.2, 58.9], O, '-', 3.5)}
fig, ax = plt.subplots(figsize=(7.6, 4.93), dpi=220)
for lab, (y, c, ls, lw) in E1.items():
    ax.plot(range(4), y, ls, color=c, lw=lw, marker='o', ms=6, label=lab)
ax.set_xticks(range(4)); ax.set_xticklabels(['1', '3', '5', '8+'], fontsize=12); ax.tick_params(axis='y', labelsize=12)
ax.set_xlabel('Số đối tượng trong phác thảo', fontsize=13); ax.set_ylabel('Giữ đủ đối tượng (%)', fontsize=13)
ax.set_ylim(30, 103); ax.grid(axis='y', color='#E5E7EB'); [ax.spines[s].set_visible(False) for s in ['top', 'right']]
ax.legend(fontsize=10.5, frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig('c_count.png'); plt.close()
print('ok')
