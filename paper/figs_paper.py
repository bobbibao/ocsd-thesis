# English figures for the OCSD research paper.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import figs as F  # thesis helpers: box, arrow, canvas, scene_strokes, draw_sketch, render_scene

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(OUT, exist_ok=True)
F.D = OUT
plt.rcParams['font.family'] = 'Liberation Serif'

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, S3, S4, S5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'


def arch():
    fig, ax = F.canvas(11, 5.6)
    box, arrow = F.box, F.arrow
    S = F.scene_strokes()
    ax.add_patch(Rectangle((0.1, 3.3), 1.5, 1.5, fc='white', ec='#555555'))
    F.draw_sketch(ax, {k: v for k, v in S.items() if k in ('house', 'tree', 'dog', 'bicycle', 'ground')}, 0.1, 3.3, 1.5, lw=0.9)
    ax.text(0.85, 4.95, 'Scene sketch $\\mathcal{S}$', ha='center', fontsize=9.5, weight='bold')
    box(ax, 0.1, 2.35, 1.5, 0.75, '"a brown dog, a red\nbicycle, a house, a tree\non a sunny day"', fc='#FAFAFA', ec='#555555', fs=7.5)
    ax.text(0.85, 2.2, 'Text prompt $\\mathcal{P}$', ha='center', fontsize=9.5, weight='bold', va='top')
    box(ax, 2.0, 2.6, 1.55, 2.0, 'M1\nSketch parsing &\nobject\ndecomposition\n\n$o_i=(c_i,\\mathbf{m}_i,\\mathbf{b}_i,a_i)$\nrelations $\\mathcal{R}$', fc='gray', fs=8.5)
    arrow(ax, 1.62, 4.0, 1.98, 3.9); arrow(ax, 1.62, 2.7, 1.98, 2.9)
    ax.add_patch(FancyBboxPatch((3.9, 3.35), 4.3, 2.05, boxstyle='round,pad=0.02', fc='#FFF8F0', ec='#C55A11', ls='--'))
    ax.text(6.05, 5.25, 'Object-level branch', ha='center', fontsize=10, weight='bold', color='#C55A11')
    box(ax, 4.05, 3.55, 1.85, 1.45, 'M2  Object generation\nControlNet Scribble\n+ Grounded-SAM\n(K candidates,\nbest-of-K selection)', fc='orange', fs=8.3)
    box(ax, 6.2, 3.55, 1.85, 1.45, 'M3  Identity learning\ntoken ‹o$_i$› + LoRA\nmasked diffusion loss\n+ attention separation', fc='orange', fs=8.3)
    arrow(ax, 5.92, 4.27, 6.18, 4.27)
    arrow(ax, 3.57, 4.1, 4.03, 4.25)
    ax.add_patch(FancyBboxPatch((3.9, 0.15), 6.95, 2.95, boxstyle='round,pad=0.02', fc='#F2F6FC', ec='#2F5597', ls='--'))
    ax.text(7.35, 2.95, 'Scene-level branch', ha='center', fontsize=10, weight='bold', color='#2F5597')
    box(ax, 4.05, 1.55, 2.0, 1.15, 'Foreground latent $\\mathbf{z}_{fg}$\n(objects placed on\nsketch masks)\n+ background $\\mathbf{z}_{bg}$', fc='blue', fs=8.3)
    box(ax, 6.35, 1.55, 2.0, 1.15, 'M4  Blended inference\n$t>\\alpha T$\n$\\mathbf{z}_{bg}(1-\\mathbf{m})+\\mathbf{z}_{fg}\\mathbf{m}$', fc='blue', fs=8.3)
    box(ax, 8.65, 1.55, 2.05, 1.15, 'M4  Customized inference\n$t\\leq\\alpha T$\nglobal prompt\nwith ‹o$_i$› tokens', fc='blue', fs=8.3)
    box(ax, 6.35, 0.3, 4.35, 0.95, 'M5  Object-aware conditioning:\n(a) region-restricted cross-attention  (b) attention-energy guidance\n(c) low-weight scene ControlNet  (d) post-hoc verification + regeneration', fc='purple', fs=8.1)
    arrow(ax, 6.07, 2.12, 6.33, 2.12); arrow(ax, 8.37, 2.12, 8.63, 2.12); arrow(ax, 9.67, 1.27, 9.67, 1.53)
    arrow(ax, 3.57, 3.0, 4.03, 2.3)
    arrow(ax, 5.0, 3.53, 5.0, 2.72); arrow(ax, 7.1, 3.53, 9.4, 2.72, rad=-0.1)
    ax.text(5.08, 3.2, 'images + masks', fontsize=7.5, color='#595959', bbox=dict(fc='white', ec='none', pad=1))
    ax.text(8.4, 3.2, 'identity tokens', fontsize=7.5, color='#595959', bbox=dict(fc='white', ec='none', pad=1))
    arrow(ax, 3.57, 2.75, 6.33, 0.8, rad=0.15, color='#7030A0', ls='--')
    ax.text(4.2, 0.55, 'masks, relations,\nobject phrases', fontsize=7.5, color='#7030A0')
    F.render_scene(ax, 9.0, 3.4, 1.8, variant=0)
    arrow(ax, 9.9, 2.72, 9.9, 3.38)
    ax.text(9.9, 5.35, 'Scene image $\\mathbf{I}$', ha='center', fontsize=9.5, weight='bold')
    F.save(fig, 'fig1_arch.png')


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#9a9993')
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


RESULTS = os.environ.get('OCSD_RESULTS', '/mnt/project-files/results/paper_v2')
S6 = '#7a5bd6'


def read_csv(name):
    import csv
    with open(os.path.join(RESULTS, 'tables', name + '.csv'), encoding='utf-8') as f:
        return list(csv.reader(f))


def mean_ci(cell):
    m, _, ci = cell.replace('**', '').partition('±')
    return float(m), (float(ci) if ci.strip() else None)


def e1_curves():
    # QuickDraw-Scenes E3 subset (36 scenes x 2 seeds), read from E1_quickdraw_long.csv.
    show = [('controlnet', 'SD + ControlNet', S1, 'o'), ('gligen', 'GLIGEN', S2, 's'),
            ('gligen_bon', 'GLIGEN best-of-N', S4, 'P'), ('collage', 'Collage', S3, '^'),
            ('ocsd', 'OCSD', S5, 'v'), ('ocsd_v2', 'OCSD-v2', S6, 'D')]
    rows = read_csv('E1_quickdraw_long')
    h = rows[0]; data = {}
    for r in rows[1:]:
        if r[0] == 'method':
            continue
        data.setdefault(r[0], {})[r[1]] = (100 * float(r[h.index('opr')]), float(r[h.index('oce_c')]))
    bins = ['1', '3', '5', '8+']; x = [1, 3, 5, 8]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=220)
    for ax, k, yl in ((axs[0], 0, 'OPR (%)  ↑'), (axs[1], 1, 'Class-wise OCE  ↓')):
        style(ax)
        for key, name, c, mk in show:
            ax.plot(x, [data[key][b][k] for b in bins], color=c, marker=mk, ms=6, lw=2, label=name,
                    markeredgecolor='white', markeredgewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(bins)
        ax.set_xlabel('Objects per scene', color=INK2); ax.set_ylabel(yl, color=INK2)
    axs[0].set_ylim(30, 102); axs[1].set_ylim(0, 9)
    hh, ll = axs[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc='upper center', ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, 'fig2_e1.png'), facecolor='white'); plt.close(fig)


ABL = [('OCSD (', 'Full OCSD'), ('- bỏ suy luận trộn', 'w/o blended inference (α = 1)'),
       ('- trộn tiềm ẩn toàn bộ', 'blend whole trajectory (α = 0)'), ('OCSD-lite', 'w/o identity learning (OCSD-lite)'),
       ('- bỏ L_att', 'w/o L_att in M3'), ('- M2 chỉ 1', 'M2 with K = 1'), ('+ M3 with the full', 'M3 with full 200 + 200 steps'),
       ('- bỏ chú ý giới hạn', 'w/o region attention M5(a)'), ('- bỏ dẫn hướng', 'w/o energy guidance M5(b)'),
       ('- bỏ ControlNet cấp cảnh', 'w/o scene ControlNet M5(c)'), ('- bỏ kiểm tra', 'w/o verification M5(d)'),
       ('- bỏ toàn bộ M5', 'w/o all of M5'), ('- chỉ câu lệnh nền', 'background prompt only'),
       ('- chỉ câu lệnh toàn cục', 'global prompt only')]


def ablation():
    # QuickDraw-Scenes, 18 scenes (3 and 5 objects), seed 0, read from E4_quickdraw.csv.
    body = read_csv('E4_quickdraw')[1:]
    rows = []
    for pre, name in ABL:
        r = next(r for r in body if r[0].replace('**', '').startswith(pre))
        rows.append((name,) + mean_ci(r[1]))
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=220)
    style(ax); ax.grid(axis='y', visible=False); ax.grid(axis='x', color=GRID, lw=0.8)
    y = np.arange(len(rows))[::-1]
    full = rows[0][1]
    for yi, (name, m, ci) in zip(y, rows):
        c = S1 if name == 'Full OCSD' else '#8fb8ea'
        ax.barh(yi, m, height=0.62, color=c, edgecolor='white', linewidth=2)
        ax.errorbar(m, yi, xerr=ci, color=INK2, lw=1, capsize=2)
        ax.text(min(m + ci + 1.2, 96), yi, f'{m:.1f}', va='center', fontsize=8, color=INK)
    ax.axvline(full, color=S1, lw=1, ls='--')
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_xlim(0, 100); ax.set_xlabel('OPR (%) with 95% bootstrap CI', color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_ablation.png'), facecolor='white'); plt.close(fig)


def alpha():
    body = read_csv('alpha_quickdraw')[1:]
    a = [float(r[0].split('=')[1]) for r in body]
    opr, opr_ci = zip(*[mean_ci(r[1]) for r in body])
    miou, miou_ci = zip(*[mean_ci(r[3]) for r in body])
    fid = [float(r[5]) for r in body]
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0), dpi=220)
    for ax, v, ci, yl, c in ((axs[0], opr, opr_ci, 'OPR (%)  ↑', S1), (axs[1], miou, miou_ci, 'Layout mIoU  ↑', S1), (axs[2], fid, None, 'FID  ↓', S1)):
        style(ax)
        if ci:
            ax.fill_between(a, np.array(v) - ci, np.array(v) + ci, color=c, alpha=0.12, lw=0)
        ax.plot(a, v, color=c, marker='o', ms=6, lw=2, markeredgecolor='white')
        ax.set_xticks(a); ax.set_xlabel('α (fraction of steps without blending)', color=INK2, fontsize=8.5)
        ax.set_ylabel(yl, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_alpha.png'), facecolor='white'); plt.close(fig)


def qualitative():
    from PIL import Image
    src = os.path.join(RESULTS, 'figures', 'qualitative_quickdraw.png')
    im = Image.open(src).convert('RGB'); im.thumbnail((2000, 2000))
    im.save(os.path.join(OUT, 'fig5_qual.jpg'), quality=85)


if __name__ == '__main__':
    arch(); e1_curves(); ablation(); alpha(); qualitative()
    print(sorted(os.listdir(OUT)))
