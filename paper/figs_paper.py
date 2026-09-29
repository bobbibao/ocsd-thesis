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


def e1_curves():
    # QuickDraw-Scenes, 36 identity-learning scenes x 2 seeds (summary.md, E1_quickdraw); OCSD rows preliminary.
    x = [1, 3, 5, 8]
    rows = [
        ('SD + ControlNet', S1, 'o', [94.4, 63.0, 72.2, 40.1], [0.28, 2.00, 2.22, 5.33]),
        ('GLIGEN', S2, 's', [100.0, 70.4, 73.3, 47.8], [0.06, 1.56, 1.83, 6.44]),
        ('Zhang et al. (re-impl.)', S3, '^', [77.8, 48.1, 32.2, 19.0], [1.11, 1.94, 4.06, 6.56]),
        ('OCSD-lite', S4, 'D', [83.3, 72.2, 56.7, 53.1], [0.67, 1.06, 2.72, 4.56]),
        ('OCSD (prelim.)', S5, 'v', [83.3, 70.4, 48.9, 47.2], [0.44, 1.00, 2.83, 5.33]),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.5), dpi=220)
    for ax, k, yl in ((axs[0], 3, 'OPR (%)  ↑'), (axs[1], 4, 'Class-wise OCE  ↓')):
        style(ax)
        for r in rows:
            ax.plot(x, r[k], color=r[1], marker=r[2], ms=6, lw=2, label=r[0],
                    markeredgecolor='white', markeredgewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(['1', '3', '5', '8+'])
        ax.set_xlabel('Objects per scene', color=INK2); ax.set_ylabel(yl, color=INK2)
    axs[0].set_ylim(0, 105); axs[1].set_ylim(0, 7.2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, 'fig2_e1.png'), facecolor='white'); plt.close(fig)


def ablation():
    # QuickDraw-Scenes, 18 scenes (3 and 5 objects), 1 seed; preliminary (run 1, before M5(b) re-tuning).
    rows = [
        ('Full OCSD', 59.6, 13.0),
        ('w/o blended inference (α = 1)', 57.4, 10.2),
        ('blend whole trajectory (α = 0)', 60.4, 12.0),
        ('w/o identity learning (OCSD-lite)', 64.4, 11.3),
        ('w/o L_att in M3', 57.0, 13.0),
        ('M2 with K = 1', 54.4, 16.5),
        ('w/o region attention M5(a)', 63.3, 13.3),
        ('w/o energy guidance M5(b)', 78.1, 8.7),
        ('w/o scene ControlNet M5(c)', 53.0, 14.6),
        ('w/o verification M5(d)', 52.6, 16.1),
        ('w/o all of M5', 68.1, 13.9),
        ('background prompt only', 66.3, 11.7),
        ('global prompt only', 62.6, 13.1),
        ('Zhang et al. (re-impl.)', 40.2, 9.2),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=220)
    style(ax); ax.grid(axis='y', visible=False); ax.grid(axis='x', color=GRID, lw=0.8)
    y = np.arange(len(rows))[::-1]
    full = rows[0][1]
    for yi, (name, m, ci) in zip(y, rows):
        c = S1 if name == 'Full OCSD' else ('#b9b8b2' if 'Zhang' in name else ('#e87b52' if 'energy' in name else '#8fb8ea'))
        ax.barh(yi, m, height=0.62, color=c, edgecolor='white', linewidth=2)
        ax.errorbar(m, yi, xerr=ci, color=INK2, lw=1, capsize=2)
        ax.text(m + ci + 1.2, yi, f'{m:.1f}', va='center', fontsize=8, color=INK)
    ax.axvline(full, color=S1, lw=1, ls='--')
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_xlim(0, 100); ax.set_xlabel('OPR (%) with 95% bootstrap CI', color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_ablation.png'), facecolor='white'); plt.close(fig)


def alpha():
    a = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    opr = [57.8, 57.8, 67.8, 51.1, 52.8, 44.4]; opr_ci = [15.3, 11.7, 10.6, 13.6, 13.1, 13.9]
    miou = [0.417, 0.384, 0.410, 0.259, 0.238, 0.164]; miou_ci = [0.137, 0.110, 0.110, 0.090, 0.074, 0.078]
    fid = [406.78, 388.51, 370.00, 363.72, 328.31, 329.80]
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


if __name__ == '__main__':
    arch(); e1_curves(); ablation(); alpha()
    print(sorted(os.listdir(OUT)))
