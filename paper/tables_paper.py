# Build the paper's result tables from the Colab CSVs, so a new run only needs RESULTS changed.
# Writes _tables/<name>.md; build_paper.py replaces <!-- TABLE:name --> in paper.md with them.
import csv, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get('OCSD_RESULTS', '/mnt/project-files/results/paper_v2')
OUT = os.path.join(HERE, '_tables'); os.makedirs(OUT, exist_ok=True)

NAME = [  # (prefix of the Colab label, English name); order = row order in the paper
    ('SD + ControlNet', 'SD + ControlNet'), ('SD + T2I-Adapter', 'T2I-Adapter'), ('GLIGEN (', 'GLIGEN'),
    ('ControlNet + chú ý vùng', 'CN + region attention'), ('ControlNet + năng lượng', 'CN + attention energy'),
    ('ControlNet + chọn tốt nhất', 'CN best-of-3'), ('ControlNet, best of N', 'CN best-of-N'),
    ('GLIGEN, best of N', 'GLIGEN best-of-N'), ('Collage: ', 'Collage'), ('Collage, best of 3', 'Collage best-of-3'),
    ('Zhang et al.', 'Zhang et al. (re-impl.)'), ('OCSD-lite', 'OCSD-lite'), ('OCSD (đề xuất)', 'OCSD'),
    ('OCSD-v2', 'OCSD-v2'), ('Ảnh thật', 'Real images (ceiling)'),
    # ablation rows
    ('- bỏ suy luận trộn', 'w/o blended inference (α = 1)'), ('- trộn tiềm ẩn toàn bộ', 'blending over the whole trajectory (α = 0)'),
    ('- bỏ chú ý giới hạn', 'w/o region attention M5(a)'), ('- bỏ dẫn hướng năng lượng', 'w/o energy guidance M5(b)'),
    ('- bỏ ControlNet cấp cảnh', 'w/o scene ControlNet M5(c)'), ('- bỏ kiểm tra hậu sinh', 'w/o verification M5(d)'),
    ('- bỏ toàn bộ M5', 'w/o all of M5'), ('- bỏ L_att', 'w/o $\\mathcal{L}_{att}$ in M3'),
    ('- M2 chỉ 1 ứng viên', 'M2 with $K=1$'), ('+ M3 with the full', 'M3 with full 200 + 200 steps'),
    ('- chỉ câu lệnh nền', 'background prompt only'), ('- chỉ câu lệnh toàn cục', 'global prompt only'),
    ('v2 - separate background', 'separate background branch (OCSD sampler)'),
    ('v2 - whole-image regeneration', 'whole-image regeneration instead of repair'),
    ('v2 - no check', 'w/o check and repair M5(d)'), ('v2 - no region', 'w/o region attention M5(a)'),
    ('v2 - no energy', 'w/o energy guidance M5(b)'), ('v2 - no M5', 'w/o all of M5'),
    ('v2 - objects never', 'objects never re-imposed (α = 1)'), ('v2 - without the small', 'w/o small-object mask fix'),
    ('v2 - energy summed', 'energy summed over objects'),
]
def english(label):
    label = label.replace('**', '').strip()
    for p, n in NAME:
        if label.startswith(p):
            return n
    raise SystemExit('unknown method label: ' + label)

COL = {'OPR (%) ↑': ('OPR (%) ↑', 1), 'OCE-lớp ↓': ('OCE~c~ ↓', -1), 'Đếm đúng (%) ↑': ('Count acc. (%) ↑', 1),
       'mIoU ↑': ('mIoU ↑', 1), 'RA (%) ↑': ('RA (%) ↑', 1), 'CLIP ↑': ('CLIP ↑', 1), 'CLIP-đt ↑': ('Obj. CLIP ↑', 1),
       'ID-Sim ↑': ('ID-Sim ↑', 1), 'FID ↓': ('FID ↓', -1), 'KID×10³ ↓': ('KID×10³ ↓', -1)}

def read(name):
    with open(os.path.join(RESULTS, 'tables', name + '.csv'), encoding='utf-8') as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]

def num(cell):
    m = re.match(r'\s*\**\s*(-?[\d.]+)', cell)
    return float(m.group(1)) if m else None

def fmt(cell, ci, col):
    cell = cell.replace('**', '').strip()
    if cell in ('', 'nan', '–'): return '–'
    if not ci: cell = cell.split('±')[0].strip()
    if col.startswith('FID'): cell = '%.1f' % float(cell)
    if col.startswith('KID'): cell = '%.1f' % float(cell)
    return cell.replace(' ± ', '\u00a0±\u00a0')

def table(src, cols, rows=None, ci_cols=('OPR (%) ↑',), out=None, bold=True, extra=None, skip_bold=()):
    head, body = read(src)
    idx = [head.index(c) for c in cols]
    data = [(english(r[0]), r) for r in body]
    if extra: data = extra + data
    if rows: data = sorted([d for d in data if d[0] in rows], key=lambda d: rows.index(d[0]))
    best = {}
    for c, j in zip(cols, idx):
        vals = [(num(r[j]) or None) for n, r in data if n not in skip_bold]
        vals = [v for v in vals if v is not None]
        if vals: best[c] = (max if COL[c][1] > 0 else min)(vals)
    lines = ['| Method | ' + ' | '.join(COL[c][0] for c in cols) + ' |', '|:--' + '|:--:' * len(cols) + '|']
    for n, r in data:
        cells = []
        for c, j in zip(cols, idx):
            s = fmt(r[j], c in ci_cols, c)
            if bold and n not in skip_bold and c in best and num(r[j]) == best[c] and s != '–': s = '**' + s + '**'
            cells.append(s)
        lines.append('| ' + n + ' | ' + ' | '.join(cells) + ' |')
    open(os.path.join(OUT, (out or src) + '.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

MAIN = ['SD + ControlNet', 'T2I-Adapter', 'CN + region attention', 'CN + attention energy', 'CN best-of-3', 'CN best-of-N',
        'GLIGEN', 'GLIGEN best-of-N', 'Collage', 'Collage best-of-3', 'Zhang et al. (re-impl.)', 'OCSD-lite', 'OCSD', 'OCSD-v2']
C_MAIN = ['OPR (%) ↑', 'OCE-lớp ↓', 'Đếm đúng (%) ↑', 'mIoU ↑', 'RA (%) ↑', 'CLIP-đt ↑', 'ID-Sim ↑', 'FID ↓', 'KID×10³ ↓']
table('E3_quickdraw', C_MAIN, MAIN)
table('E3_coco', C_MAIN, MAIN, ci_cols=())
table('E3all_quickdraw', ['OPR (%) ↑', 'OCE-lớp ↓', 'Đếm đúng (%) ↑', 'mIoU ↑', 'RA (%) ↑', 'CLIP ↑'],
      [m for m in MAIN if m not in ('Zhang et al. (re-impl.)', 'OCSD')])
abl_rows = ['OCSD'] + [english(r[0]) for r in read('E4_quickdraw')[1][1:] if not r[0].startswith('Zhang')]
table('E4_quickdraw', ['OPR (%) ↑', 'OCE-lớp ↓', 'mIoU ↑', 'RA (%) ↑', 'CLIP ↑', 'ID-Sim ↑'], abl_rows)
table('E4v2_quickdraw', ['OPR (%) ↑', 'OCE-lớp ↓', 'mIoU ↑', 'RA (%) ↑', 'CLIP ↑', 'ID-Sim ↑'])

# robustness: OPR under three evaluators + OPR@0.5
head, body = read('robustness_quickdraw')
hc, bc = read('robustness_coco')
pick = ['OPR (OWLv2)', 'OPR@0.5 (OWLv2)', 'OPR (OWLv2 + competing queries)', 'OPR (DETR (COCO, closed set))']
lab = ['OWLv2', 'OWLv2 @IoU 0.5', 'OWLv2 + comp. queries', 'DETR']
cq = {english(r[0]): r for r in bc if not r[0].startswith('Kendall')}
lines = ['| Method | ' + ' | '.join('QD ' + l for l in lab) + ' | COCO OWLv2 | COCO DETR |', '|:--' + '|:--:' * (len(lab) + 2) + '|']
for r in body:
    if r[0].startswith('Kendall'):
        tau_q = [r[head.index(p)] for p in pick]; continue
    n = english(r[0]); c = cq[n]
    lines.append('| %s | %s | %s | %s |' % (n, ' | '.join('%.1f' % float(r[head.index(p)]) for p in pick),
                                          '%.1f' % float(c[hc.index(pick[0])]), '%.1f' % float(c[hc.index(pick[3])])))
ktc = [r for r in bc if r[0].startswith('Kendall')][0]
def t(x): return '–' if x in ('nan', '') else '%.2f' % float(x)
lines.append('| Kendall τ vs. OWLv2 ranking | – | – | %s | %s | – | %s |' % (t(tau_q[2]), t(tau_q[3]), t(ktc[hc.index(pick[3])])))
if 'Real images (ceiling)' not in cq: pass
open(os.path.join(OUT, 'robustness.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

# runtime
head, body = read('runtime_quickdraw')
lines = ['| Method | Time / image (s) | M2 / scene (s) | M3 / scene (s) | Generations / image |', '|:--|:--:|:--:|:--:|:--:|']
order = {m: i for i, m in enumerate(MAIN)}
for r in sorted(body, key=lambda r: order.get(english(r[0]), 99)):
    f = lambda x: '–' if x in ('nan', '') or float(x) == 0 else x
    lines.append('| %s | %s | %s | %s | %s |' % (english(r[0]), r[1], f(r[2]), f(r[3]), r[5]))
open(os.path.join(OUT, 'runtime.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

# OPR by object count and by sketch abstraction (QuickDraw E3 subset), one table
h1, b1 = read('E1_quickdraw'); h2, b2 = read('E2_quickdraw')
r1 = {english(r[0]): r for r in b1}; r2 = {english(r[0]): r for r in b2}
lines = ['| Method | ' + ' | '.join(h + (' obj.' if h[0].isdigit() else '') for h in h1[1:] + h2[1:]) + ' |',
         '|:--' + '|:--:' * (len(h1) + len(h2) - 2) + '|']
cols = [[float(c.split('/')[0]) for c in r1[m][1:] + r2[m][1:]] for m in MAIN]
best = [max(c[j] for c in cols) for j in range(len(cols[0]))]
for m, c in zip(MAIN, cols):
    lines.append('| %s | %s |' % (m, ' | '.join(('**%.1f**' if v == best[j] else '%.1f') % v for j, v in enumerate(c))))
open(os.path.join(OUT, 'E1E2.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('tables from', RESULTS, '->', sorted(os.listdir(OUT)))
