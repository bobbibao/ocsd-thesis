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


# ---------------------------------------------------------------- held-out studies (Section 6)
import json
H1 = os.environ.get('OCSD_HELDOUT1', '/mnt/project-files/results/heldout_v1/heldout_20261104')
H2 = os.environ.get('OCSD_HELDOUT2', '/mnt/project-files/results/heldout_v2/heldout2_20261104')
ARM = {'direct_silhouette': 'Silhouettes (OCSD parser)', 'direct_box': 'InstanceDiffusion, boxes',
       'direct_scribble': 'InstanceDiffusion, boxes + 20 ink points', 'controlnet': 'SD + ControlNet scribble',
       'instance_repair_count_matte_soft': 'OCSD soft-matte'}

def summary(path):
    with open(os.path.join(path, 'heldout_summary.csv'), encoding='utf-8') as f:
        return {r['method']: r for r in csv.DictReader(f)}

def bold_best(rows, j, hi=True):
    vals = [float(r[j].strip('*')) for r in rows if r[j] not in ('–',)]
    best = (max if hi else min)(vals)
    for r in rows:
        if r[j] != '–' and float(r[j]) == best: r[j] = '**' + r[j] + '**'

def write(name, head, rows):
    lines = ['| ' + ' | '.join(head) + ' |', '|:--' + '|:--:' * (len(head) - 1) + '|']
    lines += ['| ' + ' | '.join(r) + ' |' for r in rows]
    open(os.path.join(OUT, name + '.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

# held-out v1: blind human review (primary) + OWLv2 (secondary)
hr = json.load(open(os.path.join(H1, 'human_review_analysis.json')))['per_arm']
sm = summary(H1)
order = ['direct_box', 'direct_scribble', 'controlnet', 'instance_repair_count_matte_soft']
rows = []
for a in order:
    h, m = hr[a], sm[a]
    rows.append([ARM[a].replace(' (ours: parser)', ''), '%d/36' % h['success'], str(h['fails']['shape']), str(h['fails']['count']),
                 str(h['fails']['colour']), '%.2f' % h['nat_mean'], '%.2f' % float(m['opr']), '%.2f' % float(m['miou']),
                 '%.2f' % float(m['count_acc'])])
write('heldout1', ['Method', 'Structure success (human) ↑', 'Shape fails ↓', 'Count fails ↓', 'Colour fails ↓',
                   'Naturalness (0–2) ↑', 'OPR ↑', 'mIoU ↑', 'Count acc. ↑'], rows)

# held-out v2: sketch fidelity (primary), OWLv2 (secondary), analyst review, cost
an = json.load(open(os.path.join(H2, 'analyst_review.json')))['per_arm']
sm = summary(H2)
order = ['direct_silhouette', 'direct_box', 'controlnet', 'instance_repair_count_matte_soft']
rows = []
for a in order:
    m = sm[a]; f = lambda k, d=2: ('%.' + str(d) + 'f') % float(m[k])
    rows.append([ARM[a], f('facing_acc'), f('sketch_chamfer', 3), f('count_acc'), f('opr'), f('miou'), f('ra'), f('clip', 1),
                 '%d/36' % an[a]['success'], '%.2f' % an[a]['nat'], '%.1f' % float(m['time_s'])])
for j, hi in ((1, True), (2, False), (3, True), (4, True), (5, True), (6, True), (7, True), (9, True)):
    bold_best(rows, j, hi)
for r in rows:   # structure success: bold the largest count
    pass
best = max(an[a]['success'] for a in order)
for r, a in zip(rows, order):
    if an[a]['success'] == best: r[8] = '**' + r[8] + '**'
write('heldout2', ['Method', 'Facing ↑', 'Chamfer ↓', 'Count acc. ↑', 'OPR ↑', 'mIoU ↑', 'RA ↑', 'CLIP ↑',
                   'Struct. success ↑', 'Natural. ↑', 's/img'], rows)

# held-out v3: complete OCSD v3 vs the original OCSD and the strongest paper_v2 baselines
H3 = os.environ.get('OCSD_HELDOUT3', '/mnt/project-files/results/heldout_v3/heldout3_20261105')
ARM3 = {'ocsd_v3': 'OCSD v3 (ours)', 'direct_box': 'Same generator, boxes only', 'ocsd': 'Original OCSD',
        'gligen': 'GLIGEN', 'collage': 'Collage', 'controlnet': 'SD + ControlNet'}

def v3_table(path, name, n_review):
    an = json.load(open(os.path.join(path, 'analysis.json')))
    sm = summary(path)
    order = ['ocsd_v3', 'direct_box', 'ocsd', 'gligen', 'collage', 'controlnet']
    rows = []
    for a in order:
        m = sm[a]; f = lambda k, d=2: ('%.' + str(d) + 'f') % float(m[k])
        rows.append([ARM3[a], f('opr'), f('miou'), f('ra'), f('count_acc'), f('sketch_chamfer', 3), f('facing_acc'), f('clip', 1),
                     '%d/%d' % (an[a]['structure_pass'], n_review), '%.2f' % (an[a]['natural_mean'] + 1e-9), '%.1f' % float(m['time_s'])])
    for j, hi in ((1, True), (2, True), (3, True), (4, True), (5, False), (6, True), (7, True), (9, True)):
        bold_best(rows, j, hi)
    best = max(an[a]['structure_pass'] for a in order)
    for r, a in zip(rows, order):
        if an[a]['structure_pass'] == best: r[8] = '**' + r[8] + '**'
    write(name, ['Method', 'OPR ↑', 'mIoU ↑', 'RA ↑', 'Count acc. ↑', 'Chamfer ↓', 'Facing ↑', 'CLIP ↑',
                 'Struct. pass ↑', 'Natural. ↑', 's/img'], rows)
    # by object count: OPR / mIoU / count accuracy
    with open(os.path.join(path, 'heldout_by_count_bin.csv'), encoding='utf-8') as f:
        bins = {}
        for r in csv.DictReader(f): bins.setdefault(r['method'], {})[r['count_bin']] = r
    names = [b for b in ('1', '3', '5', '8+') if b in bins['ocsd_v3']]
    rows = [[ARM3[a]] + ['%.2f / %.2f / %.2f' % (float(bins[a][b]['opr']), float(bins[a][b]['miou']), float(bins[a][b]['count_acc']))
                         for b in names] for a in order]
    write(name + '_bins', ['Method'] + [b + ' obj.' for b in names], rows)

v3_table(H3, 'heldout3', 36)
P3 = os.environ.get('OCSD_PAPER3', '/mnt/project-files/results/paper_v3/paper3_20261105')
v3_table(P3, 'paper3', 24)

print('tables from', RESULTS, '->', sorted(os.listdir(OUT)))
