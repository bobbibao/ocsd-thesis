# Paired tests for the object-first claims (Section 5.3), not in the Colab stats files.
# Scene sets: 'ocsd' = identity-learning subset (Table 2/6); 'ocsd_lite' = all training-free scenes (Table 3), every available seed averaged per scene as in the Colab tables.
# Unit = scene (mean over seeds); two-sided Wilcoxon (zsplit); Holm over the five planned pairs per metric.
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
R = os.environ.get('OCSD_RESULTS', '/mnt/project-files/results/paper_v2')
PAIRS = [('collage', 'controlnet'), ('collage', 'gligen'), ('collage_bo3', 'gligen_bon'), ('gligen', 'controlnet'), ('collage', 'ocsd_lite')]
def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); out = np.empty_like(ps); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(ps) - r) * ps[i])); out[i] = run
    return out
def sc(split, m, sids, seeds):
    d = pd.read_csv(f'{R}/per_image/{split}/per_image_{m}.csv'); d = d[d.sid.isin(sids) & d.seed.isin(seeds)]
    return d.groupby('sid')[['opr', 'miou', 'ra']].mean()
for split, seeds, ref in (('quickdraw', [0, 1], 'ocsd'), ('quickdraw', [0, 1], 'ocsd_lite'), ('coco', [0], 'ocsd'), ('coco', [0, 1], 'ocsd_lite')):
    sids = sorted(set(pd.read_csv(f'{R}/per_image/{split}/per_image_{ref}.csv').sid))
    print(f'\n{split} seeds {seeds}, {len(sids)} scenes')
    for met in ('opr', 'miou', 'ra'):
        res = []
        for a, b in PAIRS:
            d = (sc(split, a, sids, seeds)[met] - sc(split, b, sids, seeds)[met]).dropna()
            res.append((a, b, d.mean(), 1.0 if np.allclose(d, 0) else wilcoxon(d, zero_method='zsplit').pvalue))
        for (a, b, dm, p), h in zip(res, holm([r[3] for r in res])):
            print(f'  {met:5s} {a:12s} - {b:11s} {dm * (1 if met == "miou" else 100):+8.3f}  p={p:.4f}  p_holm={h:.3f}')
