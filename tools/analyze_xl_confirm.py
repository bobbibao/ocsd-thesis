"""Pre-registered analysis of the OCSD-XL confirmation (assets/xl_confirm_v1/protocol.json).
python tools/analyze_xl_confirm.py <results dir with tables/per_image.csv and review_key.json> <blind ratings.json>
Ratings: {"<sid>_s<seed>": {"A": 0|1|2, ...}}, written before the key was opened. Writes analysis.json / analysis.md."""
import json
from math import comb
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = [('opr', 1), ('miou', 1), ('sketch_chamfer', -1), ('naturalness', 1), ('pickscore', 1)]


def sign_test(d):
    w, l = int((d > 0).sum()), int((d < 0).sum())
    n = w + l
    return w, l, len(d) - n, (min(1., 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.)


def boot(v, n=10000):
    m = np.random.default_rng(0).choice(v, (n, len(v))).mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def row(t, a, b, k, better):
    x = t[k][[a, b]].dropna()
    d = (x[a] - x[b]).to_numpy()
    w, l, ties, p = sign_test(better * d)
    mu, lo, hi = boot(d)
    return dict(a=a, b=b, metric=k, scenes=len(d), wins=w, losses=l, ties=ties, p=p, diff=mu, lo=lo, hi=hi,
                favours=a if better * mu > 0 else b)


def main(results, ratings_path, protocol='xl_confirm_v1'):
    results = Path(results)
    p = json.loads((ROOT/'assets'/protocol/'protocol.json').read_text())
    df = pd.read_csv(results/'tables'/'per_image.csv')
    key, ratings = json.loads((results/'review_key.json').read_text()), json.loads(Path(ratings_path).read_text())
    nat = []
    for sheet, mapping in key.items():
        if set(mapping) != set(ratings[sheet]):
            raise ValueError(f'ratings of {sheet} do not cover its panels')
        sid, seed = sheet.rsplit('_s', 1)
        nat += [dict(method=arm, sid=sid, seed=int(seed), naturalness=ratings[sheet][letter]) for letter, arm in mapping.items()]
    df = df.merge(pd.DataFrame(nat), on=['method', 'sid', 'seed'], how='left')
    t = {k: df.groupby(['sid', 'method'])[k].mean().unstack('method') for k, _ in METRICS if k in df}
    primary = [dict(id=q['id'], **row(t, q['a'], q['b'], q['metric'], q['better'])) for q in p['analysis']['primary']]
    order, running = sorted(range(len(primary)), key=lambda i: primary[i]['p']), 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(primary) - rank) * primary[i]['p']))
        primary[i]['p_holm'] = running
    P = {r['id']: r for r in primary}
    sig = lambda r, arm: r['p_holm'] < 0.05 and r['favours'] == arm
    claims = {
        'C-faithful': (sig(P['P1'], 'ocsd_xl_faithful') or sig(P['P2'], 'ocsd_xl_faithful'))
                      and not sig(P['P1'], 'ocsd') and not sig(P['P2'], 'ocsd'),
        'C-natural': sig(P['P3'], 'ocsd_xl_natural') and not sig(P['P4'], 'ocsd')}
    arms = [a for a in t['opr'].columns]
    secondary = [row(t, a, b, k, better) for a in ('ocsd_xl_natural', 'ocsd_xl_faithful') for b in arms
                 if b != a and b in t['opr'] for k, better in METRICS if k in t and a in t[k] and b in t[k]]
    means = {a: {k: float(t[k][a].mean()) for k in t if a in t[k]} for a in arms}
    out = dict(protocol=protocol, primary=primary, claims=claims, means=means, secondary=secondary,
               images={a: int((df.method == a).sum()) for a in arms})
    (results/'analysis.json').write_text(json.dumps(out, indent=1))
    lines = ['# OCSD-XL confirmation: pre-registered analysis', '', '## Primary family (Holm over four)', '',
             '| id | comparison | metric | wins / losses / ties | difference [95% CI] | p | p (Holm) |', '|---|---|---|---|---|---|---|']
    lines += [f"| {r['id']} | {r['a']} vs {r['b']} | {r['metric']} | {r['wins']} / {r['losses']} / {r['ties']} | "
              f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p']:.4f} | {r['p_holm']:.4f} |" for r in primary]
    lines += ['', '## Claims', ''] + [f'* {k}: {"HOLDS" if v else "does not hold"}' for k, v in claims.items()]
    ks = [k for k, _ in METRICS if k in t]
    lines += ['', '## Means over scenes', '', '| arm | images | ' + ' | '.join(ks) + ' |', '|---|---|' + '---|' * len(ks)]
    lines += [f"| {a} | {out['images'][a]} | " + ' | '.join(f"{means[a].get(k, float('nan')):.3f}" for k in ks) + ' |' for a in arms]
    lines += ['', '## Secondary (exploratory, unadjusted)', '', '| a | b | metric | w / l / t | difference [95% CI] | p |', '|---|---|---|---|---|---|']
    lines += [f"| {r['a']} | {r['b']} | {r['metric']} | {r['wins']} / {r['losses']} / {r['ties']} | "
              f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p']:.4f} |" for r in secondary]
    (results/'analysis.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:14]))
    return out


if __name__ == '__main__':
    main(*sys.argv[1:])
