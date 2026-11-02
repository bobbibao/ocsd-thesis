"""Audit saved GLIGEN pools without inference. Internal probes are NOT independent ground truth."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def audit(root, out):
    root=Path(root).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'run_manifest.json').read_text())
    rid=manifest['run_id']; rows=[]
    def read(relative):
        p=(root/relative).resolve()
        if not p.is_relative_to(root):raise ValueError('Invalid artifact path')
        data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=manifest['artifacts'][relative]:
            raise ValueError(f'Artifact changed: {relative}')
        return json.loads(data)
    prefix=f'outputs/runs/{rid}/quickdraw/gligen_layout5/'
    for rel in sorted(manifest['artifacts']):
        if not rel.startswith(prefix) or not rel.endswith('.json') or '/_diagnostics/' in rel:continue
        meta=read(rel)
        probes=[v['probe'] for v in meta['verify']]
        bg=sum(p['background']['margin']>0 for p in probes)
        attrs=sum(all(a['margin']>0 for a in p['attributes']) for p in probes)
        complete=sum(p['background']['margin']>0 and all(a['margin']>0 for a in p['attributes'])
                     and p['check']['opr']==1 and p['check']['oce_c']==0 for p in probes)
        routed=read(rel.replace('/gligen_layout5/','/semantic_repair/'))
        chosen=routed['verify'][0]['probe']
        for v in routed['verify'][1:]:
            if v.get('accepted'):chosen=v['probe']
        rows.append(dict(sid=meta['sid'],candidates=len(probes),route=routed['route'],
                         bg_proxy_pass=bg,attributes_proxy_pass=attrs,complete_proxy_pass=complete,
                         best_internal_opr=max(p['check']['opr'] for p in probes),
                         final_internal_opr=chosen['check']['opr'],
                         final_bg_proxy_pass=chosen['background']['margin']>0))
    if not rows:raise ValueError('No saved five-candidate pools')
    with (out/'candidate_pool_audit.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(run_id=rid,scenes=len(rows),candidates=sum(r['candidates'] for r in rows),
                 scenes_with_no_bg_proxy_pass=sum(r['bg_proxy_pass']==0 for r in rows),
                 scenes_with_no_attributes_proxy_pass=sum(r['attributes_proxy_pass']==0 for r in rows),
                 scenes_with_no_complete_proxy_pass=sum(r['complete_proxy_pass']==0 for r in rows),
                 bg_proxy_solution_not_selected=sum(r['bg_proxy_pass']>0 and not r['final_bg_proxy_pass'] for r in rows),
                 independent_quality_claim='none; GDINO/CLIP are internal and can be wrong')
    (out/'candidate_pool_audit.json').write_text(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle-root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(audit(a.bundle_root,a.out),indent=2))
