"""Validate a returned review against manifest-bound image labels; retain missing scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

CRITERIA=('presence','count','layout','shape','relations','attributes','background','realism')
SCORES={'','0','1','2','?','NA'}


def analyze(bundle, review, out):
    root=Path(bundle).resolve();raw=Path(review).read_bytes();data=json.loads(raw)
    manifest=json.loads((root/'run_manifest.json').read_text());rid=manifest['run_id']
    if data.get('schema')!='ocsd-human-review-v1' or data.get('run_id')!=rid:
        raise ValueError('Wrong schema or run ID')
    def read(rel):
        p=(root/rel).resolve()
        if not p.is_relative_to(root):raise ValueError('Invalid artifact path')
        content=p.read_bytes()
        if hashlib.sha256(content).hexdigest()!=manifest['artifacts'].get(rel):raise ValueError('Changed artifact: '+rel)
        return content
    key=json.loads(read(f'results/runs/{rid}/review_key.json'))
    expected={};groups={}
    for item,labels in sorted(key.items()):
        by_hash={}
        for label,method in sorted(labels.items()):
            rel=f'outputs/runs/{rid}/quickdraw/{method}/{item}.png'
            digest=hashlib.sha256(read(rel)).hexdigest()
            by_hash.setdefault(digest,[]).append(label)
        for digest,labels_for_image in by_hash.items():
            group=item+':'+''.join(labels_for_image);groups[group]=labels_for_image
            for label in labels_for_image:
                expected[(item,label)]=dict(group=group,image_sha256=digest,method=key[item][label])
    ratings=data.get('ratings')
    if not isinstance(ratings,dict) or set(ratings)-set(groups):raise ValueError('Unknown rating groups')
    for r in ratings.values():
        if not isinstance(r,dict) or set(r)-set(CRITERIA)-{'comment'}:raise ValueError('Invalid rating fields')
        if any(not isinstance(r.get(c,''),str) or r.get(c,'') not in SCORES for c in CRITERIA):raise ValueError('Invalid score')
        if not isinstance(r.get('comment',''),str):raise ValueError('Invalid comment')
    rows=[];seen=set()
    for row in data.get('rows',[]):
        pair=(row.get('item'),row.get('label'))
        if pair not in expected or pair in seen:raise ValueError('Unknown or duplicate row')
        seen.add(pair);e=expected[pair];r=ratings.get(e['group'],{})
        if any(row.get(k)!=e[k] for k in ('group','image_sha256')):raise ValueError('Image/group mismatch')
        if any(row.get(c,'')!=r.get(c,'') for c in (*CRITERIA,'comment')):raise ValueError('Rows disagree with grouped ratings')
        rows.append(dict(item=pair[0],label=pair[1],**e,**{c:r.get(c,'') for c in (*CRITERIA,'comment')}))
    if seen!=set(expected):raise ValueError('Incomplete label-row export')
    rows.sort(key=lambda x:(x['item'],x['label']))
    totals=[]
    for method in sorted({r['method'] for r in rows}):
        selected=[r for r in rows if r['method']==method]
        for c in CRITERIA:
            values=[int(r[c]) for r in selected if r[c] in {'0','1','2'}]
            totals.append(dict(method=method,criterion=c,n_numeric=len(values),n_blank=sum(r[c]=='' for r in selected),
                               n_uncertain=sum(r[c]=='?' for r in selected),n_na=sum(r[c]=='NA' for r in selected),
                               mean=sum(values)/len(values) if values else '',fully_correct=sum(v==2 for v in values)))
    numeric=sum(r.get(c,'') in {'0','1','2'} for r in ratings.values() for c in CRITERIA)
    summary=dict(run_id=rid,review_sha256=hashlib.sha256(raw).hexdigest(),unique_image_groups=len(groups),
                 groups_with_numeric_scores=sum(any(r.get(c,'') in {'0','1','2'} for c in CRITERIA) for r in ratings.values()),
                 groups_with_comments=sum(bool(r.get('comment','').strip()) for r in ratings.values()),
                 unique_numeric_answers=numeric,possible_unique_answers=len(groups)*len(CRITERIA),
                 scenes_with_any_response=len({g.split(':')[0] for g,r in ratings.items() if any(r.values())}),
                 expanded_method_rows=len(rows),reviewer_count=1,
                 evidence_scope='single user development review; duplicate labels are not independent ratings; no significance test')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,records in [('ratings_by_method.csv',rows),('numeric_summary.csv',totals)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle-root',required=True);p.add_argument('--review',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(analyze(a.bundle_root,a.review,a.out),indent=2))
