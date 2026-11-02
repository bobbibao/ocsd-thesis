"""Classify saved instance-screen failure signals, without inference or human-label substitution."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


def audit(root, out):
    root=Path(root).resolve();manifest=json.loads((root/'run_manifest.json').read_text());rid=manifest['run_id']
    def read(rel):
        p=(root/rel).resolve()
        if not p.is_relative_to(root):raise ValueError('Invalid artifact path')
        data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=manifest['artifacts'].get(rel):raise ValueError(f'Artifact changed: {rel}')
        return data
    metrics=csv.DictReader(io.StringIO(read(f'results/runs/{rid}/quickdraw/per_image_instance_repair_focus.csv').decode()))
    rows=[]
    for row in metrics:
        meta=json.loads(read(f"outputs/runs/{rid}/quickdraw/instance_repair_focus/{row['sid']}_s{row['seed']}.json"))
        initial=meta['verify'][0]['check'];final=initial
        for event in meta['verify'][1:]:
            if event.get('accepted'):final=event['check']
        # Deliberately report class/count agreement only, not a fabricated internal joint endpoint.
        internal=final['opr']==1 and final['oce_c']==0
        independent=float(row['opr'])==1 and float(row['oce_c'])==0
        flags=[]
        if meta['route']=='keep' and initial['oce_c']>0:flags.append('count_error_without_missing_target')
        if internal!=independent:flags.append('detector_class_count_disagreement')
        if float(row['opr'])<1:flags.append('independent_missing_instances')
        if float(row['oce_c'])>0:flags.append('independent_count_error')
        if float(row['min_instance_iou'])<.5:flags.append('independent_instance_iou_below_half')
        ra=float(row['ra']) if row['ra'] else math.nan
        if math.isfinite(ra) and ra<1:flags.append('independent_relation_error')
        rows.append(dict(sid=row['sid'],seed=row['seed'],route=meta['route'],internal_opr=final['opr'],
                         internal_count_error=final['oce_c'],independent_opr=row['opr'],independent_count_error=row['oce_c'],
                         joint_layout_success=row['joint_layout_success'],signals=';'.join(flags) or 'no_box_metric_failure',
                         human_verified='no'))
    if not rows:raise ValueError('No metric rows')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle-root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();rows=audit(a.bundle_root,a.out)
    print(f'Wrote {len(rows)} scene diagnostics. Signals are detector outputs, not human ground truth.')
