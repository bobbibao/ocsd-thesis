"""Independent OWLv2 tables and anonymized human-review cards for the semantic development screen."""
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import report as R
from .provenance import atomic_json
from .semantic import METHODS


def review_packet(P, plan, require_sketch=False):
    root = Path(P.results)
    cards = root / 'review_blind' / 'cards'
    cards.mkdir(parents=True, exist_ok=True)
    key, rows, prompts = {}, [], []
    for job in plan:
        for scene_dir in job['scenes']:
            sid = Path(scene_dir).name
            sketch = Path(scene_dir)/'sketch.png'
            if require_sketch and not sketch.is_file():
                raise FileNotFoundError(f'Missing review input sketch: {sketch}')
            if sketch.is_file():
                inputs=root/'review_blind'/'inputs';inputs.mkdir(exist_ok=True)
                with Image.open(sketch) as image:image.save(inputs/f'{sid}.png')
            for seed in job['seeds']:
                item = f'{sid}_s{seed}'
                methods = list(job['methods'])
                rng = random.Random(hashlib.sha256((str(P.run_id)+item).encode()).hexdigest())
                rng.shuffle(methods)
                key[item] = dict(zip('ABCD', methods))
                canvas = Image.new('RGB', (1024, 1084), 'white')
                draw = ImageDraw.Draw(canvas)
                for j, method in enumerate(methods):
                    path = Path(P.outputs) / job['split'] / method / f'{item}.png'
                    with Image.open(path) as img:
                        canvas.paste(img.convert('RGB').resize((512,512)), ((j%2)*512,(j//2)*542+30))
                    label = 'ABCD'[j]
                    draw.text(((j%2)*512+8,(j//2)*542+8),label,fill='black')
                    rows.append(dict(reviewer_id='',item=item,label=label,presence='',count='',layout='',shape='',relations='',attributes='',background='',realism='',comment=''))
                meta = json.loads((path.with_suffix('.json')).read_text())
                spec = meta.get('scene_spec', {})
                prompts.append(f"- **{item}**: {spec.get('caption', 'See scene specification')}" +
                               (f" — [input sketch](inputs/{sid}.png)" if sketch.is_file() else '') + '\n')
                canvas.save(cards / f'{item}.png')
    atomic_json(root/'review_key.json', key)
    with (root/'review_blind'/'ratings.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    (root/'review_blind'/'README.md').write_text(
        '# Blind review before opening review_key.json\n\n'
        'Rate each card label independently from 1 (very poor) to 5 (fully meets request). '
        'Presence: requested classes; count: exact multiplicity; layout: positions; shape: sketch correspondence; relations: relative arrangement; attributes: requested color/material/size; '
        'background: correct setting; realism: coherent shapes, boundaries and lighting. '
        'Record visible defects in comments. Do not consult method names or scores before rating. '
        'This packet enables independent review; blank ratings are not completed evidence.\n\n'+''.join(prompts))


def report(P, E):
    from .stages import experiment_plan
    plan = experiment_plan(P, E)
    methods = METHODS
    ref = 'semantic_repair'
    if E.tier == 'instance_screen':
        from .instance_repair import METHODS as methods
        ref = 'instance_repair_focus'
    if E.tier == 'count_screen':
        from .count_repair import METHODS as methods
        ref = 'instance_repair_count'
    if E.tier == 'clean_screen':
        from .clean_repair import METHODS as methods
        ref = 'instance_repair_count_clean'
    if E.tier == 'foreground_screen':
        from .foreground import METHODS as methods
        ref = 'instance_repair_count_matte_soft'
    df = R.load_per_image(P.results, 'quickdraw')
    expected = {(m,Path(s).name,int(seed)) for j in plan for m in j['methods'] for s in j['scenes'] for seed in j['seeds']}
    found = [(r.method,r.sid,int(r.seed)) for r in df.itertuples()]
    if not expected or set(found) != expected or len(found) != len(set(found)):
        raise RuntimeError('Incomplete semantic screening population')
    df['standalone_time_s'] = df.time_s.fillna(0)
    for col in ('m2_s','m3_s'):
        if col in df: df['standalone_time_s'] += df[col].fillna(0)
    work_rows = []
    for job in plan:
        for directory in job['scenes']:
            for m in job['methods']:
                for seed in job['seeds']:
                    sid = Path(directory).name
                    meta = json.loads((Path(P.outputs)/job['split']/m/f'{sid}_s{seed}.json').read_text())
                    compute = meta['compute']
                    units = sum(compute.get(k,0) + meta.get('gligen_compute_cached',{}).get(k,0) + meta.get('m2_compute',{}).get(k,0) - meta.get('m2_compute_in_this_call',{}).get(k,0)
                                for k in ('unet_sample_calls','controlnet_sample_calls','gligen_unet_sample_calls'))
                    work_rows.append(dict(method=m,sid=sid,seed=seed,sampling_work=units,route=meta.get('route','sparse'),
                                          m2_candidates=meta.get('m2_candidate_count',0)))
    work = pd.DataFrame(work_rows)
    df = df.merge(work[['method','sid','seed','sampling_work']],on=['method','sid','seed'],validate='one_to_one')
    ps = R.per_scene(df)
    summary_metrics=['opr','oce_c','miou','ra','clip','standalone_time_s','sampling_work']
    paired_metrics=['opr','oce_c','miou','ra','obj_clip']
    if E.tier in ('instance_screen','count_screen','clean_screen','foreground_screen') and 'joint_layout_success' not in ps:
        raise RuntimeError('Instance screen requires the joint layout endpoint; rerun evaluation')
    if 'joint_layout_success' in ps:
        summary_metrics.append('joint_layout_success');paired_metrics.append('joint_layout_success')
    summary = R.summarize(ps,metrics=summary_metrics)
    if E.tier in ('count_screen','clean_screen','foreground_screen'):
        # One deliberately selected scene cannot estimate population uncertainty.
        summary = summary.drop(columns=[c for c in summary.columns if c.endswith(('_lo','_hi'))])
    tests = (pd.DataFrame(columns=['ref','other','metric','n','diff','p']) if E.tier in ('count_screen','clean_screen','foreground_screen') else
             R.paired_tests(ps,ref=ref,others=[m for m in methods if m!=ref],metrics=paired_metrics))
    tables = Path(P.results)/'tables'; tables.mkdir(parents=True,exist_ok=True)
    summary.to_csv(tables/'screen_means.csv',index=False)
    tests.to_csv(tables/'screen_paired.csv',index=False)
    work.to_csv(tables/'sampling_work.csv',index=False)
    review_packet(P,plan,require_sketch=E.tier in ('instance_screen','count_screen','clean_screen','foreground_screen'))
    result = dict(tier=E.tier,decision='requires_blind_review',research_claim='unproven',
                  n_images=len(df),n_scenes=df.sid.nunique(),figures=[],tables=['screen_means','screen_paired','sampling_work'])
    atomic_json(Path(P.results)/'summary.json',result)
    (Path(P.results)/'summary.md').write_text(
        '# Development screen — requires blind review\n\n'+summary.round(4).to_markdown(index=False)+
        '\n\nCLIP is part of routing/selection and is not independent semantic evidence. '
        'OWLv2 measures class/layout, not complete attributes, shape or background. joint_layout_success requires exact class/count coverage, every matched box IoU >= .5, and all applicable relations. It is not a silhouette metric. Complete review_blind/ratings.csv '
        'before inspecting review_key.json. No automatic paper/full escalation.\n\n'
        'Sampling work sums batch-weighted UNet and ControlNet calls; it excludes CLIP/GDINO/SAM/VAE and I/O. '
        'It is a budget proxy, not equal FLOPs or equal wall time. Standalone times charge cached M2 and shared GLIGEN generation durations; these are reconstructed costs, not independent timed runs. '
        'All scenes remain development data. Count/clean/foreground screens are one-scene mechanism diagnostics, with no significance tests.\n')
    return result
