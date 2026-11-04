"""Stage 2: locked held-out comparison (docs/HELDOUT_PROTOCOL.md).

18 committed QuickDraw-Scenes x seeds 0/1 x 4 arms:
  direct_box       InstanceDiffusion (external, Wang et al. CVPR 2024), boxes + phrases   [rule-selected]
  direct_scribble  same + 20 ink points per instance                                       [post-hoc, exploratory]
  controlnet       SD 1.5 ControlNet scribble (paper_v2 tuned cn_scale = 1.0)
  instance_repair_count_matte_soft  OCSD route with learned soft matte (foreground_screen settings)

The direct arms run in a child process first (`run_direct`), then are imported with checksums into the
experiment pipeline's output layout so that OWLv2/CLIP evaluation and the report treat all arms alike.
"""
from contextlib import redirect_stdout, redirect_stderr
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

from .provenance import atomic_json, digest, file_hash

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets/heldout_v1'
PROTOCOL = ASSETS / 'protocol.json'
DIRECT_ARMS = (('direct_box', False), ('direct_scribble', True))
EXTERNAL = tuple(name for name, _ in DIRECT_ARMS)
METHODS = ('direct_box', 'direct_scribble', 'controlnet', 'instance_repair_count_matte_soft')
SEEDS = (0, 1)
LOG = 'direct_heldout.log'


# ---------------------------------------------------------------------------------------------- scenes
def manifest():
    return json.loads((ASSETS/'manifest.json').read_text())


def verify_assets():
    files = manifest()
    bad = [rel for rel, h in files.items() if file_hash(ASSETS/rel) != h]
    present = {p.relative_to(ASSETS).as_posix() for p in (ASSETS/'quickdraw').rglob('*') if p.is_file()}
    if bad or present != set(files):
        raise ValueError('Held-out scenes differ from the locked manifest: ' + ', '.join(sorted(bad) or sorted(present ^ set(files))))
    return files


def scene_ids():
    return sorted({rel.split('/')[1] for rel in manifest()})


def install_scenes(P):
    """Copy the locked scenes into this data root; never regenerate or download them."""
    files = verify_assets()
    dst = Path(P.benchmarks)
    for rel, h in files.items():
        target = dst/rel
        if target.exists():
            if file_hash(target) != h:
                raise ValueError('Installed held-out scene was changed: ' + str(target))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ASSETS/rel, target)
    others = sorted(p.name for p in (dst/'quickdraw').iterdir() if p.is_dir() and p.name not in scene_ids())
    if others:
        raise ValueError('Held-out data root contains other scenes: ' + ', '.join(others))
    done = dict(tier='heldout_screen', scenes=len(scene_ids()), manifest_sha256=file_hash(ASSETS/'manifest.json'))
    atomic_json(dst/'.done_heldout_screen.json', done)
    return done


def plan(P, E):
    if list(E.seeds) != list(SEEDS):
        raise ValueError('Held-out protocol fixes seeds 0 and 1')
    scenes = [os.path.join(P.benchmarks, 'quickdraw', sid) for sid in scene_ids()]
    if len(scenes) != 18 or not all(os.path.exists(os.path.join(d, 'scene.json')) for d in scenes):
        raise RuntimeError('Install the 18 locked held-out scenes first (stage A)')
    return [dict(name='heldout_quickdraw', split='quickdraw', scenes=scenes, methods=list(METHODS),
                 seeds=list(SEEDS), priority=0)]


# ---------------------------------------------------------------------------------------------- direct arms
def read_protocol():
    from .direct_instances import SOURCE_COMMIT, sampling_work
    from .data import load_scene
    protocol = json.loads(PROTOCOL.read_text())
    accepted = json.loads((ROOT/protocol['accepted_cpu_report']).read_text())
    if (accepted['status'] != 'pass' or accepted['run_id'] != protocol['accepted_cpu_run']
            or accepted['model_provenance'] != protocol['model_provenance']
            or protocol['model_provenance']['source_commit'] != SOURCE_COMMIT):
        raise ValueError('Held-out direct arms must use the accepted CPU checkpoint/provenance')
    verify_assets()
    scenes = [load_scene(str(ASSETS/'quickdraw'/sid)) for sid in scene_ids()]
    if ([a['name'] for a in protocol['direct_arms']] != list(EXTERNAL) or protocol['seeds'] != list(SEEDS)
            or any(a['fusion'] != 'official_mean' for a in protocol['direct_arms'])):
        raise ValueError('Held-out direct arms are fixed')
    expected = sum(len(SEEDS)*len(EXTERNAL)*sampling_work(s.n, protocol['steps'], protocol['mis']) for s in scenes)
    if expected != protocol['direct_max_unet_sample_calls']:
        raise ValueError('Direct budget does not match the locked plan')
    return protocol, scenes


def identity(protocol, gpu):
    files = ['ocsd/heldout.py', 'ocsd/direct_instances.py', 'ocsd/direct_canary.py', 'ocsd/data.py', 'ocsd/sketch.py',
             'ocsd/provenance.py', 'ocsd/runlog.py', 'tools/run_direct_heldout.py', 'requirements-direct-preflight.txt',
             'assets/heldout_v1/protocol.json', 'assets/heldout_v1/manifest.json', protocol['accepted_cpu_report']]
    versions = {}
    for name in ('torch', 'torchvision', 'transformers', 'huggingface_hub', 'timm', 'einops', 'numpy', 'pillow', 'omegaconf'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return dict(protocol=protocol, source={f: file_hash(ROOT/f) for f in files}, python=platform.python_version(),
                libraries=versions, gpu=gpu)


def finalize(attempt, interrupted=False):
    """Logs/metadata ZIP for the direct child; callable after a kill. Images stay on Drive."""
    attempt = Path(attempt); active = attempt/'active_run.json'; directory = None; status = 'incomplete'
    if active.exists():
        directory = Path(json.loads(active.read_text())['directory']).resolve()
        if not directory.is_relative_to(attempt.parent.resolve()): raise ValueError('Run outside results directory')
        state = json.loads((directory/'run.json').read_text())
        if interrupted:
            state['status'] = 'incomplete'; state['error'] = 'Process interrupted or exceeded the time limit'
            atomic_json(directory/'run.json', state)
        status = state['status']
        history = directory/'attempts'; history.mkdir(exist_ok=True)
        atomic_json(history/(attempt.name+'.json'), dict(status=status, completed=state.get('completed', []),
            attempt_wall_s=state.get('attempt_wall_s'), failed_compute=state.get('failed_compute'),
            error=state.get('error'), interrupted=interrupted, interrupted_work_unknown=interrupted))
        if (attempt/LOG).exists(): shutil.copyfile(attempt/LOG, history/(attempt.name+'.log'))
    elif (attempt/'error.json').exists():
        status = 'failed'
    archive = attempt/f'ocsd_direct_heldout_{directory.name if directory else attempt.name}_logs.zip'
    artifacts = {}
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        for base, prefix in ((directory, 'run'), (attempt, 'attempt')):
            if base is None: continue
            for path in sorted(base.rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.txt', '.log') and path.name != 'result.json':
                    rel = prefix+'/'+path.relative_to(base).as_posix(); z.write(path, rel); artifacts[rel] = file_hash(path)
        z.writestr('checksums.json', json.dumps(artifacts, indent=2))
    result = dict(status=status, archive=str(archive), directory=str(directory) if directory else None)
    atomic_json(attempt/'result.json', result)
    print('Direct held-out bundle:', archive, 'status:', status, flush=True)
    return result


def run_direct(cache, out, attempt, setup_returncode=0):
    from .direct_canary import cached_job, save_job
    from .runlog import _Tee
    started = time.monotonic(); attempt = Path(attempt); attempt.mkdir(parents=True, exist_ok=True)
    state = None; directory = None
    with (attempt/LOG).open('w') as log, redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
        try:
            if setup_returncode: raise RuntimeError('Dependency installation failed; see setup.log')
            import torch
            from .direct_instances import load_components, scene_condition, sampling_work
            if not torch.cuda.is_available(): raise RuntimeError('Select an L4/A100 GPU runtime')
            prop = torch.cuda.get_device_properties(0)
            if prop.total_memory < 20*1024**3: raise RuntimeError('Requires at least 20 GiB VRAM; use L4/A100')
            gpu = dict(name=prop.name, total_memory=prop.total_memory, cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version())
            protocol, scenes = read_protocol(); contract = identity(protocol, gpu)
            signature = digest(contract); rid = 'heldout_'+signature[:20]
            directory = Path(out).resolve()/'runs'/rid; directory.mkdir(parents=True, exist_ok=True)
            jobs = [(s, seed, name, scribble) for s in scenes for seed in SEEDS for name, scribble in DIRECT_ARMS]
            state = dict(run_id=rid, status='running', contract=contract, quality_validated=False,
                         ocsd_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                         completed=[], total_images=len(jobs), planned_unet_sample_calls=protocol['direct_max_unet_sample_calls'])
            atomic_json(directory/'run.json', state); atomic_json(attempt/'active_run.json', dict(directory=str(directory)))
            conditions = {s.sid: scene_condition(s) for s in scenes}
            todo = []
            for s, seed, name, scribble in jobs:
                stem = f'{name}_s{seed}'
                if cached_job(directory/s.sid, stem, signature): state['completed'].append(f'{s.sid}/{stem}')
                else: todo.append((s, seed, name, scribble))
            atomic_json(directory/'run.json', state)
            if todo:
                torch.backends.cudnn.benchmark = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.set_num_threads(4)
                expected = protocol['model_provenance']
                print(f'Locked held-out direct arms: {len(todo)}/{len(jobs)} images remaining; no reranking or retries.', flush=True)
                model = load_components(cache, lock=expected['model_revisions'],
                                        expected_checkpoint_sha256=expected['checkpoint_sha256'])
                if model.provenance != expected: raise ValueError('Loaded components differ from accepted CPU weights/config')
                deadline = started + protocol['direct_max_seconds'] - 60
                for s, seed, name, scribble in todo:
                    if time.monotonic() >= deadline: raise TimeoutError('Budget exhausted before the next image')
                    torch.manual_seed(seed); torch.cuda.reset_peak_memory_stats()
                    print(f'Generating {s.sid} {name} seed {seed}...', flush=True)
                    (directory/s.sid).mkdir(exist_ok=True)
                    pixels, compute = model.generate(conditions[s.sid], seed, protocol['negative'], scribble=scribble,
                        steps=protocol['steps'], mis=protocol['mis'], alpha=protocol['alpha'], deadline=deadline)
                    torch.cuda.synchronize()
                    compute.update(peak_cuda_bytes=torch.cuda.max_memory_allocated(), seed=seed, method=name, scene=s.sid)
                    if compute['unet_sample_calls'] != sampling_work(s.n, protocol['steps'], protocol['mis']):
                        raise RuntimeError('Observed GPU work differs from plan')
                    save_job(directory/s.sid, f'{name}_s{seed}', signature, pixels, compute)
                    state['completed'].append(f'{s.sid}/{name}_s{seed}'); atomic_json(directory/'run.json', state)
                    print(f'Completed {len(state["completed"])}/{len(jobs)}', flush=True)
            state.update(status='complete', gpu_generation_finite=True, quality_validated=False)
            print('Direct held-out arms complete. The notebook continues with ControlNet and OCSD.', flush=True)
        except Exception as exc:
            traceback.print_exc()
            if state is not None: state.update(status='failed', error=str(exc), failed_compute=getattr(exc, 'ocsd_compute', None))
            atomic_json(attempt/'error.json', dict(status='failed', error=str(exc)))
        finally:
            if state is not None:
                state['attempt_wall_s'] = time.monotonic()-started
                atomic_json(directory/'run.json', state)
    return finalize(attempt)


def direct_root(P):
    return Path(P.root)/'direct'


def import_direct(P):
    """Copy the complete, checksum-valid direct run for the locked scenes into P.outputs."""
    manifest_hash = file_hash(ASSETS/'manifest.json'); protocol_hash = file_hash(PROTOCOL)
    runs = []
    for run_json in direct_root(P).glob('runs/*/run.json'):
        state = json.loads(run_json.read_text())
        src = state['contract']['source']
        if (state.get('status') == 'complete' and src.get('assets/heldout_v1/manifest.json') == manifest_hash
                and src.get('assets/heldout_v1/protocol.json') == protocol_hash):
            runs.append((run_json.stat().st_mtime, run_json.parent, state))
    if not runs:
        raise RuntimeError('No complete direct held-out run; run the direct cell (Run all) first')
    _, directory, state = max(runs, key=lambda r: r[0])
    signature = digest(state['contract']); imported = 0
    for sid in scene_ids():
        for name in EXTERNAL:
            for seed in SEEDS:
                stem = f'{name}_s{seed}'
                record = json.loads((directory/sid/(stem+'.json')).read_text())
                png = directory/sid/(stem+'.png')
                if record['signature'] != signature or file_hash(png) != record['png_sha256']:
                    raise ValueError('Direct image failed verification: ' + f'{sid}/{stem}')
                dst = Path(P.outputs)/'quickdraw'/name/f'{sid}_s{seed}.png'
                if dst.exists() and file_hash(dst) == record['png_sha256']:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(png, dst)
                compute = record['compute']
                n_obj = json.loads((ASSETS/'quickdraw'/sid/'scene.json').read_text())['n_obj']
                atomic_json(dst.with_suffix('.json'), dict(method=name, sid=sid, seed=seed, n_obj=n_obj,
                    external_method=compute.get('external_method'), source_run=state['run_id'],
                    source_commit=state['ocsd_commit'], compute=dict(unet_sample_calls=compute['unet_sample_calls']),
                    scene_time=compute['wall_s'], peak_cuda_bytes=compute['peak_cuda_bytes'], sha256=record['png_sha256']))
                imported += 1
    print(f'Imported {imported} direct images from {state["run_id"]}', flush=True)
    return state['run_id']


# ---------------------------------------------------------------------------------------------- report
def review_order(sid, seed):
    order = list(METHODS); random.Random(f'{sid}:{seed}:ocsd-heldout-v1').shuffle(order)
    return dict(zip('ABCD', order))


def make_review(P, out):
    from PIL import Image, ImageDraw
    import textwrap
    from .direct_canary import review_font
    from .data import load_scene
    font, plain = review_font(21)
    review = out/'review'; review.mkdir(parents=True, exist_ok=True)
    key, template = {}, ['Read HOW_TO_REVIEW.txt first. Write pass / fail / unclear; naturalness 0 / 1 / 2 / unclear.', '']
    for n, sid in enumerate(scene_ids(), 1):
        scene = load_scene(os.path.join(P.benchmarks, 'quickdraw', sid))
        for seed in SEEDS:
            mapping = review_order(sid, seed); key[f'{sid}_s{seed}'] = mapping
            name = f'{n:02d}_{sid}_seed{seed}'
            sheet = Image.new('RGB', (2560, 690), 'white'); draw = ImageDraw.Draw(sheet)
            listed = ', '.join(o.phrase for o in scene.objects)
            draw.text((12, 8), plain(f'Sheet {name} | {scene.n} objects'), fill='black', font=font)
            for line, text in enumerate(textwrap.wrap(plain('Caption: ' + scene.caption + ' | Objects: ' + listed), width=190)):
                draw.text((12, 40+line*27), text, fill='black', font=font)
            sheet.paste(Image.fromarray(scene.sketch).convert('RGB').resize((512, 512)), (0, 160))
            draw.text((12, 126), 'Input sketch', fill='black', font=font)
            template.append(f'=== {name} ===')
            for i, (label, m) in enumerate(mapping.items(), 1):
                draw.text((i*512+12, 126), label, fill='black', font=font)
                sheet.paste(Image.open(os.path.join(P.outputs, 'quickdraw', m, f'{sid}_s{seed}.png')).convert('RGB'), (i*512, 160))
                template.append(f'{label}: count=   colour=   shape=   position/relation=   natural(0/1/2)=   note=')
            template += ['Most natural image that still has the right structure (A/B/C/D/tie/none): ', '']
            sheet.save(review/f'{name}.png')
    atomic_json(out/'review_key.json', key)
    (review/'REVIEW_TEMPLATE.txt').write_text('\n'.join(template)+'\n', encoding='utf-8')
    (review/'HOW_TO_REVIEW.txt').write_text(
        'Open the 36 PNG files with a normal image viewer. Each PNG: input sketch, then images A, B, C, D.\n'
        'The order changes per sheet; compare only within one PNG. Check each image against caption and sketch:\n'
        '  count: every listed object present, no missing or extra copies (for 8+ objects, count carefully or write unclear).\n'
        '  colour: each object has its own attribute from the caption (e.g. "a white truck").\n'
        '  shape: recognisable, outline/pose follows the sketch; melted or broken = fail.\n'
        '  position/relation: objects where the sketch puts them; unclear if depth is hard to tell.\n'
        '  natural: 0 = clearly broken, pasted, panels or seams; 1 = looks fake but readable; 2 = natural.\n'
        'Then pick the most natural image that still has the right structure, or tie / none.\n'
        'Fill REVIEW_TEMPLATE.txt (short answers are fine) and send it back with the results ZIP.\n', encoding='utf-8')
    return key


def _scene_means(df, metric):
    return df.groupby(['method', 'sid'])[metric].mean().unstack('method')


def _boot(values, n=10000, seed=0):
    import numpy as np
    values = np.asarray(values, dtype=float); values = values[~np.isnan(values)]
    if not len(values): return (float('nan'),)*3
    rng = np.random.default_rng(seed); means = rng.choice(values, (n, len(values))).mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(P, E):
    import pandas as pd
    out = Path(P.results)
    frames = []
    for m in METHODS:
        csv = out/'quickdraw'/f'per_image_{m}.csv'
        if not csv.exists(): raise RuntimeError('Missing evaluation for ' + m)
        frames.append(pd.read_csv(csv))
    df = pd.concat(frames, ignore_index=True)
    df = df[df.sid.isin(scene_ids()) & df.seed.isin(SEEDS)]
    expected = {(m, sid, s) for m in METHODS for sid in scene_ids() for s in SEEDS}
    found = list(zip(df.method, df.sid, df.seed.astype(int)))
    if set(found) != expected or len(found) != len(expected):
        raise RuntimeError('Incomplete held-out population: %d of %d images evaluated' % (len(set(found) & expected), len(expected)))
    work = []
    for m, sid, s in sorted(expected):
        meta = json.loads((Path(P.outputs)/'quickdraw'/m/f'{sid}_s{s}.json').read_text())
        c = meta.get('compute', {})
        units = sum(c.get(k, 0) + meta.get('gligen_compute_cached', {}).get(k, 0) + meta.get('m2_compute', {}).get(k, 0)
                    - meta.get('m2_compute_in_this_call', {}).get(k, 0)
                    for k in ('unet_sample_calls', 'controlnet_sample_calls', 'gligen_unet_sample_calls'))
        work.append(dict(method=m, sid=sid, seed=s, sampling_work=units, time_s=meta.get('scene_time')))
    df = df.drop(columns=[c for c in ('time_s',) if c in df]).merge(pd.DataFrame(work), on=['method', 'sid', 'seed'], validate='one_to_one')
    metrics = ['count_acc', 'opr', 'oce_c', 'miou', 'ra', 'joint_layout_success', 'clip', 'obj_clip', 'time_s', 'sampling_work']
    metrics = [m for m in metrics if m in df]
    rows = []
    for m in METHODS:
        r = dict(method=m)
        for k in metrics:
            mu, lo, hi = _boot(_scene_means(df, k)[m].to_numpy())
            r[k], r[k+'_lo'], r[k+'_hi'] = mu, lo, hi
        rows.append(r)
    summary = pd.DataFrame(rows)
    pairs = [('direct_box', 'controlnet'), ('direct_box', 'instance_repair_count_matte_soft'),
             ('direct_scribble', 'controlnet'), ('direct_scribble', 'instance_repair_count_matte_soft'),
             ('direct_scribble', 'direct_box')]
    paired = []
    for a, b in pairs:
        for k in ['count_acc', 'opr', 'oce_c', 'miou', 'ra', 'joint_layout_success', 'clip']:
            if k not in df: continue
            t = _scene_means(df, k); mu, lo, hi = _boot((t[a]-t[b]).to_numpy())
            paired.append(dict(a=a, b=b, metric=k, diff=mu, lo=lo, hi=hi,
                               role='post-hoc exploratory' if 'direct_scribble' in (a, b) else 'secondary (automatic)'))
    paired = pd.DataFrame(paired)
    by_bin = df.groupby(['method', 'count_bin'])[[k for k in ('count_acc', 'opr', 'miou', 'joint_layout_success') if k in df]].mean().reset_index()
    tables = out/'tables'; tables.mkdir(exist_ok=True)
    summary.to_csv(tables/'heldout_summary.csv', index=False); paired.to_csv(tables/'heldout_paired.csv', index=False)
    by_bin.to_csv(tables/'heldout_by_count_bin.csv', index=False)
    make_review(P, out)
    def fmt(r, k): return f"{r[k]:.3f} [{r[k+'_lo']:.3f}, {r[k+'_hi']:.3f}]"
    lines = ['# Held-out comparison (automatic metrics; human review pending)', '',
             'Locked protocol: docs/HELDOUT_PROTOCOL.md. 18 scenes x seeds 0/1. Means over scenes (seeds averaged) with',
             '95% bootstrap CIs over scenes. These OWLv2/CLIP numbers are secondary; the primary endpoint is the blind',
             'human review of review/*.png. direct_scribble was added after stage 1 and is exploratory.', '',
             '| method | ' + ' | '.join(metrics) + ' |', '|---|' + '---|'*len(metrics)]
    for _, r in summary.iterrows():
        lines.append(f"| {r['method']} | " + ' | '.join(fmt(r, k) for k in metrics) + ' |')
    lines += ['', '## Paired differences (a - b)', '', '| a | b | metric | diff [95% CI] | role |', '|---|---|---|---|---|']
    for r in paired.itertuples():
        lines.append(f'| {r.a} | {r.b} | {r.metric} | {r.diff:.3f} [{r.lo:.3f}, {r.hi:.3f}] | {r.role} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'summary.json', dict(tier='heldout_screen', research_claim='unproven until human review',
                                         summary=summary.to_dict('records'), paired=paired.to_dict('records')))
    bundle = out/'ocsd_heldout_results.zip'; review_zip = out/'REVIEW_heldout.zip'
    with ZipFile(bundle, 'w', ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in ('.csv', '.json', '.md', '.txt', '.log') and 'review' not in path.parts:
                z.write(path, path.relative_to(out).as_posix())
        for m in METHODS:
            for path in sorted((Path(P.outputs)/'quickdraw'/m).glob('*_s*.json')):
                z.write(path, f'outputs/{m}/{path.name}')
        z.write(out/'review_key.json', 'review_key.json')
    with ZipFile(review_zip, 'w', ZIP_DEFLATED) as z:
        for path in sorted((out/'review').iterdir()):
            z.write(path, 'review/'+path.name)
    print('Held-out results ZIP:', bundle, '| review ZIP:', review_zip, flush=True)
    return dict(figures=[], summary=str(out/'summary.md'), archive=str(bundle), review_archive=str(review_zip))
