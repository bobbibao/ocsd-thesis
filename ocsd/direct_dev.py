"""Stage-1 development study for direct instance conditioning (8 scenes x 2 seeds x 3 arms).

External baseline: InstanceDiffusion (Wang et al., CVPR 2024), pinned official core.
Our change under test: the region-aware MIS merge (`fusion='region'`).
Locked protocol: assets/direct_dev_v1/protocol.json. No reranking, retries with other
settings or extra seeds. Completed images are resumed; changed ones stop the run.
"""
from contextlib import redirect_stdout, redirect_stderr
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

from .direct_canary import render_scene, cached_job, save_job, review_font
from .provenance import atomic_json, digest, file_hash
from .runlog import _Tee

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'assets/direct_dev_v1/protocol.json'
ARMS = ('direct_box', 'direct_scribble', 'direct_scribble_region')
LOG = 'direct_dev.log'


def read_protocol():
    from .direct_instances import SOURCE_COMMIT, sampling_work, FUSIONS
    protocol = json.loads(PROTOCOL.read_text())
    accepted = json.loads((ROOT/protocol['accepted_cpu_report']).read_text())
    if (accepted['status'] != 'pass' or accepted['run_id'] != protocol['accepted_cpu_run']
            or accepted['model_provenance'] != protocol['model_provenance']
            or protocol['model_provenance']['source_commit'] != SOURCE_COMMIT):
        raise ValueError('Development run must use the accepted CPU checkpoint/provenance')
    scenes = json.loads((ROOT/protocol['scenes']).read_text())['scenes']
    if (len(scenes) != 8 or protocol['seeds'] != [0, 1] or tuple(a['name'] for a in protocol['arms']) != ARMS
            or any(a['fusion'] not in FUSIONS for a in protocol['arms'])):
        raise ValueError('Development study is fixed to 8 scenes, seeds 0/1 and three arms')
    expected = sum(len(protocol['seeds'])*len(ARMS)*sampling_work(len(s['objects']), protocol['steps'], protocol['mis'])
                   for s in scenes)
    if expected != protocol['max_unet_sample_calls'] or protocol['max_seconds'] != 2400:
        raise ValueError('Development budget does not match its frozen plan')
    return protocol, scenes


def jobs(protocol, scenes):
    """Scene-major order so an interrupted run still has complete paired sets."""
    return [(s['sid'], seed, arm) for s in scenes for seed in protocol['seeds'] for arm in protocol['arms']]


def identity(protocol, gpu):
    files = ['ocsd/direct_dev.py', 'ocsd/direct_canary.py', 'ocsd/direct_instances.py', 'ocsd/provenance.py',
             'ocsd/runlog.py', 'tools/run_direct_dev.py', 'requirements-direct-preflight.txt',
             str(PROTOCOL.relative_to(ROOT)), protocol['scenes'], protocol['accepted_cpu_report']]
    versions = {}
    for name in ('torch', 'torchvision', 'transformers', 'huggingface_hub', 'timm', 'einops', 'numpy', 'pillow', 'omegaconf'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return dict(protocol=protocol, source={f: file_hash(ROOT/f) for f in files}, python=platform.python_version(),
                libraries=versions, gpu=gpu)


def review_order(sid, seed):
    """Fixed, seeded label order per sheet; the key is stored separately."""
    order = list(ARMS); random.Random(f'{sid}:{seed}:ocsd-dev-v1').shuffle(order)
    return dict(zip('ABC', order))


def make_review(directory, protocol, scenes):
    from PIL import Image, ImageDraw
    import textwrap
    font, plain = review_font(21)
    review = directory/'review'; review.mkdir(exist_ok=True)
    key, template = {}, ['Read HOW_TO_REVIEW.txt first. Write pass / fail / unclear; naturalness 0 / 1 / 2 / unclear.', '']
    for n, spec in enumerate(scenes, 1):
        for seed in protocol['seeds']:
            sid = spec['sid']; mapping = review_order(sid, seed); key[f'{sid}_s{seed}'] = mapping
            name = f'{n:02d}_{sid}_seed{seed}'
            sheet = Image.new('RGB', (2048, 690), 'white'); draw = ImageDraw.Draw(sheet)
            draw.text((12, 8), plain(f'Sheet {name} | objects: ' + ', '.join(o['phrase'] for o in spec['objects'])),
                      fill='black', font=font)
            for line, text in enumerate(textwrap.wrap(plain('Caption: ' + spec['caption']), width=150)):
                draw.text((12, 40+line*27), text, fill='black', font=font)
            sheet.paste(Image.open(directory/sid/'sketch.png').convert('RGB'), (0, 160))
            draw.text((12, 126), 'Input sketch', fill='black', font=font)
            template.append(f'=== {name} ===')
            for i, (label, arm) in enumerate(mapping.items(), 1):
                draw.text((i*512+12, 126), label, fill='black', font=font)
                sheet.paste(Image.open(directory/sid/f'{arm}_s{seed}.png').convert('RGB'), (i*512, 160))
                template.append(f'{label}: count=   colour=   shape=   position/relation=   natural(0/1/2)=   note=')
            template += ['Most natural image that still has the right structure (A/B/C/tie/none): ', '']
            sheet.save(review/f'{name}.png')
    atomic_json(directory/'review_key.json', key)
    (review/'REVIEW_TEMPLATE.txt').write_text('\n'.join(template)+'\n', encoding='utf-8')
    (review/'HOW_TO_REVIEW.txt').write_text(
        'Open the 16 PNG files in review/ with a normal image viewer (no HTML needed).\n'
        'Each PNG: input sketch on the left, three generated images A, B, C. The order changes per sheet.\n'
        'Compare only within one PNG. Judge each image against the caption and sketch:\n'
        '  count: exactly the listed objects, no missing or extra copies.\n'
        '  colour: each object has its own colour from the caption (e.g. red car left, white car right).\n'
        '  shape: the object is recognisable and its outline/pose follows the sketch; melted or broken = fail.\n'
        '  position/relation: objects where the sketch puts them; write unclear if depth is hard to tell.\n'
        '  natural: 0 = clearly broken, pasted, panels or seams; 1 = looks fake but readable; 2 = natural lighting and perspective.\n'
        'Then pick the most natural image that still has the right structure, or tie / none.\n'
        'Fill REVIEW_TEMPLATE.txt and send it back, or paste your answers in the chat. Short notes are welcome.\n'
        'This is a development study on 8 scenes, not a paper evaluation.\n', encoding='utf-8')


def finalize(attempt, interrupted=False):
    """Also callable after an OS kill/timeout; never needs torch or model weights."""
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
    label = directory.name if directory else attempt.name
    archive = attempt/f'ocsd_direct_dev_{label}.zip'
    review_zip = attempt/f'REVIEW_direct_dev_{label}.zip'
    artifacts = {}
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        for base, prefix in ((directory, 'run'), (attempt, 'attempt')):
            if base is None: continue
            for path in sorted(base.rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.png', '.txt', '.log') and path.name != 'result.json':
                    rel = prefix+'/'+path.relative_to(base).as_posix()
                    z.write(path, rel); artifacts[rel] = file_hash(path)
        z.writestr('checksums.json', json.dumps(artifacts, indent=2))
    result = dict(status=status, archive=str(archive), directory=str(directory) if directory else None)
    if directory is not None and (directory/'review').is_dir():
        # Small blind bundle for the reviewer: sheets + instructions, no key.
        with ZipFile(review_zip, 'w', ZIP_DEFLATED) as z:
            for path in sorted((directory/'review').iterdir()):
                if path.is_file(): z.write(path, 'review/'+path.name)
        result['review_archive'] = str(review_zip)
    atomic_json(attempt/'result.json', result)
    print('Development bundle:', archive, 'status:', status, flush=True)
    return result


def run(cache, out, attempt, setup_returncode=0):
    started = time.monotonic(); attempt = Path(attempt); attempt.mkdir(parents=True, exist_ok=True)
    state = None; directory = None
    with (attempt/LOG).open('w') as log, redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
        try:
            if setup_returncode: raise RuntimeError('Dependency installation failed; see setup.log')
            import torch
            from PIL import Image
            from .direct_instances import load_components, scene_condition, sampling_work
            if not torch.cuda.is_available(): raise RuntimeError('Select an L4/A100 GPU runtime')
            prop = torch.cuda.get_device_properties(0)
            # Canary peak was 5.95 GiB; the gate stays at L4/A100 because the CPU-side
            # checkpoint load needs their host RAM.
            if prop.total_memory < 20*1024**3: raise RuntimeError('Requires at least 20 GiB VRAM; use L4/A100')
            gpu = dict(name=prop.name, total_memory=prop.total_memory, cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version())
            protocol, scenes = read_protocol(); contract = identity(protocol, gpu)
            signature = digest(contract); rid = 'dev_'+signature[:20]
            directory = Path(out).resolve()/'runs'/rid; directory.mkdir(parents=True, exist_ok=True)
            all_jobs = jobs(protocol, scenes)
            state = dict(run_id=rid, status='running', contract=contract, quality_validated=False,
                         ocsd_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                         completed=[], total_images=len(all_jobs), planned_unet_sample_calls=protocol['max_unet_sample_calls'])
            atomic_json(directory/'run.json', state)
            atomic_json(attempt/'active_run.json', dict(directory=str(directory)))
            conditions, sizes = {}, {}
            for spec in scenes:
                scene = render_scene(spec); sdir = directory/spec['sid']; sdir.mkdir(exist_ok=True)
                conditions[spec['sid']] = scene_condition(scene); sizes[spec['sid']] = scene.n
                atomic_json(sdir/'condition.json', conditions[spec['sid']])
                Image.fromarray(scene.sketch).save(sdir/'sketch.png')
            todo = []
            for sid, seed, arm in all_jobs:
                stem = f'{arm["name"]}_s{seed}'
                if cached_job(directory/sid, stem, signature): state['completed'].append(f'{sid}/{stem}')
                else: todo.append((sid, seed, arm))
            atomic_json(directory/'run.json', state)
            if todo:
                torch.backends.cudnn.benchmark = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.set_num_threads(4)
                expected = protocol['model_provenance']
                print(f'Fixed development study: {len(todo)}/{len(all_jobs)} images remaining; '
                      'no reranking or automatic retries.', flush=True)
                model = load_components(cache, lock=expected['model_revisions'],
                                        expected_checkpoint_sha256=expected['checkpoint_sha256'])
                if model.provenance != expected: raise ValueError('Loaded components differ from accepted CPU weights/config')
                deadline = started + protocol['max_seconds'] - 60
                for sid, seed, arm in todo:
                    if time.monotonic() >= deadline: raise TimeoutError('Budget exhausted before the next image')
                    torch.manual_seed(seed); torch.cuda.reset_peak_memory_stats()
                    print(f'Generating {sid} {arm["name"]} seed {seed}...', flush=True)
                    pixels, compute = model.generate(conditions[sid], seed, protocol['negative'], scribble=arm['scribble'],
                        steps=protocol['steps'], mis=protocol['mis'], alpha=protocol['alpha'], deadline=deadline,
                        fusion=arm['fusion'])
                    torch.cuda.synchronize()
                    compute.update(peak_cuda_bytes=torch.cuda.max_memory_allocated(), seed=seed, method=arm['name'], scene=sid)
                    if compute['unet_sample_calls'] != sampling_work(sizes[sid], protocol['steps'], protocol['mis']):
                        raise RuntimeError('Observed GPU work differs from plan')
                    stem = f'{arm["name"]}_s{seed}'; save_job(directory/sid, stem, signature, pixels, compute)
                    state['completed'].append(f'{sid}/{stem}'); atomic_json(directory/'run.json', state)
                    print(f'Completed {len(state["completed"])}/{len(all_jobs)}; '
                          f'peak CUDA {compute["peak_cuda_bytes"]/1024**3:.2f} GiB', flush=True)
            make_review(directory, protocol, scenes)
            state.update(status='complete', gpu_generation_finite=True, quality_validated=False)
            print('All images complete. Review the PNG sheets; this is not a quality PASS.', flush=True)
        except Exception as exc:
            traceback.print_exc()
            if state is not None: state.update(status='failed', error=str(exc), failed_compute=getattr(exc, 'ocsd_compute', None))
            atomic_json(attempt/'error.json', dict(status='failed', error=str(exc)))
        finally:
            if state is not None:
                state['attempt_wall_s'] = time.monotonic()-started
                atomic_json(directory/'run.json', state)
    return finalize(attempt)
