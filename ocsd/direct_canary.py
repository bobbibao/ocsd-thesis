"""Four-image development canary. No metric/paper claim and no automatic expansion."""
from contextlib import redirect_stdout, redirect_stderr
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED

from .provenance import atomic_json, digest, file_hash
from .runlog import _Tee

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'assets/direct_canary_v1/protocol.json'


def render_scene(spec):
    """Render fixed, attributed QuickDraw strokes; no masks or external downloads."""
    import numpy as np
    from PIL import Image, ImageDraw
    size = spec['size']; objects = []
    for obj in spec['objects']:
        strokes = obj['source']['drawing']
        xs = [x for s in strokes for x in s[0]]; ys = [y for s in strokes for y in s[1]]
        x0, y0, x1, y1 = obj['placement_box']
        width, height = max(xs)-min(xs), max(ys)-min(ys)
        scale = min((x1-x0)/max(width, 1), (y1-y0)/max(height, 1))
        ox, oy = x0+(x1-x0-width*scale)/2, y0+(y1-y0-height*scale)/2
        canvas = Image.new('L', (size, size), 255); draw = ImageDraw.Draw(canvas)
        for stroke in strokes:
            points = [(round(ox+(x-min(xs))*scale), round(oy+(y-min(ys))*scale)) for x, y in zip(*stroke)]
            if len(points) == 1:
                x, y = points[0]; draw.ellipse((x-1, y-1, x+1, y+1), fill=0)
            else:
                draw.line(points, fill=0, width=3)
        ink = np.asarray(canvas); yy, xx = np.where(ink < 128)
        if not len(xx): raise ValueError('Empty canary object')
        box = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
        objects.append(SimpleNamespace(cls=obj['cls'], phrase=obj['phrase'], box=box, sketch=ink))
    return SimpleNamespace(sid=spec['sid'], caption=spec['caption'], n=len(objects), objects=objects,
                           sketch=np.minimum.reduce([o.sketch for o in objects]))


def read_protocol():
    from .direct_instances import SOURCE_COMMIT, sampling_work
    protocol = json.loads(PROTOCOL.read_text())
    accepted = json.loads((ROOT/protocol['accepted_cpu_report']).read_text())
    if (accepted['status'] != 'pass' or accepted['run_id'] != protocol['accepted_cpu_run']
            or not accepted['probe']['cached_conditioning_exact'] or not accepted['probe']['unet_probe_finite']
            or accepted['model_provenance'] != protocol['model_provenance']
            or protocol['model_provenance']['source_commit'] != SOURCE_COMMIT):
        raise ValueError('Canary must use the accepted CPU checkpoint/provenance')
    spec = json.loads((ROOT/protocol['scene']).read_text())
    # This is a fixed pilot, not an arbitrary sweep hidden behind a small UI default.
    if protocol['seeds'] != [0, 1] or protocol['methods'] != ['direct_box', 'direct_scribble']:
        raise ValueError('Canary is limited to two paired seeds and two arms')
    expected = len(protocol['seeds'])*len(protocol['methods'])*sampling_work(len(spec['objects']), protocol['steps'], protocol['mis'])
    if expected != protocol['max_unet_sample_calls'] or expected > 864 or protocol['max_seconds'] != 600:
        raise ValueError('Canary budget does not match its frozen plan')
    return protocol, spec


def jobs(protocol):
    return [(seed, method) for seed in protocol['seeds'] for method in protocol['methods']]


def identity(protocol, gpu):
    files = ['ocsd/direct_canary.py', 'ocsd/direct_instances.py', 'ocsd/provenance.py', 'ocsd/runlog.py',
             'tools/run_direct_canary.py', 'requirements-direct-preflight.txt',
             str(PROTOCOL.relative_to(ROOT)), protocol['scene'], protocol['accepted_cpu_report']]
    versions = {}
    for name in ('torch', 'torchvision', 'transformers', 'huggingface_hub', 'timm', 'einops', 'numpy', 'pillow', 'omegaconf'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return dict(protocol=protocol, source={f:file_hash(ROOT/f) for f in files}, python=platform.python_version(),
                libraries=versions, gpu=gpu)


def cached_job(directory, stem, signature):
    meta, png = directory/(stem+'.json'), directory/(stem+'.png')
    if not meta.exists(): return None  # includes an interrupted image write without its completion record
    record = json.loads(meta.read_text())
    if record['signature'] != signature or not png.exists() or file_hash(png) != record['png_sha256']:
        raise ValueError('Changed completed image; preserve the run for audit: '+stem)
    return record


def save_job(directory, stem, signature, pixels, log):
    from PIL import Image
    buffer = io.BytesIO(); Image.fromarray(pixels).save(buffer, format='PNG')
    png = directory/(stem+'.png'); tmp = directory/(stem+'.png.tmp')
    tmp.write_bytes(buffer.getvalue()); os.replace(tmp, png)
    record = dict(signature=signature, png_sha256=file_hash(png), compute=log)
    atomic_json(directory/(stem+'.json'), record)
    return record


def make_review(directory, protocol, spec, condition):
    """PNG originals and a plain-text response form; no JavaScript/HTML dependency."""
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    try: font = ImageFont.truetype('DejaVuSans.ttf', 21)
    except OSError: font = ImageFont.load_default(size=21)
    key, template = {}, ['Đọc HUONG_DAN.txt. Điền đạt/chưa đạt/không rõ; tự nhiên: 0/1/2/không rõ.', '']
    for seed in protocol['seeds']:
        order = protocol['methods'] if seed == 0 else list(reversed(protocol['methods']))
        mapping = dict(zip('AB', order)); key[str(seed)] = mapping
        sheet = Image.new('RGB', (1536, 670), 'white'); draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), f'Seed {seed} | Phác thảo + mô tả: tổng 2 xe đạp và 1 ghế băng', fill='black', font=font)
        for line, text in enumerate(textwrap.wrap(spec['caption_vi'], width=115)):
            draw.text((12, 38+line*27), text, fill='black', font=font)
        sheet.paste(Image.open(directory/'sketch.png').convert('RGB'), (0, 150))
        draw.text((12, 116), 'Phác thảo đầu vào', fill='black', font=font)
        for i, (label, method) in enumerate(mapping.items(), 1):
            draw.text((i*512+12, 116), label, fill='black', font=font)
            sheet.paste(Image.open(directory/f'{method}_s{seed}.png').convert('RGB'), (i*512, 150))
            template += [f'Seed {seed}, ảnh {label}:', '  Số lượng:', '  Hình dáng:', '  Vị trí:',
                         '  Quan hệ:', '  Màu sắc:', '  Tự nhiên (0/1/2/không rõ):', '  Ghi chú:']
        template += [f'Seed {seed}: ảnh tự nhiên hơn nhưng vẫn đúng cấu trúc: A/B/ngang nhau/cả hai chưa đạt', '']
        sheet.save(directory/f'SO_SANH_seed{seed}.png')
    atomic_json(directory/'review_key.json', key)
    atomic_json(directory/'condition.json', condition)
    (directory/'REVIEW_TEMPLATE.txt').write_text('\n'.join(template)+'\n', encoding='utf-8')
    (directory/'HUONG_DAN.txt').write_text(
        'Mở SO_SANH_seed0.png và SO_SANH_seed1.png bằng trình xem ảnh bình thường.\n'
        'Mỗi PNG: phác thảo ở trái, hai ảnh A/B ở phải. A/B đổi thứ tự theo seed.\n'
        'Chỉ so A/B trong cùng PNG; không suy ra cải thiện toàn bộ mô hình từ 4 ảnh này.\n'
        'Yêu cầu: đúng 2 xe đạp (vàng trái, xanh dương phải), 1 ghế băng gỗ phía sau trong công viên.\n'
        'Số lượng: thừa hoặc thiếu đều là lỗi. Hình dáng: so bánh xe, khung xe, tư thế với nét vẽ.\n'
        'Vị trí/quan hệ: xe trái-phải, ghế phía trên/phía sau; nếu chiều sâu khó thấy, ghi không rõ.\n'
        'Màu sắc: có gắn đúng màu cho từng xe không?\n'
        'Tự nhiên: 0 = méo/ghép/dính rõ; 1 = còn giả nhưng nhận ra được; 2 = tự nhiên, hợp ánh sáng/phối cảnh.\n'
        'Điền REVIEW_TEMPLATE.txt rồi gửi lại, hoặc chép câu trả lời vào chat. Không cần HTML hay JSON.\n'
        'Gửi cả ZIP gốc để đối chiếu chi phí/phiên bản. Đây là thử nghiệm phát triển, chưa phải đánh giá paper.\n', encoding='utf-8')


def finalize(attempt, interrupted=False):
    """Also callable after an OS kill/timeout; never needs torch or model weights."""
    attempt = Path(attempt); active = attempt/'active_run.json'; directory = None
    status = 'incomplete'
    if active.exists():
        directory = Path(json.loads(active.read_text())['directory']).resolve()
        if not directory.is_relative_to(attempt.parent.resolve()): raise ValueError('Run outside canary results')
        state = json.loads((directory/'run.json').read_text())
        if interrupted:
            state['status'] = 'incomplete'; state['error'] = 'Process interrupted or exceeded the 600-second limit'
            atomic_json(directory/'run.json', state)
        status = state['status']
        history = directory/'attempts'; history.mkdir(exist_ok=True)
        atomic_json(history/(attempt.name+'.json'), dict(status=status, completed=state.get('completed', []),
            attempt_wall_s=state.get('attempt_wall_s'), failed_compute=state.get('failed_compute'),
            error=state.get('error'), interrupted=interrupted,
            interrupted_work_unknown=interrupted))
        if (attempt/'canary.log').exists(): shutil.copyfile(attempt/'canary.log', history/(attempt.name+'.log'))
    elif (attempt/'error.json').exists():
        status = 'failed'
    archive = attempt/f'ocsd_direct_canary_{directory.name if directory else attempt.name}.zip'
    artifacts = {}
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        for base, prefix in ((directory, 'run'), (attempt, 'attempt')):
            if base is None: continue
            for path in sorted(base.rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.png', '.txt', '.log') and path.name not in ('result.json',):
                    rel = prefix+'/'+path.relative_to(base).as_posix()
                    z.write(path, rel); artifacts[rel] = file_hash(path)
        z.writestr('checksums.json', json.dumps(artifacts, indent=2))
    result = dict(status=status, archive=str(archive), directory=str(directory) if directory else None)
    atomic_json(attempt/'result.json', result)
    print('Canary bundle:', archive, 'status:', status, flush=True)
    return result


def run(cache, out, attempt, setup_returncode=0):
    started = time.monotonic(); attempt = Path(attempt); attempt.mkdir(parents=True, exist_ok=True)
    state = None; directory = None
    with (attempt/'canary.log').open('w') as log, redirect_stdout(_Tee(sys.stdout,log)), redirect_stderr(_Tee(sys.stderr,log)):
        try:
            if setup_returncode: raise RuntimeError('Dependency installation failed; see setup.log')
            import torch
            from PIL import Image
            from .direct_instances import load_components, scene_condition, sampling_work
            if not torch.cuda.is_available(): raise RuntimeError('Select an L4/A100 GPU runtime for the canary')
            prop = torch.cuda.get_device_properties(0)
            if prop.total_memory < 20*1024**3: raise RuntimeError('Canary requires at least 20 GiB VRAM; use L4/A100')
            gpu = dict(name=prop.name, total_memory=prop.total_memory, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
            protocol, spec = read_protocol(); contract = identity(protocol, gpu)
            signature = digest(contract); rid = 'direct_'+signature[:20]
            directory = Path(out).resolve()/'runs'/rid; directory.mkdir(parents=True, exist_ok=True)
            state = dict(run_id=rid, status='running', contract=contract, quality_validated=False,
                         ocsd_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                         completed=[], total_images=4, planned_unet_sample_calls=protocol['max_unet_sample_calls'])
            atomic_json(directory/'run.json', state)
            atomic_json(attempt/'active_run.json', dict(directory=str(directory)))
            scene = render_scene(spec); condition = scene_condition(scene)
            atomic_json(directory/'scene.json', spec); atomic_json(directory/'condition.json', condition)
            Image.fromarray(scene.sketch).save(directory/'sketch.png')
            for i, obj in enumerate(scene.objects): Image.fromarray(obj.sketch).save(directory/f'object_{i}.png')
            todo = []
            for seed, method in jobs(protocol):
                stem = f'{method}_s{seed}'
                record = cached_job(directory, stem, signature)
                if record: state['completed'].append(stem)
                else: todo.append((seed, method))
            atomic_json(directory/'run.json', state)
            if todo:
                torch.backends.cudnn.benchmark = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.set_num_threads(4)
                expected = protocol['model_provenance']
                print(f'Fixed canary: {len(todo)} images remaining, no reranking or automatic retries.', flush=True)
                model = load_components(cache, lock=expected['model_revisions'], expected_checkpoint_sha256=expected['checkpoint_sha256'])
                if model.provenance != expected: raise ValueError('Loaded components differ from accepted CPU weights/config')
                # Reserve time for ZIP export; the notebook additionally kills the child at 600s.
                deadline = started + protocol['max_seconds'] - 30
                for seed, method in todo:
                    if time.monotonic() >= deadline: raise TimeoutError('Canary budget exhausted before the next image')
                    torch.manual_seed(seed); torch.cuda.reset_peak_memory_stats()
                    print(f'Generating {method}, seed {seed}...', flush=True)
                    pixels, compute = model.generate(condition, seed, protocol['negative'], scribble=method=='direct_scribble',
                        steps=protocol['steps'], mis=protocol['mis'], alpha=protocol['alpha'], deadline=deadline)
                    torch.cuda.synchronize()
                    compute.update(peak_cuda_bytes=torch.cuda.max_memory_allocated(), seed=seed, method=method)
                    if compute['unet_sample_calls'] != sampling_work(scene.n, protocol['steps'], protocol['mis']):
                        raise RuntimeError('Observed GPU work differs from plan')
                    stem = f'{method}_s{seed}'; save_job(directory, stem, signature, pixels, compute)
                    state['completed'].append(stem); atomic_json(directory/'run.json', state)
                    print(f'Completed {len(state["completed"])}/4; peak CUDA {compute["peak_cuda_bytes"]/1024**3:.2f} GiB', flush=True)
            make_review(directory, protocol, spec, condition)
            state.update(status='complete', gpu_generation_finite=True, quality_validated=False)
            print('Four images complete. Review PNG comparisons; this is not a quality PASS.', flush=True)
        except Exception as exc:
            traceback.print_exc()
            if state is not None: state.update(status='failed', error=str(exc), failed_compute=getattr(exc, 'ocsd_compute', None))
            atomic_json(attempt/'error.json', dict(status='failed', error=str(exc)))
        finally:
            if state is not None:
                state['attempt_wall_s'] = time.monotonic()-started
                atomic_json(directory/'run.json', state)
    return finalize(attempt)
