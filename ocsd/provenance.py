"""Content identities, immutable run namespaces and atomic experiment manifests."""
from __future__ import annotations

import contextlib
import ast
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISIONS = {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def files_identity(paths):
    return [(str(p), file_hash(p)) for p in sorted(map(str, paths))]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('w') as f:
            json.dump(value, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def source_identity():
    paths = list((ROOT / 'ocsd').glob('*.py')) + [ROOT / 'requirements-colab.txt', ROOT / 'tools/make_notebooks.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def notebook_identity(path):
    """Hash executable code, ignoring Colab's saved outputs and operational form values.

    Scientific settings are also checked in the experiment contract. Source and generator hashes
    still catch implementation changes. Only literal form assignments are normalised here.
    """
    parameters = {'REPO_URL', 'BRANCH', 'DRIVE_ROOT', 'TIER', 'BACKBONE', 'MAX_MINUTES',
                  'SKIP_JOBS', 'RUN_ID', 'PUSH_RESULTS'}
    cells = []
    for cell in json.loads(Path(path).read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in parameters:
                    try:
                        ast.literal_eval(node.value)
                    except (ValueError, TypeError, SyntaxError):
                        continue
                    node.value = ast.Constant(value=f'<runtime:{name}>')
        cells.append(ast.dump(tree, include_attributes=False))
    return digest(cells)


def dataset_identity(P):
    paths = [p for p in Path(P.benchmarks).rglob('*') if p.is_file() and
             (p.suffix in ('.png', '.npz') or p.name in ('scene.json', 'tuning_split.json'))]
    paths += list((Path(P.data) / 'coco_ref').glob('*.jpg'))
    return digest({str(p.relative_to(P.data)): file_hash(p) for p in sorted(paths)})


def model_kwargs(repo_id):
    if MODEL_REVISIONS:
        if repo_id not in MODEL_REVISIONS:
            raise RuntimeError(f'Model {repo_id} missing from this run model lock')
        return dict(revision=MODEL_REVISIONS[repo_id])
    return {}


def resolve_models(backbone):
    from huggingface_hub import HfApi
    from .config import BACKBONES, AUX_MODELS
    ids = {v for k, v in BACKBONES[backbone].items() if k != 'resolution' and v} | set(AUX_MODELS.values())
    api = HfApi()
    return {repo: api.model_info(repo).sha for repo in sorted(ids)}


def runtime_versions():
    names = ['torch', 'torchvision', 'diffusers', 'transformers', 'peft', 'accelerate', 'huggingface_hub',
             'torchmetrics', 'controlnet_aux', 'numpy', 'pandas', 'scipy', 'torch-fidelity', 'timm',
             'scikit-image', 'opencv-python-headless', 'pillow', 'einops', 'pycocotools', 'matplotlib',
             'tokenizers', 'safetensors']
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def start_run(P, E, run_id='', notebook='notebooks/RUN_ALL.ipynb', model_revisions=None):
    """Blank ID resumes the latest compatible run; explicit ID refuses incompatible state.

    Model SHAs are resolved once for a new run. Resume uses the recorded SHAs, never a moving model branch.
    Legacy results stay at their original paths; new runs never import their caches or tuned choices.
    """
    from .config import Paths, TIERS
    from .runlog import environment
    if run_id and not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('RUN_ID may contain only letters, digits, underscore and hyphen')
    runs = Path(P.root) / 'results/runs'
    code = source_identity()
    nb = ROOT / notebook
    contract = dict(code=code, notebook=notebook, notebook_sha256=notebook_identity(nb) if nb.exists() else None,
                    experiment=dataclasses.asdict(E), tier=TIERS[E.tier], libraries=runtime_versions())
    key = digest(contract)
    dataset = dataset_identity(Paths(P.root, run_id="probe"))
    candidates = [runs / run_id / 'run_manifest.json'] if run_id else sorted(runs.glob('*/run_manifest.json'), reverse=True)
    record = None
    for path in candidates:
        if not path.exists():
            continue
        saved = json.loads(path.read_text())
        match = saved['contract_id'] == key and saved.get('dataset_id') in (None, dataset)
        if model_revisions is not None:
            match = match and saved['model_revisions'] == model_revisions
        if match:
            record, run_id = saved, saved['run_id']
            break
        if run_id:
            raise RuntimeError('This RUN_ID has different code, libraries, configuration or dataset; use a new RUN_ID')
    if record is None:
        run_id = run_id or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
        record = dict(run_id=run_id, started_at=time.time(), status='running', contract=contract, contract_id=key,
                      dataset_id=None, stages={}, jobs={}, model_revisions=(resolve_models(E.backbone)
                      if model_revisions is None else model_revisions), runtime=environment(str(ROOT)))
        def git(*args):
            r = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else None
        record['git'] = dict(commit=git('rev-parse', 'HEAD'), status=git('status', '--porcelain'))
    MODEL_REVISIONS.clear()
    MODEL_REVISIONS.update(record['model_revisions'])
    P = Paths(P.root, run_id=run_id)
    P.makedirs()
    record['status'] = 'running'
    record['notebook_file_sha256'] = file_hash(nb) if nb.exists() else None
    record.pop('ended_at', None)
    record['resumed_at'] = time.time()
    atomic_json(Path(P.results) / 'run_manifest.json', record)
    atomic_json(Path(P.root) / 'results/latest_run.json', dict(run_id=run_id, results=P.results, outputs=P.outputs))
    return P, Run(P, record)


class Run:
    def __init__(self, paths, record):
        self.paths, self.record = paths, record

    def save(self):
        atomic_json(Path(self.paths.results) / 'run_manifest.json', self.record)

    @contextlib.contextmanager
    def stage(self, name):
        from . import runlog
        self.record['stages'][name] = dict(status='running', started_at=time.time())
        self.save()
        try:
            with runlog.stage(name, raise_errors=True):
                yield
        except BaseException as exc:
            self.record['stages'][name].update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
            self.record['status'] = 'failed'
            self.record['ended_at'] = time.time()
            raise
        else:
            self.record['stages'][name]['status'] = 'complete'
        finally:
            self.record['stages'][name]['ended_at'] = time.time()
            progress = Path(self.paths.results) / 'progress.json'
            if progress.exists():
                self.record['jobs'] = json.loads(progress.read_text())
            self.save()

    def seal_data(self):
        self.record['dataset_id'] = dataset_identity(self.paths)
        self.save()

    def plan(self, E):
        from .stages import experiment_plan, log_progress
        self.record['planned_jobs'] = [dict(name=j['name'], split=j['split'], methods=j['methods'],
            seeds=j['seeds'], scenes=[Path(d).name for d in j['scenes']]) for j in experiment_plan(self.paths, E)]
        self.record['jobs'] = log_progress(self.paths, E)
        self.save()

    def finish(self, E):
        from .stages import log_progress
        jobs = log_progress(self.paths, E)
        self.record['jobs'] = {k:dict(v, status='complete' if v['todo'] == 0 else 'incomplete') for k,v in jobs.items()}
        complete = bool(jobs) and all(j['todo'] == 0 for j in jobs.values())
        complete = complete and all(s['status'] == 'complete' for s in self.record['stages'].values())
        self.record['status'] = 'complete' if complete else 'incomplete'
        self.record['ended_at'] = time.time()
        artifacts = {}
        for base in (self.paths.results, self.paths.outputs):
            for p in Path(base).rglob('*'):
                if p.is_file() and p.name != 'run_manifest.json' and p.suffix in ('.json', '.csv', '.png', '.md'):
                    artifacts[str(p.relative_to(self.paths.root))] = file_hash(p)
        self.record['artifacts'] = artifacts
        self.save()
