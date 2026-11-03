"""Check real direct-conditioning weights on CPU; never generate a scene or use CUDA."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid
from zipfile import ZipFile, ZIP_DEFLATED

# Set before importing torch or any model code, including in a GPU Colab runtime.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocsd.provenance import atomic_json, file_hash


def load_components(cache):
    # Import inside the recorded try block: dependency failures must produce a ZIP too.
    from ocsd.direct_instances import load_components as load
    return load(cache)


class Tee:
    def __init__(self, terminal, file): self.terminal, self.file = terminal, file
    def write(self, text):
        self.terminal.write(text); self.file.write(text); self.flush()
        return len(text)
    def flush(self): self.terminal.flush(); self.file.flush()


def probe(model):
    """Real pretrained text encoder, UniFusion and UNet on a small noise tensor.

    A numerical compatibility probe only. No VAE scene decode or photo scoring.
    """
    import numpy as np
    import torch
    from ocsd.direct_instances import conditioning_batch, cached_conditioning, NullCondition, sketch_points
    objects=[]
    for i, phrase in enumerate(('a red bicycle', 'a blue train')):
        ink=np.full((64,64),255,np.uint8)
        ink[12+i*25:22+i*25,8+i*30]=0;ink[21+i*25,8+i*30:23+i*30]=0
        objects.append(dict(index=i,phrase=phrase,box=[.1+i*.45,.1+i*.4,.4+i*.45,.4+i*.4],scribble=sketch_points(ink)))
    condition=dict(caption='a red bicycle and a blue train near a train station',objects=objects)
    features,contexts,negative=model.encode(condition,'blurry')
    if not torch.isfinite(features).all() or features.shape != (2,768):
        raise RuntimeError('Unexpected phrase embedding output')
    batch=conditioning_batch(condition,features)
    model.model.position_net.test_drop_scribbles=False
    model.model.position_net.test_drop_masks=True
    model.model.grounding_tokenizer_input=NullCondition(batch)
    args=tuple(batch[k] for k in ('boxes','masks','positive_embeddings','scribbles','polygons','segs','points'))
    print('Checking real learned box/scribble conditioning...',flush=True)
    with cached_conditioning(model.model.position_net) as cache:
        first=model.model.position_net(*args)[0]
        second=model.model.position_net(*args)[0]
    uncached=model.model.position_net(*args)[0]
    if not torch.isfinite(first).all() or not torch.equal(first,second) or not torch.equal(first,uncached):
        raise RuntimeError('Real-weight conditioning cache parity failed')
    changed=list(args);changed[3]=batch['scribbles'].clone();changed[3][:,:2]=.7
    difference=float((first-model.model.position_net(*changed)[0]).abs().max())
    if not math.isfinite(difference) or difference <= 0:
        raise RuntimeError('Scribble input does not produce a finite change in the learned conditioner')
    print('Checking pretrained UNet on an 8x8 latent (no scene generation)...',flush=True)
    noise=torch.randn((1,4,8,8),generator=torch.Generator('cpu').manual_seed(0))
    item=dict(x=noise,timesteps=torch.tensor([501]),context=contexts[0],grounding_input=batch)
    out=model.model(item)
    if out.shape != noise.shape or not torch.isfinite(out).all():
        raise RuntimeError('Pretrained UNet compatibility probe failed')
    return dict(phrase_features=list(features.shape),conditioning_shape=list(first.shape),
                conditioning_cache=cache,cached_conditioning_exact=True,changed_scribble_max_difference=difference,
                unet_probe_shape=list(out.shape),unet_probe_finite=True,device='cpu',scene_generation=False,
                quality_validated=False,gpu_fp16_validated=False)


def run(cache, out, setup_log=None, setup_returncode=0):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    directory=out/rid;directory.mkdir()
    root=Path(__file__).resolve().parents[1]
    result=dict(run_id=rid,status='running',
                ocsd_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                python=platform.python_version(),libraries={},scope='CPU real-weight compatibility; not an image-quality experiment',
                code_sha256={p:file_hash(root/p) for p in ('ocsd/direct_instances.py','tools/preflight_direct.py','requirements-direct-preflight.txt')},
                setup_returncode=setup_returncode)
    for name in ('torch','torchvision','transformers','huggingface_hub','timm','einops','numpy','omegaconf'):
        try: result['libraries'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['libraries'][name]=None
    start=time.perf_counter()
    with (directory/'preflight.log').open('w') as log, redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
        try:
            if setup_returncode: raise RuntimeError('Dependency installation failed; see setup.log in this ZIP')
            import torch
            from ocsd.direct_instances import SOURCE_COMMIT
            result['source_commit']=SOURCE_COMMIT
            torch.set_num_threads(4)
            if torch.cuda.is_available(): raise RuntimeError('Preflight must run without CUDA')
            print('CPU preflight: downloads official public weights; no image generation.',flush=True)
            model=load_components(cache)
            result['model_provenance']=model.provenance
            with torch.no_grad(): result['probe']=probe(model)
            result['status']='pass'
            print('PASS: CPU compatibility. GPU precision, image quality and naturalness remain untested.',flush=True)
        except Exception as exc:
            result['status']='failed';result['error']=str(exc)
            traceback.print_exc()
            print('Preflight failed. Keep this log; do not start a GPU quality run.',flush=True)
        result['wall_s']=time.perf_counter()-start
        atomic_json(directory/'preflight.json',result)
    archive=out/f'ocsd_direct_preflight_{rid}.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for name in ('preflight.json','preflight.log'):z.write(directory/name,name)
        if setup_log and Path(setup_log).is_file():z.write(setup_log,'setup.log')
    atomic_json(out/'latest_preflight.json',dict(run_id=rid,status=result['status'],archive=str(archive)))
    print('Saved CPU preflight:',archive,flush=True)
    return result,archive


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--setup-log');p.add_argument('--setup-returncode',type=int,default=0);a=p.parse_args()
    result,_=run(a.cache,a.out,a.setup_log,a.setup_returncode)
    sys.exit(0 if result['status']=='pass' else 1)
