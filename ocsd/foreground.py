"""Experimental learned foreground extraction; probability is NOT a calibrated alpha matte.

BiRefNet-general-lite ONNX distributed by rembg; inference stays on the local CPU.
Pre/postprocessing follows rembg's BiRefNetSessionGeneral implementation.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import time
import urllib.request
import cv2
import numpy as np
from PIL import Image

MODEL_URL='https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx'
MODEL_SHA256='5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333'
METHODS=('instance_repair_count_clean','instance_repair_count_context','instance_repair_count_matte_hard','instance_repair_count_matte_soft')
CONTEXT_METHODS=METHODS[1:]
REFINED_METHODS=METHODS[2:]


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def model_path(cache_dir):
    path=Path(cache_dir or '/tmp/ocsd-model-cache')/'foreground/birefnet-general-lite.onnx'
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        import tempfile,os
        fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.download')
        try:
            with os.fdopen(fd,'wb') as out, urllib.request.urlopen(MODEL_URL,timeout=60) as src:
                for block in iter(lambda:src.read(2**20),b''):out.write(block)
            if _sha(tmp)!=MODEL_SHA256:raise ValueError('Foreground model checksum mismatch')
            Path(tmp).replace(path)
        finally:
            Path(tmp).unlink(missing_ok=True)
    if _sha(path)!=MODEL_SHA256:raise ValueError('Foreground model checksum mismatch')
    return path


class ForegroundRefiner:
    def __init__(self, cache_dir):
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model_path(cache_dir)),sess_options=options,providers=['CPUExecutionProvider'])
        self.version=ort.__version__
        self.cache={}

    def predict(self,image):
        key=hashlib.sha256(image.tobytes()+str(image.shape).encode()).hexdigest()
        if key in self.cache:
            probability,elapsed=self.cache[key]
            return probability.copy(),elapsed,True
        start=time.perf_counter()
        x=np.array(Image.fromarray(image).convert('RGB').resize((1024,1024),Image.Resampling.LANCZOS)).astype(np.float32)
        x=x/max(float(x.max()),1e-6)
        x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        logits=self.session.run(None,{self.session.get_inputs()[0].name:x.transpose(2,0,1)[None]})[0][0,0]
        p=1/(1+np.exp(-np.clip(logits,-60,60)))
        if not np.isfinite(p).all() or float(p.max()-p.min())<1e-6:raise ValueError('Degenerate foreground prediction')
        p=(p-p.min())/(p.max()-p.min())
        probability=np.asarray(Image.fromarray((p*255).astype(np.uint8)).resize((image.shape[1],image.shape[0]),Image.Resampling.LANCZOS)).astype(np.float32)/255
        elapsed=time.perf_counter()-start;self.cache[key]=(probability,elapsed)
        return probability.copy(),elapsed,False


@lru_cache(maxsize=2)
def _refiner(cache_dir):return ForegroundRefiner(cache_dir)


def predict_foreground(image,cache_dir,log):
    start=time.perf_counter();model=_refiner(str(cache_dir) if cache_dir else None)
    probability,seconds,reused=model.predict(image)
    log['foreground_model']=dict(url=MODEL_URL,sha256=MODEL_SHA256,onnxruntime=model.version,provider='CPUExecutionProvider',
                                actual_wall_s=time.perf_counter()-start,standalone_forward_s=seconds,forward_calls_in_this_method=0 if reused else 1)
    # Charge a reused forward, without pretending warmup/downloads are a latency benchmark.
    log['foreground_reused_seconds']=seconds if reused else 0.
    return probability


def place_foreground(image, probability, original_mask, info, placement, size, placed_original_mask, hard=False):
    """Freeze original placement and footprint; resize premultiplied color to avoid background bleed."""
    probability=np.asarray(probability,dtype=np.float32)
    if probability.shape!=original_mask.shape or probability.shape!=image.shape[:2] or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
        raise ValueError('Invalid foreground probability')
    alpha=(probability>=.5).astype(np.float32) if hard else probability.copy()
    alpha*=original_mask.astype(bool)
    scale,dx,dy=placement
    side=max(int(round(info.side*scale)),4)
    a=cv2.resize(alpha,(side,side),interpolation=cv2.INTER_AREA)
    premul=cv2.resize(image.astype(np.float32)*alpha[...,None],(side,side),interpolation=cv2.INTER_AREA)
    rgb=np.divide(premul,a[...,None],out=np.zeros_like(premul),where=a[...,None]>1e-8)
    cx,cy=info.x0+info.side/2+dx,info.y0+info.side/2+dy
    sx,sy=int(round(cx-side/2)),int(round(cy-side/2))
    x0,y0=max(sx,0),max(sy,0);x1,y1=min(sx+side,size),min(sy+side,size)
    canvas=np.zeros((size,size,3),np.uint8);placed=np.zeros((size,size),np.float32)
    if x1>x0 and y1>y0:
        canvas[y0:y1,x0:x1]=rgb[y0-sy:y1-sy,x0-sx:x1-sx].round().clip(0,255).astype(np.uint8)
        placed[y0:y1,x0:x1]=a[y0-sy:y1-sy,x0-sx:x1-sx]
    placed*=placed_original_mask.astype(bool)
    return canvas,placed
