"""CPU-only foreground audit on saved, manifest-verified proposals; never regenerates scene images."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw
from ocsd.foreground import predict_foreground,MODEL_SHA256


def audit(bundle,out,cache,method='instance_repair_count',item='qd_3_simple_000_s0'):
    root=Path(bundle).resolve();manifest=json.loads((root/'run_manifest.json').read_text());rid=manifest['run_id']
    prefix=f'outputs/runs/{rid}/quickdraw/{method}/_diagnostics/{item}/'
    def read(name):
        rel=prefix+name;p=(root/rel).resolve()
        if not p.is_relative_to(root):raise ValueError('Invalid input path')
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest['artifacts'].get(rel):raise ValueError('Changed input artifact')
        return np.array(Image.open(p).convert('RGB')),hashlib.sha256(raw).hexdigest()
    image,source_hash=read('proposal.png');mask_rgb,mask_hash=read('proposal_mask.png');mask=mask_rgb[...,0]>0
    if not mask.any() or mask.all():raise ValueError('Audit needs foreground and background')
    labels=np.where(mask,cv2.GC_PR_FGD,cv2.GC_BGD).astype(np.uint8)
    cv2.setRNGSeed(0);cv2.grabCut(image,labels,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
    grab=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD])
    log={};probability=predict_foreground(image,cache,log);clipped=probability*mask
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    variants=[('Original SAM mask',mask.astype(np.float32)),('GrabCut (5 iterations)',grab.astype(np.float32)),('BiRefNet foreground probability',probability),('BiRefNet restricted to original mask',clipped)]
    h,w=image.shape[:2];yy,xx=np.indices((h,w));checker=np.repeat(np.where((xx//20+yy//20)%2,210,250).astype(np.uint8)[...,None],3,axis=2)
    canvas=Image.new('RGB',(w*2,(h+30)*2),'white');draw=ImageDraw.Draw(canvas)
    for idx,(label,alpha) in enumerate(variants):
        result=np.rint(image*alpha[...,None]+checker*(1-alpha[...,None])).clip(0,255).astype(np.uint8)
        x=idx%2*w;y=idx//2*(h+30);draw.text((x+5,y+5),label,fill='black');canvas.paste(Image.fromarray(result),(x,y+25))
    canvas.save(out/'comparison.jpg',quality=92)
    Image.fromarray((clipped*255).round().astype(np.uint8)).save(out/'foreground_probability.png')
    result=dict(source_run=rid,item=item,source_method=method,source_image_sha256=source_hash,source_mask_sha256=mask_hash,
                model_sha256=MODEL_SHA256,model=log['foreground_model'],opencv=cv2.__version__,
                grabcut_retained_area_fraction=float(grab.sum()/mask.sum()),learned_probability_mass_fraction=float(clipped.sum()/mask.sum()),
                quality_metrics='none; area/mass fractions are not accuracy or preservation scores',
                inference_scope='one saved proposal, CPU only; no new scene diffusion')
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle-root',required=True);p.add_argument('--out',required=True);p.add_argument('--model-cache',required=True)
    p.add_argument('--method',default='instance_repair_count');p.add_argument('--item',default='qd_3_simple_000_s0');a=p.parse_args()
    print(json.dumps(audit(a.bundle_root,a.out,a.model_cache,a.method,a.item),indent=2))
