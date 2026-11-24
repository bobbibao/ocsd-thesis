"""CPU test of ocsd.bg_swap with the tiny SD 1.5-shaped stack: one image per background, scene restored afterwards,
and the first (own) background gives the same image as a plain OCSD run of the same steps.  python tests/test_bg_swap.py"""
import json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import cv2
import numpy as np
import open_clip
from tiny import FakeVision, make_engine
from ocsd import bg_swap, method, trace_ocsd
from ocsd.data import load_scene
from ocsd.ocsd_xl import upscale_scene

W = "/tmp/ocsd_bgswap_test"
shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
BPE = os.path.join(os.path.dirname(open_clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
S = 128
H3 = os.path.join(ROOT, "assets/heldout_v3/quickdraw")
sdir = next(os.path.join(H3, d) for d in sorted(os.listdir(H3))
            if json.load(open(os.path.join(H3, d, "scene.json")))["n_obj"] == 3)
eng = make_engine(os.path.join(W, "tok"), BPE, S)
vis = FakeVision()
tiny = dict(height=S, width=S, steps=6, obj_steps=3, K=2, S1=4, S2=4, lora_rank=4, tau=2, R=1)
meta = bg_swap.run(W, sid="t", seed=0, bgs=("on a snowy field", "on a sandy beach"), eng=eng, vis=vis, out_root=W,
                   scene_dir=sdir, overrides=tiny, size=S)
own = json.load(open(os.path.join(sdir, "scene.json")))["bg"]
assert [r["bg"] for r in meta["runs"]][0] == own and len(meta["runs"]) == 3, meta["runs"]
for r in meta["runs"]:
    assert r["bg"] in r["caption"] and os.path.exists(r["file"]), r
assert os.path.exists(meta["zip"]) and os.path.exists(os.path.join(W, "bgswap_t_s0", "SHEET.jpg"))
ims = [cv2.imread(r["file"]).astype(float) for r in meta["runs"]]
print("mean diff own vs snowy:", np.abs(ims[0] - ims[1]).mean())
# reference: plain OCSD run (same order as ocsd.runner) with the own background
scene = upscale_scene(load_scene(sdir), S)
cfg = trace_ocsd.ocsd_config(tiny)
objs = method.object_branch(eng, vis, scene, cfg, seed=0)
fg = method.compose_foreground(scene, objs, cfg)
ids = method.learn_identity(eng, scene, objs, cfg, seed=0, log={})
ref = method.generate(eng, vis, scene, method.Prepared(objs, fg, ids, {}), cfg, 0, log={})
eng.reset_identity()
d = np.abs(cv2.cvtColor(ref, cv2.COLOR_RGB2BGR).astype(float) - ims[0]).mean()
print("own background vs plain run, mean abs diff:", d)
assert d < 2.0, d
print("OK")
