"""CPU test of ocsd.trace_ocsd with the tiny SD 1.5-shaped stack: one scene, every module's intermediates (M1-M5,
M3 on) are written, the traced run gives the same image as an untraced run of the same steps, and the wrappers are
removed afterwards.  python tests/test_trace_ocsd.py"""
import json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import cv2
import numpy as np
import open_clip
from tiny import FakeVision, make_engine
from ocsd import method, trace_ocsd
from ocsd.data import load_scene
from ocsd.ocsd_xl import upscale_scene

W = "/tmp/ocsd_trace15_test"
shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
BPE = os.path.join(os.path.dirname(open_clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
S = 128
H3 = os.path.join(ROOT, "assets/heldout_v3/quickdraw")
sdir = next(os.path.join(H3, d) for d in sorted(os.listdir(H3))
            if json.load(open(os.path.join(H3, d, "scene.json")))["n_obj"] == 3)
eng = make_engine(os.path.join(W, "tok"), BPE, S)
vis = FakeVision()
tiny = dict(height=S, width=S, steps=6, obj_steps=3, K=2, S1=4, S2=4, lora_rank=4, tau=2, R=1)
trace_ocsd.ATTN_STEPS = (0, 2, 5)
names = ("sample", "build_scene", "repair_scene", "scene_cond", "anchor_masks", "compose_foreground")
orig = [getattr(method, k) for k in names]

meta = trace_ocsd.trace(W, sid="t", seed=0, eng=eng, vis=vis, out_root=W, scene_dir=sdir, overrides=tiny, size=S)
assert [getattr(method, k) for k in names] == orig
assert not {"cfg_eps", "energy_update", "q_sample"} & set(eng.__dict__)
out = os.path.join(W, "t_s0")
c = meta["cfg"]
assert c["use_identity"] and c["blend_mode"] == "separate" and c["alpha"] == 0.1 and c["lora_scale"] == 0.5, c
assert len(meta["m1"]["objects"]) == 3
assert all(len(o["candidates"]) == 2 and sum(x["selected"] for x in o["candidates"]) == 1 for o in meta["m2"])
m3 = meta["m3"]
assert m3["used"] and len(m3["losses"]) > 0 and len(m3["check"]) == 3, m3
traj = [r for r in meta["m5"]["trajectory"] if r["phase"] == "scene"]
assert [r["step"] for r in traj] == list(range(6)), traj
k = meta["m4"]["blend_steps"]
assert k == round(0.9 * 6) and sum(r["blend"] for r in traj) == k, (k, [r["blend"] for r in traj])
assert all(len(r["attn"]) == 3 for r in traj if r["step"] in (0, 2, 5))
assert meta["m5d"]["checks"], "verification not recorded"
for f in ("OVERVIEW.jpg", "NOISE_TO_IMAGE.jpg", "trace.json", "final.png", "M4/composed_foreground.jpg",
          "M3/training_loss.png"):
    assert os.path.exists(os.path.join(out, f)), f
assert os.path.exists(meta["zip"])

# the traced run produces the same image as a plain run (same order as ocsd.runner)
traced = cv2.cvtColor(cv2.imread(os.path.join(out, "final.png")), cv2.COLOR_BGR2RGB).astype(np.float32)
cfg = trace_ocsd.ocsd_config(tiny)
scene = upscale_scene(load_scene(sdir), S)
vis.set_scene(scene)
objs = method.object_branch(eng, vis, scene, cfg, seed=0)
fg = method.compose_foreground(scene, objs, cfg)
ids = method.learn_identity(eng, scene, objs, cfg, seed=0)
plain = method.generate(eng, vis, scene, method.Prepared(objs, fg, ids, {}), cfg, 0).astype(np.float32)
eng.reset_identity()
d = float(np.abs(traced - plain).mean())
assert d == 0.0, d
print("trace OK:", len(traj), "steps,", k, "blend steps, M3", len(m3["losses"]), "loss values,",
      len(meta["m5d"]["checks"]), "checks, mean |traced - plain| =", d)

# a failed check: the retry (M5d) steps are traced too
real = vis.gdino
vis.gdino = lambda img, classes, thr=0.3, **kw: real(img, classes, thr, **kw)[1:]
meta = trace_ocsd.trace(W, sid="r", seed=0, eng=eng, vis=vis, out_root=W, scene_dir=sdir, overrides=tiny, size=S)
rt = [r for r in meta["m5"]["trajectory"] if r["phase"].startswith(("retry", "repair"))]
assert rt and len(meta["m5d"]["checks"]) >= 2 and not meta["m5d"]["checks"][0]["passed"], meta["m5d"]
print("retry traced:", len(rt), "steps,", len(meta["m5d"]["checks"]), "checks, selected", meta["m5d"]["selected_try"])
