"""CPU test of ocsd.trace_xl with the tiny SDXL-shaped stack: one scene, every module's intermediates are written,
the traced run gives the same image as an untraced run_arm, and the wrappers are removed afterwards.
python tests/test_trace_xl.py"""
import json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import numpy as np
import open_clip
from tiny import FakeVision, make_engine_xl
from ocsd import method, ocsd_xl, trace_xl
from ocsd.data import load_scene

W = "/tmp/ocsd_trace_test"
shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
BPE = os.path.join(os.path.dirname(open_clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
S = 128
H3 = os.path.join(ROOT, "assets/heldout_v3/quickdraw")
sdir = next(os.path.join(H3, d) for d in sorted(os.listdir(H3))
            if json.load(open(os.path.join(H3, d, "scene.json")))["n_obj"] == 3)
eng = make_engine_xl(os.path.join(W, "tok"), BPE, S)
vis = FakeVision()
tiny = dict(height=S, width=S, steps=6, obj_steps=3, K=2)
trace_xl.ATTN_STEPS = (0, 2, 5)
orig = (method.sample, method.build_scene, method.repair_scene, method.scene_cond)

meta = trace_xl.trace(W, sid="t", seed=0, eng=eng, vis=vis, out_root=W, scene_dir=sdir, size=S, overrides=tiny)
assert (method.sample, method.build_scene, method.repair_scene, method.scene_cond) == orig
assert "cfg_eps" not in eng.__dict__ and "energy_update" not in eng.__dict__
out = os.path.join(W, "t_s0")
assert meta["round"] == "c1" and meta["cfg"]["omega"] == 0.4 and meta["cfg"]["use_energy"] is True
assert len(meta["m1"]["objects"]) == 3
assert all(len(o["candidates"]) == 2 and sum(c["selected"] for c in o["candidates"]) == 1 for o in meta["m2"])
traj = [r for r in meta["m5"]["trajectory"] if r["phase"] == "scene"]
assert [r["step"] for r in traj] == list(range(6)), traj
assert sum(r["blend"] for r in traj) == meta["m4"]["blend_steps"] > 0
assert all(len(r["attn"]) == 3 for r in traj if r["step"] in (0, 2, 5))
assert meta["m5"]["energy"] and all(e["phase"] == "scene" for e in meta["m5"]["energy"])
assert meta["m5d"]["checks"], "verification not recorded"
for f in ("OVERVIEW.jpg", "NOISE_TO_IMAGE.jpg", "trace.json", "final_1024.jpg", "M4/composed_foreground.jpg"):
    assert os.path.exists(os.path.join(out, f)), f
assert os.path.exists(meta["zip"])

# the traced run produces the same image as a plain run of the same arm
import cv2
traced = cv2.cvtColor(cv2.imread(os.path.join(out, "final_512.png")), cv2.COLOR_BGR2RGB).astype(np.float32)
scene = ocsd_xl.upscale_scene(load_scene(sdir), S)
vis.set_scene(scene)
objs = ocsd_xl.make_objects(eng, vis, scene, 0, {}, ocsd_xl.arm_m2(trace_xl.ARM))
plain = cv2.resize(ocsd_xl.run_arm(trace_xl.ARM, eng, vis, scene, 0, objs, {}), (512, 512), interpolation=cv2.INTER_NEAREST).astype(np.float32)
d = float(np.abs(traced - plain).mean())
assert d == 0.0, d
print("trace OK:", len(traj), "steps,", len(meta["m5d"]["checks"]), "checks, mean |traced - plain| =", round(d, 2))

# a failed check: the repair (M5d) steps are traced too
real = vis.gdino
vis.gdino = lambda img, classes, thr=0.3, **k: real(img, classes, thr, **k)[1:] if thr >= 0.35 else real(img, classes, thr, **k)
meta = trace_xl.trace(W, sid="r", seed=0, eng=eng, vis=vis, out_root=W, scene_dir=sdir, size=S, overrides=tiny)
rep = [r for r in meta["m5"]["trajectory"] if r["phase"].startswith("repair")]
assert rep and len(meta["m5d"]["checks"]) >= 2 and not meta["m5d"]["checks"][0]["passed"], meta["m5d"]
print("repair traced:", len(rep), "steps,", len(meta["m5d"]["checks"]), "checks")
