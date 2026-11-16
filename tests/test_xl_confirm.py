"""CPU test of the OCSD-XL confirmation path with tiny random models: the locked protocol loads, legacy images are
imported, missing ones fall back to the SD 1.5 runner, OCSD-v2 is generated, every arm is scored and reviewed, and the
pre-registered analysis runs. python tests/test_xl_confirm.py"""
import json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests")); sys.path.insert(0, os.path.join(ROOT, "tools"))
import cv2
import numpy as np
import open_clip
from tiny import FakeVision, make_engine, make_engine_xl
from ocsd import ocsd_xl, xl_study

W = os.environ.get("XLC_DIR", "/tmp/ocsd_xl_confirm_test")
BPE = os.path.join(os.path.dirname(open_clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
S = 128

lock = xl_study.load_protocol("xl_confirm_v1")           # digest of the SDXL arms and of the SD 1.5 settings
assert len(lock["scenes"]) == 18 and lock["reference_arm"] == "ocsd" and len(xl_study.all_arms(lock)) == 11
assert ocsd_xl.arm_config("ocsd_xl_natural")[1].omega == 0.4 and ocsd_xl.arm_config("ocsd_xl_natural")[1].use_energy
c = ocsd_xl.arm_config("ocsd_xl_faithful")[1]
assert c.omega == 0.8 and not c.use_energy and c.eta == 0.02 and not c.use_identity
_, variants, _, baselines = xl_study.sd15_configs(lock)
assert variants["ocsd_v2"].blend_mode == "joint" and variants["ocsd"].alpha == 0.1 and baselines["gligen"][1].gligen_beta == 1.0
print("lock OK")

ocsd_xl.XL_COMMON.update(height=S, width=S, steps=4, obj_steps=3, K=1)
ocsd_xl.XL_OCSD.update(tau=2)
sids = lock["scenes"][:2]
p = dict(lock, name="test", drive_root="xl_confirm_test", scenes=sids, models={}, size=S,
         imported_arms={"ocsd": "old/quickdraw/ocsd", "controlnet": "old/quickdraw/controlnet",
                        "ocsd_v3": "old/quickdraw/ocsd_v3"}, sd15_fallback=["ocsd", "controlnet"])
rng = np.random.default_rng(0)
# 64 px copies of the two scenes: the tiny SD 1.5 UNet is too slow on CPU at 512 px
from ocsd.data import load_scene
T = 64
for sid in sids:
    sc, d = load_scene(os.path.join(ROOT, lock["scene_assets"], "quickdraw", sid)), os.path.join(W, "small/quickdraw", sid)
    os.makedirs(d)
    small = lambda a: np.where(cv2.resize(a, (T, T), interpolation=cv2.INTER_AREA) < 235, 0, 255).astype(np.uint8)
    meta = {k: v for k, v in sc.meta.items() if k != "dir"}
    for o in meta["objects"]: o["box"] = [v * T / 512 for v in o["box"]]
    json.dump(meta, open(os.path.join(d, "scene.json"), "w"))
    cv2.imwrite(os.path.join(d, "sketch.png"), small(sc.sketch))
    np.savez(os.path.join(d, "masks.npz"), masks=np.stack([cv2.resize(o.mask.astype(np.uint8), (T, T), interpolation=cv2.INTER_NEAREST) > 0 for o in sc.objects]))
    for i, o in enumerate(sc.objects): cv2.imwrite(os.path.join(d, f"obj_{i:02d}.png"), small(o.sketch))
xl_study.scene_dirs = lambda q: [os.path.join(W, "small/quickdraw", sid) for sid in q["scenes"]]
for arm in ("ocsd", "ocsd_v3"):                          # "controlnet" is absent on purpose: it must be regenerated
    os.makedirs(os.path.join(W, "old/quickdraw", arm))
    for sid in sids:
        for seed in (0, 1):
            if (arm, sid, seed) == ("ocsd_v3", sids[1], 1): continue       # cannot be regenerated: stays missing
            f = os.path.join(W, "old/quickdraw", arm, f"{sid}_s{seed}")
            cv2.imwrite(f + ".png", rng.integers(0, 255, (T, T, 3), dtype=np.uint8))
            json.dump(dict(method=arm, scene_time=1.5), open(f + ".json", "w"))
tiny = dict(steps=3, obj_steps=2, K=1, S1=2, S2=2, tau=1, height=T, width=T)
real = xl_study.sd15_configs
xl_study.sd15_configs = lambda q: (lambda E, v, b, bl: (E, {k: x.replace(**tiny) for k, x in v.items()}, b.replace(**tiny),
                                                        {k: (kind, x.replace(**tiny)) for k, (kind, x) in bl.items()}))(*real(q))
vis = FakeVision()
missing = xl_study.import_legacy(W, p)
assert len(missing) == 5 and {m[0] for m in missing} == {"controlnet", "ocsd_v3"}, missing
xl_study.generate(W, p, eng=make_engine_xl(os.path.join(W, "tok_xl"), BPE, S), vis=vis)
eng15 = make_engine(os.path.join(W, "tok"), BPE, T)
xl_study.generate_sd15(W, p, missing, eng=eng15, vis=vis)
dirs = xl_study.Dirs(W, p)
have = {a: sum(dirs.png(a, s, k).exists() for s in sids for k in (0, 1)) for a in xl_study.all_arms(p)}
assert have == dict(xl_controlnet=4, xl_collage=4, ocsd_xl_natural=4, ocsd_xl_faithful=4, ocsd_v2=4, ocsd=4,
                    controlnet=4, ocsd_v3=3), have
calls = dict(eng15.compute)
assert xl_study.import_legacy(W, p) == [("ocsd_v3", sids[1], 1)]
xl_study.generate_sd15(W, p, [("ocsd_v3", sids[1], 1)], eng=eng15, vis=vis)      # resume: nothing to do
assert eng15.compute == calls
print("import + generation OK:", have)


class FakePick:
    def score(self, img, text): return float(img.mean() / 10)


xl_study.evaluate(W, p, vis=vis, picker=FakePick())
out = xl_study.report(W, p)
text = open(out["summary"]).read()
assert all(a in text for a in xl_study.all_arms(p)) and "against ocsd" in text
key = json.load(open(dirs.results / "review_key.json"))
assert len(key) == 4 and sorted(len(v) for v in key.values()) == [7, 8, 8, 8]
import analyze_xl_confirm as A
ratings = {s: {k: (i + len(s)) % 3 for i, k in enumerate(m)} for s, m in key.items()}
json.dump(ratings, open(os.path.join(W, "ratings.json"), "w"))
A.ROOT = type(A.ROOT)(ROOT)
res = A.main(dirs.results, os.path.join(W, "ratings.json"))
assert [r["id"] for r in res["primary"]] == ["P1", "P2", "P3", "P4"] and all(0 <= r["p"] <= r["p_holm"] <= 1 for r in res["primary"])
assert set(res["claims"]) == {"C-faithful", "C-natural"} and "naturalness" in res["means"]["ocsd"]
assert os.path.exists(dirs.results / "analysis.md")
print("evaluate + report + analysis OK")
print("ALL XL CONFIRM TESTS PASSED")
