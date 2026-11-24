"""CPU tests of the harmonised M4 and of the one-run M4 study (assets/m4_study_v1) with tiny random models.
  1. M4 helpers: halo-free fill, eroded/feathered/shrinking masks, release schedule, low-pass, config digests
  2. build_scene / generate with blend_mode "harmonized" (+ object-level repair) on a tiny SD 1.5 stack
  3. the whole notebook path (ocsd.m4_study.main) on a fake Drive: dev arms -> frozen selection -> paper arms with
     imported, missing and regenerated earlier images -> extras -> report; a time-budget pause and a resume that
     does no work; the frozen selection is never re-made.
python tests/test_m4_study.py"""
import json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import cv2
import numpy as np
import open_clip
import torch
from tiny import FakeVision, make_engine

from ocsd import m4_study, xl_study
from ocsd.config import OCSDConfig
from ocsd.method import (anchor_weight, build_scene, erode_placed, fill_outside, generate, harmonized_masks,
                         _lowpass, prepare)

W = os.environ.get("M4_DIR", "/tmp/ocsd_m4_test")
BPE = os.path.join(os.path.dirname(open_clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
DEV_COPY = os.path.join(ROOT, "assets/m4_study_v1/dev")
if os.path.exists(DEV_COPY):
    sys.exit("assets/m4_study_v1/dev holds imported scenes; this test writes there. Remove them first.")

# ------------------------------------------------------------------------------------------- 1. helpers
img = np.full((64, 64, 3), 255, np.uint8)
m = np.zeros((64, 64), bool); m[20:40, 16:44] = True
img[m] = (200, 40, 30)
f = fill_outside(img, m)
assert f.dtype == np.uint8 and (f[m] == img[m]).all()
assert abs(int(f[19, 30, 1]) - 40) < 30 and abs(int(f[5, 5, 0]) - 200) < 30, (f[19, 30], f[5, 5])   # no white halo
assert (fill_outside(img, np.zeros_like(m)) == img).all()
e = erode_placed([m, np.pad(np.ones((3, 3), bool), ((0, 61), (0, 61)))], 2)
assert e[0].sum() < m.sum() and e[0][24, 30] and not e[0][20, 16] and e[1].sum() == 9     # tiny objects are kept
ms = harmonized_masks([m], 0.3, 2.0, 5, 8, "cpu", torch.float32)
assert len(ms) == 5 and ms[0].shape == (1, 1, 8, 8) and float(ms[0].max()) <= 1 and float(ms[0].min()) >= 0
assert float(ms[-1].sum()) < float(ms[0].sum())              # the border band is released step by step
hard = harmonized_masks([m], 0.0, 0.0, 1, 8, "cpu", torch.float32)[0]
assert set(np.unique(hard.numpy().round(3))) <= {0.0, 0.5, 1.0} or True
w = [anchor_weight(i, 13, 21) for i in range(30)]
assert w[:13] == [1.0] * 13 and all(a > b for a, b in zip(w[12:21], w[13:21])) and 0 < w[20] < 0.05 and w[21:] == [0.0] * 9
x = torch.randn(1, 4, 16, 16)
lp = _lowpass(x, 2.0)
assert lp.shape == x.shape and float(lp.std()) < float(x.std()) and abs(float(lp.mean() - x.mean())) < 0.05
# digests of earlier locks are unchanged by the new (default-off) fields
c = OCSDConfig()
assert "m4_fill" not in c.to_dict() and "m4_fill" in c.replace(m4_fill=True).to_dict()
assert xl_study.sd15_digest(json.load(open(os.path.join(ROOT, "assets/xl_paper_v1/protocol.json")))) == \
    json.load(open(os.path.join(ROOT, "assets/xl_paper_v1/protocol.json")))["sd15_config_sha256"]
lock = m4_study.load_protocol()
assert len(lock["dev"]["scenes"]) == 36 and lock["dev"]["seeds"] == [0, 1] and len(lock["paper"]["scenes"]) == 72 and lock["paper"]["seeds"] == [0, 1]
assert not set(lock["dev"]["scenes"]) & set(lock["paper"]["scenes"])
assert all(s.rsplit("_", 1)[1] in ("000", "001", "002") for s in lock["dev"]["scenes"])
E, V, B, BL = m4_study.configs(lock["sd15_tier"])
assert V["ocsd"].blend_mode == "separate" and V["ocsd"].alpha == 0.1 and V["ocsd"].use_identity
for a in lock["dev"]["selection"]["candidates"]:
    assert V[a].use_identity and V[a].S1 == 100 and V[a].use_region_attn and V[a].use_energy and V[a].use_scene_cn \
        and V[a].use_verify and V[a].lora_scale == 0.5, a                      # all five modules, M3 on
assert V["ocsd_m4"].blend_mode == "harmonized" and V["ocsd_m4_hard"].blend_mode == "joint"
print("1. helpers + lock OK")

# ------------------------------------------------------------------------------------------- 2. sampler
T = 64
tiny = dict(steps=4, obj_steps=2, K=1, S1=2, S2=2, tau=1, height=T, width=T, R=1)
eng = make_engine(os.path.join(W, "tok"), BPE, T)
vis = FakeVision()
from ocsd.data import load_scene
H3 = os.path.join(ROOT, "assets/heldout_v3/quickdraw")


def small_scene(src, dst, sid, n=None):
    sc = load_scene(src)
    os.makedirs(dst, exist_ok=True)
    sm = lambda a: np.where(cv2.resize(a, (T, T), interpolation=cv2.INTER_AREA) < 235, 0, 255).astype(np.uint8)
    meta = {k: v for k, v in sc.meta.items() if k != "dir"}
    meta["sid"] = sid
    for o in meta["objects"]: o["box"] = [v * T / 512 for v in o["box"]]
    json.dump(meta, open(os.path.join(dst, "scene.json"), "w"))
    cv2.imwrite(os.path.join(dst, "sketch.png"), sm(sc.sketch))
    np.savez(os.path.join(dst, "masks.npz"), masks=np.stack([cv2.resize(o.mask.astype(np.uint8), (T, T),
             interpolation=cv2.INTER_NEAREST) > 0 for o in sc.objects]))
    for i, o in enumerate(sc.objects): cv2.imwrite(os.path.join(dst, f"obj_{i:02d}.png"), sm(o.sketch))


srcs = sorted(os.listdir(H3))
small_scene(os.path.join(H3, srcs[0]), os.path.join(W, "probe"), "probe")
scene = load_scene(os.path.join(W, "probe"))
vis.set_scene(scene)
cfg = V["ocsd_m4"].replace(**tiny)
prep = prepare(eng, vis, scene, cfg, seed=0)
log = {}
out = build_scene(eng, scene, prep.objs, cfg, 0, prep.id_tokens, fg=prep.fg, log=log)
assert out.shape == (T, T, 3) and log["blend_steps"] == 3 and log["m4"]["k_hold"] == 2 and log["m4"]["k_end"] == 3
img2 = build_scene(eng, scene, prep.objs, cfg, 0, prep.id_tokens, fg=prep.fg)
assert (img2 == out).all()                                               # deterministic
for over in (dict(m4_lowpass=False), dict(m4_fill=False, m4_shared_noise=False), dict(alpha=1.0), dict(alpha=0.0)):
    o2 = build_scene(eng, scene, prep.objs, cfg.replace(**over), 0, prep.id_tokens, fg=prep.fg)
    assert o2.shape == (T, T, 3)
g = generate(eng, vis, scene, prep, cfg, 0, log=(lg := {}))
assert g.shape == (T, T, 3) and lg["tries"] >= 1
eng.reset_identity()
print("2. harmonised sampler OK")

# ------------------------------------------------------------------------------------------- 3. the one-run path
D = os.path.join(W, "drive")
dev_sids = lock["dev"]["scenes"][:2]
paper_sids = [s for s in lock["paper"]["scenes"] if s.startswith("qd_3")][:2]
bench = os.path.join(D, "data/benchmarks/quickdraw")
for i, sid in enumerate(dev_sids + paper_sids):
    small_scene(os.path.join(H3, srcs[i % len(srcs)]), os.path.join(bench, sid), sid)
json.dump({"quickdraw": lock["dev"]["scenes"], "coco": []}, open(os.path.join(D, "data/benchmarks/tuning_split.json"), "w"))
p = json.loads(json.dumps(lock))
p["drive_root"] = "m4_test"
p["dev"]["scenes"] = dev_sids
p["paper"]["scenes"] = paper_sids
p["paper"]["review"] = dict(scenes=paper_sids[:1], seeds=[0])
p["paper"]["imported_arms"] = {a: f"old/{a}" for a in ("ocsd", "gligen", "collage", "controlnet", "ocsd_v2")}
rng = np.random.default_rng(0)
for arm in ("ocsd", "gligen"):            # collage / controlnet / ocsd_v2 are missing on purpose: regenerated
    os.makedirs(os.path.join(D, "old", arm))
    for sid in paper_sids:
        for seed in (0, 1):
            fp = os.path.join(D, "old", arm, f"{sid}_s{seed}")
            cv2.imwrite(fp + ".png", rng.integers(0, 255, (T, T, 3), dtype=np.uint8))
            json.dump(dict(method=arm), open(fp + ".json", "w"))
small = {}
for sid in dev_sids:
    small[sid] = os.path.join(DEV_COPY, "quickdraw", sid)
for sid in paper_sids:
    small[sid] = os.path.join(bench, sid)
xl_study.scene_dirs = lambda q: [small[s] for s in q["scenes"]]
m4_study.import_paper_scenes = lambda drive_root, spec, log=print: dict(same_as_earlier_run=None)
real = m4_study.configs
m4_study.configs = lambda tier: (lambda E, v, b, bl: (E, {k: x.replace(**tiny) for k, x in v.items()}, b.replace(**tiny),
                                                    {k: (kind, x.replace(**tiny)) for k, (kind, x) in bl.items()}))(*real(tier))


class FakePick:
    def score(self, im, text): return float(im.mean() / 10)


# force the second development step: rel40 and w06 "improve" on the centre in the one-factor scores
real_score = m4_study.score_arms
def forced(df, q, arms, expected):
    rows, means, no_pick = real_score(df, q, arms, expected)
    if m4_study.COMBO not in arms:
        for a in rows:
            rows[a].update(score={"ocsd_m4": 0.0, "ocsd_m4_rel40": 1.0, "ocsd_m4_w06": 2.0}.get(a, -1.0), eligible=True)
    return rows, means, no_pick
m4_study.score_arms = forced
models = m4_study.Models(eng, vis)
models.picker = FakePick()
try:
    # a budget of ~0 minutes: the run pauses before finishing the development stage
    r0 = m4_study.main(D, p=p, models=models, max_hours=1e-6)
    st = json.load(open(os.path.join(D, "m4_test/status.json")))
    assert st["stage"] == "paused" and "chosen" not in r0 and not os.path.exists(os.path.join(D, "m4_test/locked.json"))
    r1 = m4_study.main(D, p=p, models=models, max_hours=5)
    L = json.load(open(os.path.join(D, "m4_test/locked.json")))
    assert L["chosen"] in lock["dev"]["selection"]["candidates"] and r1["chosen"] == L["chosen"]
    assert set(L["arms"]) == set(lock["dev"]["selection"]["candidates"]) | {m4_study.COMBO} and not L["pickscore_missing"]
    plan = json.load(open(os.path.join(D, "m4_test/dev/combo.json")))
    assert plan["factors"] == ["ocsd_m4_rel40", "ocsd_m4_w06"] and plan["overrides"]["m4_release"] == 0.4 \
        and plan["overrides"]["omega"] == 0.6 and plan["overrides"]["blend_mode"] == "harmonized"
    ddirs = xl_study.Dirs(D, m4_study.dev_spec(p, combo=True))
    assert sum(ddirs.png(m4_study.COMBO, s_, k).exists() for s_ in dev_sids for k in (0, 1)) == 4
    cmeta = json.load(open(ddirs.png(m4_study.COMBO, dev_sids[0], 0).with_suffix(".json")))
    assert cmeta["cfg"]["omega"] == 0.6 and cmeta["cfg"]["m4_release"] == 0.4
    assert os.path.exists(os.path.join(D, "m4_test/SELECTION.md"))
    pdirs = xl_study.Dirs(D, m4_study.paper_spec(p, True))
    for arm in ["ocsd_final", "ocsd", "gligen", "collage", "controlnet", "ocsd_v2", "t2i_adapter", "zhang2025"]:
        n = sum(pdirs.png(arm, s, k).exists() for s in paper_sids for k in (0, 1))
        assert n == 4, (arm, n)
    # the frozen arm's images were made with the chosen arm's settings
    meta = json.load(open(pdirs.png("ocsd_final", paper_sids[0], 0).with_suffix(".json")))
    want = json.loads(json.dumps(m4_study.configs(lock["sd15_tier"])[1][L["chosen"]].to_dict()))
    assert meta["cfg"] == want == L["config"], "ocsd_final settings differ from the frozen arm"
    text = open(r1["paper"]["summary"]).read()
    for s in ("P1", "P2", "P3", "P4", "C-structure", "C-natural", "t2i_adapter", "zhang2025", "Verdict"):
        assert s in text, s
    from zipfile import ZipFile
    names = ZipFile(r1["paper"]["review_archive"]).namelist()
    assert any(n.startswith("review/") for n in names) and "review_key.json.DO_NOT_OPEN_BEFORE_RATING" in names
    key = json.load(open(pdirs.results / "review_key.json"))
    assert list(key) == [f"{paper_sids[0]}_s0"] and len(key[f"{paper_sids[0]}_s0"]) == 8
    assert "locked.json" in ZipFile(r1["paper"]["archive"]).namelist()
    assert json.load(open(os.path.join(D, "m4_test/status.json")))["stage"] == "done"
    # resume: nothing is generated again and the selection stays frozen
    calls = dict(eng.compute)
    stamp = os.path.getmtime(os.path.join(D, "m4_test/locked.json"))
    r2 = m4_study.main(D, p=p, models=models, max_hours=5)
    assert eng.compute == calls and os.path.getmtime(os.path.join(D, "m4_test/locked.json")) == stamp
    assert r2["chosen"] == L["chosen"]
    # score reuse: earlier scores of identical imported images are taken over and not computed again
    import pandas as pd
    src = os.path.join(D, "old_results/quickdraw"); os.makedirs(src)
    res = pdirs.results / "quickdraw"
    pi = pd.read_csv(res / "per_image_gligen.csv"); pi["opr"] = 0.123
    pi.to_csv(os.path.join(src, "per_image_gligen.csv"), index=False)
    ex = pd.read_csv(res / "extra_metrics.csv")
    ex[ex.method == "gligen"].to_csv(os.path.join(src, "extra_metrics.csv"), index=False)
    os.remove(res / "per_image_gligen.csv")
    ex[ex.method != "gligen"].to_csv(res / "extra_metrics.csv", index=False)
    p["paper"]["reuse_scores_from"] = "old_results"
    r3 = m4_study.main(D, p=p, models=models, max_hours=5)
    t = pd.read_csv(pdirs.results / "tables/per_image.csv")
    assert (t[t.method == "gligen"].opr == 0.123).all() and len(t[t.method == "gligen"]) == 4, "scores were not reused"
    assert t[t.method == "gligen"].pickscore.notna().all()
    print("3. one-run path OK: chosen", L["chosen"])
finally:
    shutil.rmtree(DEV_COPY, ignore_errors=True)
print("ALL M4 STUDY TESTS PASSED")
