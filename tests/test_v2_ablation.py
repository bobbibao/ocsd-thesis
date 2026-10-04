"""CPU end-to-end test of the 10 OCSD-v2 ablation rows (E4v2) on a 3-object and a 5-object QuickDraw scene with tiny
random models: config resolution through stages.tier_variants (every row differs from OCSD-v2 in exactly its own
fields and survives the plan's dedup rule), generation through the Runner, per-row evidence in the logs and images
that the ablated component is really off, evaluation, and the E4v2 report table. python tests/test_v2_ablation.py"""
import json, os, shutil, sys
from dataclasses import asdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import cv2
import numpy as np
import open_clip
import pandas as pd
from tiny import FakeVision, make_engine
from ocsd import config, report as R, stages
from ocsd.config import ExperimentConfig
from ocsd.data import build_quickdraw_scenes, download_quickdraw, list_scenes, load_scene
from ocsd.method import attn_masks, compose_foreground, scene_cond
from ocsd.methods import V2_ABLATIONS
from ocsd.metrics import evaluate_images
from ocsd.runner import Runner

W = os.environ.get("V2ABL_DIR", "/tmp/ocsd_v2abl")
BPE = os.path.join(os.path.dirname(open_clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
os.makedirs(W, exist_ok=True)
ROWS = list(V2_ABLATIONS)
assert len(ROWS) == 10 and R.ABLATION_V2 == ["ocsd_v2"] + ROWS, ROWS

# ---------------------------------------------------------------- configs as the paper run resolves them
small = dict(height=128, width=128, steps=6, obj_steps=4, K=2, S1=3, S2=3, tau=2, R=1, lora_rank=4, bo_n=2)
config.TIERS["tiny_v2abl"] = dict(qd_per_cell=1, trained_n=2, coco_n=0, coco_trained_n=0, ablation_n=2, alpha_n=1,
                                  seeds=[0, 1], seeds_all=1, tune=True, cfg=small)
E = ExperimentConfig(tier="tiny_v2abl")
chosen_v2 = dict(alpha=0.4, anchor_shrink=0.3)          # a v2 tuning result inside config.TUNE_V2
var, _ = stages.tier_variants(E, tuned={}, tuned_v2=chosen_v2)
ref = asdict(var["ocsd_v2"])
assert ref["alpha"] == 0.4 and ref["anchor_shrink"] == 0.3 and ref["blend_mode"] == "joint"
for m, over in V2_ABLATIONS.items():
    diff = {f for f, v in asdict(var[m]).items() if v != ref[f]}
    assert diff == {f for f, v in over.items() if v != ref[f]} and diff, (m, diff, over)
# the plan drops rows that tuning made identical to OCSD-v2 (stages.experiment_plan); on the v2 grid that can only
# happen to the anchor-shrink row, when the tuned shrink is 0
for a in config.TUNE_V2["alpha"]:
    for s in config.TUNE_V2["anchor_shrink"]:
        v, _ = stages.tier_variants(E, tuned={}, tuned_v2=dict(alpha=a, anchor_shrink=s))
        kept = [m for m in ROWS if v[m] != v["ocsd_v2"]]
        assert kept == (ROWS if s != 0.0 else [m for m in ROWS if m != "v2abl_no_shrink"]), (a, s, kept)
print("configs OK")

# ---------------------------------------------------------------- one 3-object and one 5-object scene
qd_raw = os.environ.get("QD_RAW", os.path.join(W, "qd_raw"))
download_quickdraw(qd_raw, per_class=40)
bench = os.path.join(W, "bench")
if not os.path.isdir(os.path.join(bench, "quickdraw")):
    build_quickdraw_scenes(qd_raw, os.path.join(bench, "quickdraw"), per_cell=1, size=128)
dirs = list_scenes(bench, "quickdraw", count_bins=["3"])[:1] + list_scenes(bench, "quickdraw", count_bins=["5"])[:1]
assert [load_scene(d).n for d in dirs] == [3, 5], dirs
eng = make_engine(os.path.join(W, "tok"), BPE, 128)


class MissFirst(FakeVision):
    """The M5(d) check (gdino) never finds object 0 and sees one extra instance of its class far from every object,
    so verification always fails and every row that checks has to act on it. Evaluation (owlv2) is the normal fake."""

    def __init__(self):
        super().__init__()
        self.gdino = self._det

    def _det(self, img, classes, thr=0.3, **kw):
        out = [dict(cls=c, box=b, score=0.9) for c, b in self.current_gt[1:]]
        return out + [dict(cls=self.current_gt[0][0], box=(0, 0, 6, 6), score=0.9)]


vis = MissFirst()
methods = ["ocsd_v2"] + ROWS
seeds = [0, 1]
out = os.path.join(W, "out")
shutil.rmtree(out, ignore_errors=True)
r = Runner(eng, vis, out, variants={m: var[m] for m in methods}, base_cfg=var["ocsd"])
st = r.run(dirs, methods, seeds=seeds, split="quickdraw")
assert st["failed"] == 0 and st["done"] == len(dirs) * len(seeds) * len(methods), st
st = r.run(dirs, methods, seeds=seeds, split="quickdraw")       # resume: nothing left
assert st["done"] == 0 and st["failed"] == 0, st
print("generation OK", st)


def load(m, sid, s):
    lg = json.load(open(os.path.join(out, "quickdraw", m, f"{sid}_s{s}.json")))
    img = cv2.imread(os.path.join(out, "quickdraw", m, f"{sid}_s{s}.png"))
    assert img is not None and img.shape == (128, 128, 3), (m, sid, s)
    return lg, img.astype(int)


def energy_used(lg):
    return any(len(e) for e in lg.get("energy", []))


T = small["steps"]
k_v2 = int(round((1 - chosen_v2["alpha"]) * T))
# a row whose ablated component does not touch the full first pass (the check / repair rows) only shows in the logs;
# every other row must change the image
differs = {m: True for m in ROWS}
differs.update(v2abl_no_repair=None, v2abl_no_verify=None)
for d in dirs:
    sid = os.path.basename(d)
    scene = load_scene(d)
    for s in seeds:
        lv, iv = load("ocsd_v2", sid, s)
        # OCSD-v2: joint sampler, M2 objects re-imposed for (1 - alpha) T steps, energy on, a failed check -> repair
        assert lv["cfg"]["blend_mode"] == "joint" and lv["blend_steps"] == k_v2 and energy_used(lv), lv
        assert lv["tries"] == 2 and [t.get("kind") for t in lv["verify"]] == ["full", "repair"], lv["verify"]
        assert len(lv["repairs"]) == 1 and lv["repairs"][0]["missing"] == 1, lv.get("repairs")
        for m in ROWS:
            lg, im = load(m, sid, s)
            for f, v in V2_ABLATIONS[m].items():
                assert lg["cfg"][f] == v, (m, f, lg["cfg"][f])
            assert lg["m2_key"] == lv["m2_key"], (m, lg["m2_key"])          # same M2 objects as OCSD-v2
            reps, kinds = lg.get("repairs", []), [t.get("kind") for t in lg["verify"]]
            if m in ("v2abl_separate", "v2abl_no_repair"):   # thesis M5(d): the whole image again, new seed
                assert lg["tries"] == 1 + small["R"] and not reps and kinds == [None] * lg["tries"], (m, lg["verify"])
            elif m in ("v2abl_no_verify", "v2abl_no_m5"):    # no check at all
                assert lg["tries"] == 1 and lg["verify"] == [] and not reps, (m, lg["verify"])
            else:                                            # same check / repair loop as OCSD-v2
                assert lg["tries"] == 2 and len(reps) == 1 and kinds == ["full", "repair"], (m, lg["verify"])
            if m in ("v2abl_no_energy", "v2abl_no_m5"):
                assert not energy_used(lg), m
            else:
                assert energy_used(lg), m
            if m == "v2abl_no_anchor":
                assert lg["blend_steps"] == 0, lg["blend_steps"]
            else:
                assert lg["blend_steps"] == k_v2, (m, lg["blend_steps"])
            if m == "v2abl_no_mincell":
                # the fix only matters where an object covers no attention cell by 30%; use the sampler's own masks
                _, objs, _ = r._objects("quickdraw", scene, var[m], s)      # cached by the run above
                ctx = scene_cond(eng, scene, var["ocsd_v2"], None, compose_foreground(scene, objs, var[m]))
                lh = small["height"] // eng.vae_factor
                lost = any(not attn_masks(ctx.region_masks, res, "cpu", True).equal(
                    attn_masks(ctx.region_masks, res, "cpu", False)) for res in (lh, lh // 2, lh // eng.attn_div))
                want = lost
            else:
                want = differs[m]
            if want is not None:
                assert (np.abs(im - iv).max() > 0) == want, (m, sid, s, int(np.abs(im - iv).max()))
print("per-row evidence OK")

# ---------------------------------------------------------------- evaluation + the E4v2 table
res = os.path.join(W, "results")
shutil.rmtree(res, ignore_errors=True)
df = evaluate_images(out, bench, "quickdraw", methods, seeds, FakeVision(), res, scene_dirs=dirs)
assert set(df.method) == set(methods) and len(df) == len(methods) * len(dirs) * len(seeds), df.groupby("method").size()
assert set(df.n_obj) == {3, 5}
tabs = R.build_all(res, splits=("quickdraw",))
assert "E4v2_quickdraw" in tabs, list(tabs)
tab = tabs["E4v2_quickdraw"]
labels = set(map(str, tab.iloc[:, 0]))
from ocsd.methods import LABELS
missing = [m for m in methods if LABELS.get(m, m) not in labels and m not in labels]
assert not missing and len(tab) == len(methods), (missing, tab.iloc[:, 0].tolist())
assert os.path.exists(os.path.join(res, "tables", "E4v2_quickdraw.csv"))
print(tab.to_string())
print("V2 ABLATION TEST OK")
