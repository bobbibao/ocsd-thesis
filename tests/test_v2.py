"""CPU tests of OCSD-v2 and the evaluation changes with tiny random models: small-object mask fix, mean energy,
shrinking anchor masks, caption class words, colour decomposition, conditional RA / matched IoU, the joint sampler,
object-level repair, the collage and best-of-N baselines, per-seed M2 objects, and the new report functions.
python tests/test_v2.py"""
import glob, json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
from tiny import FakeVision, make_clip_tokenizer, make_engine
from ocsd import report as R
from ocsd.attention import OCSDController, attention_energy, binarize_masks
from ocsd.data import build_quickdraw_scenes, download_quickdraw, list_scenes, load_scene
from ocsd.matching import consistency
from ocsd.method import anchor_masks, build_scene, generate, prepare, repair_regions, repair_scene, scene_cond
from ocsd.methods import OCSD_VARIANTS
from ocsd.runner import Runner
from ocsd.sketch import Scene, SceneObject, decompose_by_color, make_caption, scene_global_prompt

W = os.environ.get("V2_DIR", "/tmp/ocsd_v2")
BPE = os.path.join(os.path.dirname(open_clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
os.makedirs(W, exist_ok=True)

# ---------------------------------------------------------------- small-object mask fix (M5(a)/(b))
soft = torch.zeros(2, 8, 8)
soft[0, 2, 3], soft[0, 2, 4] = 0.2, 0.1
soft[1, 5:7, 5:7] = 1.0
b0, b1 = binarize_masks(soft), binarize_masks(soft, min_cell=True)
assert not b0[0].any() and int(b1[0].sum()) == 1 and bool(b1[0, 2, 3]) and torch.equal(b0[1], b1[1])
m = torch.zeros(1, 64, 64)
m[0, 22:26, 22:28] = 1          # < 30% of every 8x8-latent cell -> vanishes at the 8x8 (mid-block) resolution
for fix, everywhere in ((False, True), (True, False)):
    ctrl = OCSDController(8)
    ctrl.set_regions(m, [[1, 2]], [], min_cell=fix)
    pen = ctrl._base_bias(64, "cpu")
    assert bool((pen[:, 1] == 1).all()) == everywhere, (fix, pen[:, 1])
print("min-cell OK")

# ---------------------------------------------------------------- mean energy
maps = torch.rand(3, 4, 4)
masks = (torch.rand(3, 4, 4) > 0.5).float()
masks[:, 0, 0] = 1
assert torch.isclose(attention_energy(maps, masks, 1.0) / 3, attention_energy(maps, masks, 1.0, "mean"))

# ---------------------------------------------------------------- shrinking anchor masks keep small objects
placed = [np.zeros((128, 128), bool) for _ in range(2)]
placed[0][10:90, 10:90] = True
placed[1][100:108, 100:106] = True
am = anchor_masks(placed, 0.3, 5, 16, "cpu", torch.float32)
areas = [float(a.sum()) for a in am]
assert areas == sorted(areas, reverse=True) and areas[0] > areas[-1] > 0
assert float(am[-1][..., 12:14, 12:14].sum()) > 0           # the 8x6 px object keeps its core
assert float(anchor_masks(placed, 0.0, 3, 16, "cpu", torch.float32)[-1].sum()) == areas[0]
print("anchor masks OK")

# ---------------------------------------------------------------- caption words naming a class
tok = make_clip_tokenizer(BPE, os.path.join(W, "tok"))
tok.add_tokens(["<o0>", "<o1>"])
z8 = np.zeros((8, 8), bool)
objs = [SceneObject("person", "person", (0, 0, 4, 4), z8, z8), SceneObject("horse", "horse", (4, 4, 8, 8), z8, z8)]
sc = Scene("t", z8, objs, "on a beach", "Two men riding a horse on a beach.")
gp = scene_global_prompt(tok, sc, None, caption=True, class_groups=True)
ids = tok(gp.text).input_ids
assert [ids[i] for i in gp.groups["cls:person"]] == tok("men", add_special_tokens=False).input_ids
assert [ids[i] for i in gp.groups["cls:horse"]] == tok("horse", add_special_tokens=False).input_ids
assert min(gp.groups["cls:person"]) > max(gp.groups["bg"])     # only caption tokens, never the object list
qd = Scene("q", z8, objs, "on a beach", make_caption(["person", "horse"], "on a beach"))
assert not any(k.startswith("cls:") for k in scene_global_prompt(tok, qd, None, caption=True, class_groups=True).groups)
print("caption class groups OK")

# ---------------------------------------------------------------- one object per stroke colour (app)
img = np.full((64, 64, 3), 255, np.uint8)
cv2.line(img, (5, 5), (20, 20), (255, 0, 0), 2)
cv2.line(img, (40, 40), (60, 50), (0, 0, 255), 2)
cv2.line(img, (5, 50), (20, 60), (0, 0, 0), 2)
assert len(decompose_by_color(img, min_pixels=10)) == 3
black = np.full((64, 64, 3), 255, np.uint8)
cv2.line(black, (5, 5), (60, 60), (0, 0, 0), 2)
assert decompose_by_color(black, min_pixels=10) == []

# ---------------------------------------------------------------- RA / IoU that only count detected objects
two = Scene("r", z8, [SceneObject("dog", "dog", (0, 0, 10, 10), z8, z8), SceneObject("cat", "cat", (20, 0, 30, 10), z8, z8)],
            "x", "x", relations=[(0, 1, "left_of")])
c = consistency(two, [dict(cls="dog", box=(0, 0, 10, 10), score=0.9)], 0.1)
assert c["ra"] == 0.0 and np.isnan(c["ra_cond"]) and c["miou_matched"] == 1.0 and c["miou"] == 0.5
c = consistency(two, [dict(cls="dog", box=(0, 0, 10, 10), score=0.9), dict(cls="cat", box=(20, 0, 30, 10), score=0.9)])
assert c["ra"] == c["ra_cond"] == 1.0
print("matching OK")

# ---------------------------------------------------------------- joint sampler, repair, baselines on tiny models
qd_raw = os.environ.get("QD_RAW", os.path.join(W, "qd_raw"))
download_quickdraw(qd_raw, per_class=40)
bench = os.path.join(W, "bench")
shutil.rmtree(os.path.join(W, "out"), ignore_errors=True)
build_quickdraw_scenes(qd_raw, os.path.join(bench, "quickdraw"), per_cell=1, size=128)
dirs = list_scenes(bench, "quickdraw", count_bins=["3"])[:1] + list_scenes(bench, "quickdraw", count_bins=["5"])[:1]
eng = make_engine(os.path.join(W, "tok"), BPE, 128)
vis = FakeVision()
small = dict(height=128, width=128, steps=6, obj_steps=4, K=2, S1=3, S2=3, tau=2, R=1, lora_rank=4, bo_n=2)
v2 = OCSD_VARIANTS["ocsd_v2"].replace(**small)


class MissFirst(FakeVision):
    """Detector that always misses object 0 and adds one extra detection far from every object."""

    def __init__(self, scene):
        super().__init__()
        self.set_scene(scene)
        self.gdino = self._det

    def _det(self, img, classes, thr=0.3, **kw):
        out = [dict(cls=c, box=b, score=0.9) for c, b in self.current_gt[1:]]
        return out + [dict(cls=self.current_gt[0][0], box=(0, 0, 6, 6), score=0.9)]


scene = load_scene(dirs[1])
mv = MissFirst(scene)
prep = prepare(eng, mv, scene, v2, seed=0)
log = {}
im = generate(eng, mv, scene, prep, v2, 0, log=log)
assert im.shape == (128, 128, 3) and log["tries"] == 2 and len(log["repairs"]) == 1 and log["verify"][1]["kind"] == "repair"
# repair only changes the failing regions: object 0 and the extra detection
img0 = build_scene(eng, scene, prep.objs, v2, 0, None, None, prep.fg)
dets = mv.gdino(img0, [o.cls for o in scene.objects])
c0 = consistency(scene, dets, 0.1)
ctx = scene_cond(eng, scene, v2, None, prep.fg)
M, missing = repair_regions(scene, ctx, c0, dets, 128)
assert missing == [0] and (M & ctx.region_masks[0]).sum() == ctx.region_masks[0].sum()
img1 = repair_scene(eng, scene, prep, v2, img0, c0, dets, 7, v2.lambda0 * 1.5)
far = cv2.dilate(M.astype(np.uint8), np.ones((31, 31), np.uint8)) == 0
assert far.any() and np.abs(img1.astype(int) - img0.astype(int))[far].max() <= 1
# the thesis sampler with the same settings still runs (v2abl_separate) and the identity path of the joint sampler too
build_scene(eng, scene, prep.objs, v2.replace(blend_mode="separate"), 0, None, None, prep.fg)
print("joint sampler + repair OK")

variants = {k: OCSD_VARIANTS[k].replace(**small) for k in ("ocsd_v2", "v2abl_separate", "v2abl_no_repair",
                                                           "v2abl_no_anchor", "v2abl_energy_sum", "pv2_a0.6")}
base = OCSD_VARIANTS["ocsd"].replace(**small)
bl_cfg = base.replace(region_min_cell=True, caption_class_masks=True)
baselines = {b: (b, bl_cfg.replace(m2_per_seed=True) if b.startswith("collage") else bl_cfg)
             for b in ("controlnet", "controlnet_bon", "collage", "collage_bo3", "cn_region", "cn_energy")}
baselines["pcn_s0.7"] = ("controlnet", bl_cfg.replace(cn_scale=0.7))
out = os.path.join(W, "out")
r = Runner(eng, vis, out, variants=variants, base_cfg=base, baselines=baselines)
st = r.run(dirs, list(variants) + list(baselines), seeds=[0, 1], split="quickdraw")
assert st["failed"] == 0 and st["done"] == 2 * 2 * (len(variants) + len(baselines)), st
sid = os.path.basename(dirs[0])
assert os.path.exists(os.path.join(out, "quickdraw", "_objects", "K2_phrase_s1", sid, "objects.json"))
lg = json.load(open(os.path.join(out, "quickdraw", "ocsd_v2", f"{sid}_s1.json")))
assert lg["m2_key"] == "K2_phrase_s1" and lg["cfg"]["blend_mode"] == "joint"
lg = json.load(open(os.path.join(out, "quickdraw", "collage_bo3", f"{sid}_s1.json")))
assert lg["m2_key"] == "K2_phrase_s1" and 1 <= lg["tries"] <= 2
assert json.load(open(os.path.join(out, "quickdraw", "pcn_s0.7", f"{sid}_s0.json")))["cfg"]["cn_scale"] == 0.7
st = r.run(dirs, list(variants) + list(baselines), seeds=[0, 1], split="quickdraw")   # resume: nothing left
assert st["done"] == 0 and st["failed"] == 0
print("runner OK")

# ---------------------------------------------------------------- report: prereg, stats families, robustness, pareto
res = os.path.join(W, "results")
shutil.rmtree(res, ignore_errors=True)
rng = np.random.default_rng(0)
rows, det_rows = [], []
methods = ["ocsd_v2", "ocsd", "gligen", "controlnet", "ocsd_lite", "gligen_bon", "controlnet_bon"]
for k, m in enumerate(methods):
    for i in range(24):
        for s in (0, 1):
            opr = float(np.clip(0.5 + 0.05 * (len(methods) - k) / len(methods) + rng.normal(0, 0.1), 0, 1))
            row = dict(method=m, sid=f"qd_{i:02d}", seed=s, split="quickdraw", n_obj=8 if i < 12 else 3,
                       count_bin="8+" if i < 12 else "3", complexity="simple", opr=opr, oce=1.0, oce_c=1.0,
                       count_acc=0.0, miou=0.5, ra=0.5, clip=25.0, obj_clip=20.0, id_sim=0.4)
            rows.append(row)
            det_rows.append({k2: row[k2] for k2 in ("method", "sid", "seed", "split", "n_obj", "count_bin", "complexity")}
                            | dict(n_eval=row["n_obj"], opr=opr * 0.9, oce_c=1.0, count_acc=0.0, miou=0.5,
                                   miou_matched=0.7, ra=0.5, ra_cond=0.8))
for name, data, sub in (("main", rows, ""), ("det", det_rows, "det_owlv2d")):
    d = os.path.join(res, "quickdraw", sub)
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame(data)
    for m in methods:
        df[df.method == m].to_csv(os.path.join(d, f"per_image_{m}.csv"), index=False)
df = R.load_per_image(res, "quickdraw")
ps = R.per_scene(df)
pr = R.prereg_tests(ps, new_sids={f"qd_{i:02d}" for i in range(6, 12)})
for fam, ref in (("primary:H1", "ocsd"), ("primary:H2", "ocsd_v2")):
    prim = pr[pr.family == fam]
    assert len(prim) == 2 and set(prim.ref) == {ref} and set(prim.method) == {"gligen", "controlnet"}
    assert (prim.p_holm >= prim.p - 1e-12).all() and prim.subset.iloc[0] == "count_bin=8+"
assert {"secondary:new_scenes_8plus", "secondary:new_scenes_8plus_v2", "secondary:compute_matched",
        "secondary:v2_vs_thesis", "secondary:identity_learning"} <= set(pr.family)
st = R.stats_families(df, methods, set(ps.sid))
assert {"main_vs_ocsd", "main_vs_ocsd_v2"} == set(st.family)
rb = R.robustness_table(res, "quickdraw", df, methods, set(ps.sid))
assert rb is not None and "Kendall" in rb.iloc[-1, 0] and any("RA-cond" in c for c in rb.columns)
tabs = R.build_all(res, splits=("quickdraw",))
assert "robustness_quickdraw" in tabs and "E3_quickdraw" in tabs
# pareto: per-image rows of the sweeps + their KID
pm = R.PARETO_METHODS
prow = [dict(r0, method=m, opr=float(i) / len(pm), clip=20.0 + i % 5) for i, m in enumerate(pm)
        for r0 in rows[:4]]
pd.DataFrame(prow).to_csv(os.path.join(res, "quickdraw", "per_image_pareto.csv"), index=False)
pd.DataFrame([dict(method=m, fid=10.0 + i, kid=float(i)) for i, m in enumerate(pm)]).to_csv(
    os.path.join(res, "quickdraw", "quality_fid_kid.csv"), index=False)
f = R.plot_pareto(res, "quickdraw")
assert f and os.path.exists(f)
assert "pareto_quickdraw" in R.build_all(res, splits=("quickdraw",))
# realism A/B pack
for m in ("ocsd_v2", "controlnet"):
    os.makedirs(os.path.join(out, "quickdraw", m), exist_ok=True)
pack = R.realism_pairs_pack(out, "quickdraw", [sid], [("ocsd_v2", "controlnet")], os.path.join(res, "realism"))
key = pd.read_csv(os.path.join(pack, "answer_key.csv"))
assert len(key) == 1 and {key.left[0], key.right[0]} == {"ocsd_v2", "controlnet"}
print("report OK")
print("V2 TEST OK")
