"""Kiểm thử khói (CPU): chạy MỌI phương pháp + đánh giá + báo cáo với mô hình thu nhỏ. python tests/smoke_test.py"""
import os, sys, time, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import torch
from tiny import FakeVision, make_engine
from ocsd.config import OCSDConfig
from ocsd.data import list_scenes
from ocsd import methods as M
from ocsd.runner import Runner
from ocsd.metrics import evaluate_images
from ocsd import report

W = os.environ.get("SMOKE_DIR", "/tmp/ocsd_smoke")
BENCH = os.environ.get("SMOKE_BENCH")
BPE = os.environ.get("CLIP_BPE")
eng = make_engine(os.path.join(W, "tok"), BPE, 128)
vis = FakeVision()
small = dict(height=128, width=128, steps=6, obj_steps=4, K=2, S1=3, S2=3, tau=2, R=1, lora_rank=4)
variants = {k: v.replace(**small) for k, v in M.OCSD_VARIANTS.items()}
base = OCSDConfig().replace(**small)
methods = [m for m in M.BASELINES if m != "gligen"] + list(variants)
scenes = list_scenes(BENCH, split="quickdraw", limit=int(os.environ.get("SMOKE_N", "3")))
out = os.path.join(W, "outputs")
r = Runner(eng, vis, out, variants=variants, base_cfg=base)
t = time.time()
st = r.run(scenes, methods, seeds=[0, 1], split="quickdraw")
print("generation", time.time() - t, st)
assert st["failed"] == 0, st
st2 = r.run(scenes, methods, seeds=[0, 1], split="quickdraw")   # tiếp tục: phải bỏ qua hết
assert st2["done"] == 0 and st2["failed"] == 0, st2
res = os.path.join(W, "results")
df = evaluate_images(out, BENCH, "quickdraw", methods, [0, 1], vis, res, scene_dirs=scenes)
print(df.groupby("method")[["opr", "oce_c", "miou", "ra", "clip", "obj_clip", "id_sim"]].mean().round(3))
tabs = report.build_all(res, splits=("quickdraw",))
for k, v in tabs.items():
    print("==", k); print(v.head(20).to_string())
print(report.plot_curves(res, "quickdraw"))
print(report.plot_alpha(res, "quickdraw"))
sids = report.pick_showcase(res, "quickdraw", k=2)
print(report.qualitative_grid(out, BENCH, "quickdraw", sids, ["controlnet", "t2i_adapter", "zhang2025", "ocsd"],
                              out=os.path.join(res, "figures", "qual.png")))
print(report.user_study_pack(out, BENCH, "quickdraw", sids, ["controlnet", "zhang2025", "ocsd"], os.path.join(res, "user_study")))
print("SMOKE TEST OK")
