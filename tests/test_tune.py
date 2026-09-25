"""CPU test of the pilot -> paper flow with tiny models: tuning split, pilot archive, stage T (tuning),
tuned values applied in generation, evaluation scenes disjoint from tuning scenes. python tests/test_tune.py"""
import glob, json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import open_clip
from tiny import FakeVision, make_engine
from ocsd import config, stages
from ocsd.config import ExperimentConfig, Paths
from ocsd.data import build_quickdraw_scenes, download_quickdraw

W = os.environ.get("TUNE_DIR", "/tmp/ocsd_tune")
shutil.rmtree(os.path.join(W, "root"), ignore_errors=True)
P = Paths(os.path.join(W, "root")); P.makedirs()
qd_raw = os.path.join(W, "qd_raw")
download_quickdraw(qd_raw, per_class=40)
small = dict(height=128, width=128, steps=6, obj_steps=4, K=2, S1=3, S2=3, tau=2, R=1, lora_rank=4)
config.TIERS["tiny_pilot"] = dict(qd_per_cell=1, trained_n=2, coco_n=0, coco_trained_n=0, ablation_n=1, alpha_n=1,
                                  seeds=[0], seeds_all=1, tune=False, cfg=small)
config.TIERS["tiny_paper"] = dict(qd_per_cell=1, trained_n=2, coco_n=0, coco_trained_n=0, ablation_n=1, alpha_n=1,
                                  seeds=[0, 1], seeds_all=1, tune=True, cfg=small)
config.TUNE_GRID = dict(alpha=[0.0, 0.5], lora_scale=[0.5, 1.0])
config.TUNE_SEEDS = [0]
stages.MAIN[:] = [m for m in stages.MAIN if m != "gligen"]   # GLIGEN weights cannot be downloaded offline
qd = os.path.join(P.benchmarks, "quickdraw")

# pilot: 1 scene per cell, some fake pilot results that must be archived
build_quickdraw_scenes(qd_raw, qd, per_cell=1, size=128)
pilot_sids = sorted(os.listdir(qd))
os.makedirs(os.path.join(P.results, "quickdraw"), exist_ok=True)
open(os.path.join(P.results, "quickdraw", "per_image_ocsd.csv"), "w").write("method,sid,seed\nocsd,x,0\n")

# paper: freeze split + build the extra scenes (what setup_data does, without the downloads)
E = ExperimentConfig(tier="tiny_paper")
split = stages.freeze_tuning_split(P)
assert split["quickdraw"] == pilot_sids, split
assert os.path.exists(os.path.join(P.results, "pilot", "quickdraw", "per_image_ocsd.csv"))
assert not os.path.exists(os.path.join(P.results, "quickdraw"))
extra = stages._max_per_cell(P, split["quickdraw"])
build_quickdraw_scenes(qd_raw, qd, per_cell=1 + extra, size=128)
assert len(os.listdir(qd)) == 2 * len(pilot_sids)

eng = make_engine(os.path.join(W, "tok"), os.path.join(os.path.dirname(open_clip.__file__),
                                                       "bpe_simple_vocab_16e6.txt.gz"), 128)
vis = FakeVision()
try:
    stages.generate(P, E, eng=eng, vis=vis)
    raise AssertionError("generate must refuse to run before tuning")
except RuntimeError as e:
    print("ok, refused:", e)

stages.tune(P, E, eng=eng, vis=vis)
tuned = json.load(open(os.path.join(P.results, "tuning", "tuned.json")))
print("tuned:", tuned)
var, _ = stages.tier_variants(E, P=P)
for k in ("ocsd", "ocsd_lite", "abl_no_region"):
    assert var[k].alpha == tuned["chosen"]["alpha"] and var[k].lora_scale == tuned["chosen"]["lora_scale"], k
assert var["alpha_1.0"].alpha == 1.0 and var["abl_alpha0"].alpha == 0.0
assert var["zhang2025"].alpha == 0.5 and var["zhang2025"].lora_scale == 1.0

plan = stages.experiment_plan(P, E)
used = {os.path.basename(d) for j in plan for d in j["scenes"]}
assert used and not used & set(pilot_sids), used & set(pilot_sids)
names = {j["name"]: j for j in plan}
assert names["E3_quickdraw_free"]["seeds"] == [0] and names["E3_quickdraw_free_seeds"]["seeds"] == [1]
stages.generate(P, E, eng=eng, vis=vis)
stages.evaluate(P, E, vis=vis, fid=False)
stages.report(P, E)
rows = [l for f in glob.glob(os.path.join(P.results, "quickdraw", "per_image_*.csv")) for l in open(f).readlines()[1:]]
assert rows and not any(s in l for l in rows for s in pilot_sids)
print("TUNE TEST OK")
