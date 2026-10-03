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
                                  seeds=[0, 1], seeds_all=1, tune=True, power=["ocsd", "ocsd_lite", "gligen", "controlnet"],
                                  cfg=small)
config.TUNE_GRID = dict(alpha=[0.0, 0.5], lora_scale=[0.5, 1.0])
config.TUNE_SEEDS = [0]
config.TUNE_ENERGY = {k: v for k, v in config.TUNE_ENERGY.items() if k in ('e_off', 'e_id20', 'e_ph20')}
config.TUNE_M5_ALPHA = [0.5]
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

# a tuned.json from before phase 3 existed (phases 1-2 only, no "phases" key) resumes at phase 3
stages.tune(P, E, eng=eng, vis=vis)
tp = os.path.join(P.results, "tuning", "tuned.json")
tuned = json.load(open(tp))
assert tuned["phases"] == list(stages.TUNE_PHASES)
for k in ("use_region_attn", "phases", "method_m5", "reference_m5"):
    tuned.pop(k, None)
tuned["chosen"].pop("use_region_attn")
json.dump(tuned, open(tp, "w"))
assert not stages.tuning_finished(P)
stages.tune(P, E, eng=eng, vis=vis)
tuned = json.load(open(tp))
print("tuned:", tuned)
assert stages.tuning_finished(P) and "use_region_attn" in tuned["chosen"]
for name in ("tuning", "tuning_energy", "tuning_m5"):
    assert os.path.exists(os.path.join(P.results, "tuning", f"{name}_table.md")), name
var, _ = stages.tier_variants(E, P=P)
from ocsd.methods import VARIANT_OVERRIDES
for k in ("ocsd", "ocsd_lite", "abl_no_region", "abl_no_energy", "abl_m5ab_both"):
    for f, v in tuned["chosen"].items():
        assert getattr(var[k], f) == VARIANT_OVERRIDES[k].get(f, v), (k, f)
assert var["abl_no_energy"].use_energy is False and var["abl_no_region"].use_region_attn is False
assert var["abl_m5ab_both"].use_energy and var["abl_m5ab_both"].use_region_attn
assert var["alpha_1.0"].alpha == 1.0 and var["abl_alpha0"].alpha == 0.0 and var["alpha_0.5"].alpha == 0.5
assert var["zhang2025"].alpha == 0.5 and var["zhang2025"].lora_scale == 1.0 and var["zhang2025"].K == 1
assert not var["zhang2025"].use_caption and var["ocsd"].use_caption
assert var["abl_m3_long"].S1 == 200 and var["ocsd"].S1 == 3     # the tier setting never replaces a variant's own

# outputs made before .gen_configs.json existed: only what the new defaults change is stale
# (caption in P_g -> COCO rows of the OCSD family and the P_g baselines); Zhang et al. and QuickDraw are kept
leg = stages._gen_signatures(E, tuned=tuned["chosen"], defaults=config.LEGACY_DEFAULTS)
new = stages._gen_signatures(E, P=P)
assert leg["quickdraw"]["ocsd"] == new["quickdraw"]["ocsd"] and leg["quickdraw"]["cn_region"] == {}
assert leg["coco"]["ocsd"] != new["coco"]["ocsd"] and leg["coco"]["cn_energy"] != new["coco"]["cn_energy"]
assert leg["coco"]["zhang2025"] == new["coco"]["zhang2025"]

plan = stages.experiment_plan(P, E)
names = {j["name"]: j for j in plan}
# ablation rows identical to OCSD after tuning are not generated (the tiny tier has no 3/5-object trained scene, so
# its E4 job is empty: check the rule on the variants instead)
from ocsd.report import ABLATION
abl = [m for m in ABLATION if m not in ("ocsd", "zhang2025", "ocsd_lite") and var[m] != var["ocsd"]]
assert "abl_m3_long" in abl and ("abl_m5ab_both" in abl) != (var["ocsd"].use_region_attn and var["ocsd"].use_energy)
assert ("abl_no_region" in abl) == var["ocsd"].use_region_attn and ("abl_no_energy" in abl) == var["ocsd"].use_energy
if "E4_ablation" in names:
    assert names["E4_ablation"]["methods"] == abl
pw = names["E3_quickdraw_power"]
assert set(pw["methods"]) == {"ocsd", "ocsd_lite", "controlnet"} and pw["seeds"] == [0, 1]
assert pw["scenes"] == names["E3_quickdraw_free"]["scenes"]
used = {os.path.basename(d) for j in plan for d in j["scenes"]}
assert used and not used & set(pilot_sids), used & set(pilot_sids)
names = {j["name"]: j for j in plan}
assert names["E3_quickdraw_free"]["seeds"] == [0] and names["E3_quickdraw_free_seeds"]["seeds"] == [1]
stages.generate(P, E, eng=eng, vis=vis)
stages.evaluate(P, E, vis=vis, fid=False)
stages.report(P, E)
# changed tuned settings -> OCSD-family images are archived and generated again, baselines are kept
n_cn = len(os.listdir(os.path.join(P.outputs, "quickdraw", "controlnet")))
tp = os.path.join(P.results, "tuning", "tuned.json"); tj = json.load(open(tp))
tj["chosen"]["eta"] = 7.0; json.dump(tj, open(tp, "w"))
stages.generate(P, E, eng=eng, vis=vis)
assert glob.glob(os.path.join(P.outputs, "_stale", "*", "quickdraw", "ocsd"))
assert len(os.listdir(os.path.join(P.outputs, "quickdraw", "controlnet"))) == n_cn
assert stages._count_todo(P, [j for j in stages.experiment_plan(P, E) if j["name"] == "E3_quickdraw_trained"][0]) == 0
stages.evaluate(P, E, vis=vis, fid=False)
rows = [l for f in glob.glob(os.path.join(P.results, "quickdraw", "per_image_*.csv")) for l in open(f).readlines()[1:]]
assert rows and not any(s in l for l in rows for s in pilot_sids)

# rows left by a later pilot-tier run (tuning scenes) must not reach the paper tables
import pandas as pd
from ocsd import metrics, report as R
cf = os.path.join(P.results, "quickdraw", "per_image_controlnet.csv")
d = pd.read_csv(cf); extra = d.iloc[:1].copy(); extra["sid"] = pilot_sids[0]
pd.concat([d, extra]).to_csv(cf, index=False)
assert os.path.exists(os.path.join(P.results, "quickdraw", "plan.json"))
assert pilot_sids[0] not in set(R.load_per_image(P.results, "quickdraw").sid)

# FID/KID: `pairs` picks exactly those images per method, and a changed selection is not reused from the cache
seen = []


def fake_fid_kid(gen, ref, dev):
    seen.append(len(gen))
    return dict(fid=float(len(gen)), kid=0.0, kid_std=0.0, n_gen=len(gen))


metrics.fid_kid = fake_fid_kid
plan = stages.experiment_plan(P, E)
tr = [j for j in plan if j["name"] == "E3_quickdraw_trained"][0]
pairs = {(os.path.basename(x), s) for x in tr["scenes"] for s in tr["seeds"]}
ref = glob.glob(os.path.join(P.outputs, "quickdraw", "controlnet", "*.png"))
q = metrics.image_level_quality(P.outputs, P.benchmarks, "quickdraw", ["controlnet", "ocsd"], [0, 1], ref, P.results,
                                with_lpips=False, pairs={"controlnet": pairs, "ocsd": pairs})
assert list(q.n_gen) == [len(pairs), len(pairs)], q
q2 = metrics.image_level_quality(P.outputs, P.benchmarks, "quickdraw", ["controlnet"], [0], ref, P.results,
                                 with_lpips=False)
assert len(seen) == 3 and int(q2.n_gen.iloc[0]) != len(pairs), (seen, q2)
q3 = metrics.image_level_quality(P.outputs, P.benchmarks, "quickdraw", ["controlnet"], [0], ref, P.results,
                                 with_lpips=False)
assert len(seen) == 3 and int(q3.n_gen.iloc[0]) == int(q2.n_gen.iloc[0])   # same selection -> cached
print("PLAN FILTER + FID PAIRS OK")
print("TUNE TEST OK")
