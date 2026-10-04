"""CPU test of the pilot -> paper flow with tiny models: tuning split, pilot archive, stage T (all phases, incl. OCSD-v2
and the baselines), tuned values applied in generation, evaluation scenes disjoint from tuning scenes, stale outputs,
the power job, extra evaluators and the report. python tests/test_tune.py"""
import glob, json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import open_clip
from tiny import FakeVision, make_engine
from ocsd import config, stages
from ocsd.config import ExperimentConfig, Paths
from ocsd.data import build_quickdraw_scenes, download_quickdraw
import pandas as pd

W = os.environ.get("TUNE_DIR", "/tmp/ocsd_tune")
shutil.rmtree(os.path.join(W, "root"), ignore_errors=True)
P = Paths(os.path.join(W, "root")); P.makedirs()
qd_raw = os.path.join(W, "qd_raw")
download_quickdraw(qd_raw, per_class=40)
small = dict(height=128, width=128, steps=6, obj_steps=4, K=2, S1=3, S2=3, tau=2, R=1, lora_rank=4, bo_n=2)
config.TIERS["tiny_pilot"] = dict(qd_per_cell=1, trained_n=2, coco_n=0, coco_trained_n=0, ablation_n=1, alpha_n=1,
                                  seeds=[0], seeds_all=1, tune=False, cfg=small)
config.TIERS["tiny_paper"] = dict(qd_per_cell=1, trained_n=2, coco_n=0, coco_trained_n=0, ablation_n=1, alpha_n=1,
                                  seeds=[0, 1], seeds_all=1, tune=True, tune_baselines=True, pareto=True,
                                  power=list(config.TIERS["paper"]["power"]), cfg=small)
config.TUNE_GRID = dict(alpha=[0.0, 0.5], lora_scale=[0.5, 1.0])
config.TUNE_SEEDS = [0]
config.TUNE_ENERGY = {k: v for k, v in config.TUNE_ENERGY.items() if k in ('e_off', 'e_id20', 'e_ph20')}
config.TUNE_M5_ALPHA = [0.5]
config.TUNE_V2 = dict(alpha=[0.2, 0.4], anchor_shrink=[0.3])
config.TUNE_BASELINES = [("controlnet", "cn_scale", [0.8, 1.0]), ("cn_region", "lambda0", [8.0]),
                         ("collage", "collage_strength", [0.3]), ("zhang2025", "alpha", [0.5])]
stages.MAIN[:] = [m for m in stages.MAIN if m not in ("gligen", "gligen_bon")]   # GLIGEN cannot be downloaded offline
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
assert tuned["phases"] == stages.tune_phases(E) == list(stages.TUNE_PHASES)
for k in ("use_region_attn", "phases", "method_m5", "reference_m5"):
    tuned.pop(k, None)
tuned["chosen"].pop("use_region_attn")
json.dump(tuned, open(tp, "w"))
assert not stages.tuning_finished(P, E)
stages.tune(P, E, eng=eng, vis=vis)
tuned = json.load(open(tp))
print("tuned:", tuned)
assert stages.tuning_finished(P, E) and "use_region_attn" in tuned["chosen"]
for name in ("tuning", "tuning_energy", "tuning_m5", "tuning_v2", "tuning_bl_controlnet", "tuning_bl_cn_region",
             "tuning_bl_collage", "tuning_bl_zhang2025"):
    assert os.path.exists(os.path.join(P.results, "tuning", f"{name}_table.md")), name
assert set(tuned["chosen_v2"]) == {"alpha", "anchor_shrink"}
assert set(tuned["baselines"]) == {"controlnet", "cn_region", "collage", "zhang2025"}
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
# OCSD-v2 family: its own tuned values, never the thesis ones; ablations keep what defines them
v2 = var["ocsd_v2"]
assert (v2.alpha, v2.anchor_shrink) == (tuned["chosen_v2"]["alpha"], tuned["chosen_v2"]["anchor_shrink"])
assert v2.blend_mode == "joint" and not v2.use_identity and v2.energy_reduce == "mean" and v2.lora_scale == 0.5
assert var["v2abl_no_anchor"].alpha == 1.0 and var["v2abl_separate"].blend_mode == "separate"
assert var["v2abl_no_shrink"].anchor_shrink == 0.0 and var["v2abl_no_shrink"].alpha == v2.alpha
assert var["pv2_a0.6"].alpha == 0.6 and not var["pv2_a0.6"].use_verify and var["pv1lite_a0.3"].alpha == 0.3
assert var["zhang2025"].alpha == tuned["baselines"]["zhang2025"]["alpha"]
# baselines: tuned knobs, inherited ControlNet scale, the fixes, and untuned sweep points
bl = stages.tier_baselines(E, P=P)
cs = tuned["baselines"]["controlnet"]["cn_scale"]
assert bl["controlnet"][1].cn_scale == bl["controlnet_bon"][1].cn_scale == bl["cn_region"][1].cn_scale == cs
assert bl["cn_region"][1].lambda0 == tuned["baselines"]["cn_region"]["lambda0"] and bl["cn_region"][1].region_min_cell
assert bl["collage"][1].m2_per_seed and bl["collage_bo3"][1].collage_strength == bl["collage"][1].collage_strength
assert bl["pcn_s0.7"][0] == "controlnet" and bl["pcn_s0.7"][1].cn_scale == 0.7 and bl["controlnet_bon"][1].bo_n == 2

# outputs made before .gen_configs.json existed: only what changed is stale - the caption (COCO rows of the OCSD
# family and the P_g baselines) and the region-mask fix of the P_g baselines; Zhang et al. and QuickDraw OCSD are kept
leg = stages._gen_signatures(E, tuned=tuned["chosen"], defaults=config.LEGACY_DEFAULTS, legacy=True)
new = stages._gen_signatures(E, P=P)
same = lambda sp, m: stages._complete(leg[sp][m], new[sp][m]) == new[sp][m]
assert same("quickdraw", "ocsd") and not same("quickdraw", "cn_region") and not same("coco", "ocsd")
assert not same("coco", "cn_energy") and same("coco", "zhang2025")
assert same("quickdraw", "controlnet") == (cs == 1.0)

plan = stages.experiment_plan(P, E)
names = {j["name"]: j for j in plan}
# ablation rows identical to OCSD after tuning are not generated (the tiny tier has no 3/5-object trained scene, so
# its E4 job is empty: check the rule on the variants instead)
from ocsd.report import ABLATION, ABLATION_V2, PARETO_METHODS
abl = [m for m in ABLATION if m not in ("ocsd", "zhang2025", "ocsd_lite") and var[m] != var["ocsd"]]
assert "abl_m3_long" in abl and ("abl_m5ab_both" in abl) != (var["ocsd"].use_region_attn and var["ocsd"].use_energy)
assert ("abl_no_region" in abl) == var["ocsd"].use_region_attn and ("abl_no_energy" in abl) == var["ocsd"].use_energy
if "E4_ablation" in names:
    assert names["E4_ablation"]["methods"] == abl
assert all(var[m] != v2 for m in ABLATION_V2[1:])
pw = names["E3_quickdraw_power"]
assert set(pw["methods"]) == {"ocsd_v2", "ocsd", "ocsd_lite", "controlnet", "controlnet_bon"} and pw["seeds"] == [0, 1]
assert pw["scenes"] == names["E3_quickdraw_free"]["scenes"] and pw["priority"] == 1
assert names["pareto"]["methods"] == PARETO_METHODS and names["pareto"]["priority"] == 2
assert {"ocsd_v2", "collage", "collage_bo3", "controlnet_bon"} <= set(names["E3_quickdraw_free"]["methods"])
# main-table images come before the power job's, which come before the sweeps
items = stages.work_items(P, E)
prio = [min(j["priority"] for j in plan if j["split"] == sp and d in j["scenes"]
            and any(m in j["methods"] and s in j["seeds"] for m, s in ms)) for sp, d, ms in items]
assert prio == sorted(prio)
# the sweeps need GLIGEN (not available offline): leave them out from here on
E = ExperimentConfig(tier="tiny_paper", skip_jobs="pareto")
plan = stages.experiment_plan(P, E)
assert "pareto" not in {j["name"] for j in plan}
used = {os.path.basename(d) for j in plan for d in j["scenes"]}
assert used and not used & set(pilot_sids), used & set(pilot_sids)
names = {j["name"]: j for j in plan}
assert names["E3_quickdraw_free"]["seeds"] == [0] and names["E3_quickdraw_free_seeds"]["seeds"] == [1]
stages.generate(P, E, eng=eng, vis=vis)
stages.evaluate(P, E, vis=vis, fid=False)
stages.report(P, E)
qo = os.path.join(P.outputs, "quickdraw")
assert glob.glob(os.path.join(qo, "_objects", "K2_phrase_s1", "*", "objects.json"))      # M2 per seed (OCSD-v2)
logs = [json.load(open(f)) for f in glob.glob(os.path.join(qo, "ocsd_v2", "*.json"))]
assert logs and all(l["cfg"]["blend_mode"] == "joint" for l in logs)
assert any(l.get("repairs") for l in logs), "no repair pass was exercised"
assert all(l["m2_key"] == ("K2_phrase" if l["seed"] == 0 else "K2_phrase_s1") for l in logs)
assert glob.glob(os.path.join(qo, "collage_bo3", "*.png")) and glob.glob(os.path.join(qo, "controlnet_bon", "*.png"))
for det in ("owlv2", "owlv2d", "detr"):
    assert os.path.exists(os.path.join(P.results, "quickdraw", f"det_{det}", "per_image_ocsd_v2.csv")), det
assert "opr50" in pd.read_csv(os.path.join(P.results, "quickdraw", "det_owlv2", "per_image_ocsd_v2.csv"))
tabs = os.path.join(P.results, "tables")
# (stats_quickdraw needs >= 6 scenes shared by all main methods; the tiny tier has 2 - see tests/test_v2.py)
for t in ("E3_quickdraw", "E3power_quickdraw", "prereg_quickdraw", "robustness_quickdraw"):
    assert os.path.exists(os.path.join(tabs, t + ".csv")), t
pr = pd.read_csv(os.path.join(tabs, "prereg_quickdraw.csv"))
from ocsd.report import PREREG     # the 8+ families need >= 6 crowded scenes: the tiny tier has 3, the paper tier 18
assert set(pr.ref) <= {"ocsd_v2", "ocsd"} and set(pr.family) <= {f"{f['kind']}:{f['name']}" for f in PREREG["families"]}
assert {"secondary:all_scenes", "secondary:all_scenes_v2", "secondary:compute_matched"} <= set(pr.family)
# changed tuned settings -> OCSD-family images are archived and generated again, baselines are kept
n_cn = len(os.listdir(os.path.join(P.outputs, "quickdraw", "controlnet")))
tp = os.path.join(P.results, "tuning", "tuned.json"); tj = json.load(open(tp))
tj["chosen"]["eta"] = 7.0; json.dump(tj, open(tp, "w"))
stages.generate(P, E, eng=eng, vis=vis)
assert glob.glob(os.path.join(P.outputs, "_stale", "*", "quickdraw", "ocsd"))
assert not glob.glob(os.path.join(P.outputs, "_stale", "*", "quickdraw", "ocsd_v2"))   # thesis values never reach v2
assert len(os.listdir(os.path.join(P.outputs, "quickdraw", "controlnet"))) == n_cn
assert stages._count_todo(P, [j for j in stages.experiment_plan(P, E) if j["name"] == "E3_quickdraw_trained"][0]) == 0
stages.evaluate(P, E, vis=vis, fid=False)
rows = [l for f in glob.glob(os.path.join(P.results, "quickdraw", "per_image_*.csv")) for l in open(f).readlines()[1:]]
assert rows and not any(s in l for l in rows for s in pilot_sids)

# rows left by a later pilot-tier run (tuning scenes) must not reach the paper tables
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

# a larger power set (power_per_cell) adds scenes to the power job only: the E3 / E4 scenes stay the same
before = {j["name"]: j["scenes"] for j in stages.experiment_plan(P, E)}
build_quickdraw_scenes(qd_raw, qd, per_cell=3, size=128)          # one more scene per cell
assert {j["name"]: j["scenes"] for j in stages.experiment_plan(P, E)} == before
config.TIERS["tiny_paper"]["power_per_cell"] = 2
after = {j["name"]: j["scenes"] for j in stages.experiment_plan(P, E)}
assert all(after[k] == v for k, v in before.items() if k != "E3_quickdraw_power")
assert set(before["E3_quickdraw_power"]) < set(after["E3_quickdraw_power"]) and len(after["E3_quickdraw_power"]) == 24
print("POWER SET OK")
print("TUNE TEST OK")
