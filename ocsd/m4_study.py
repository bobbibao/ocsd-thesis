"""OCSD with harmonised M4: one-run study on SD 1.5 (M1-M5, M3 on), docs/M4_STUDY_PROTOCOL.md.

One Colab "Run all" does every stage in order; every stage is resumable from Drive:
  1. dev      the development arms (methods.M4_ARMS + the original OCSD) on 24 scenes of the paper_v2 TUNING split
  2. select   the pre-registered rule of assets/m4_study_v1/protocol.json picks one arm; its settings are frozen in
              <drive>/m4_study_v1/locked.json (written once, never changed by a later Run all)
  3. paper    the frozen arm ("ocsd_final") on the 72 paper benchmark scenes x seeds 0/1; the original OCSD, GLIGEN,
              Collage, ControlNet and OCSD-v2 images of the earlier locked runs on the same scenes and seeds are imported
              and re-scored with the same evaluator; primary report
  4. extras   T2I-Adapter and Zhang et al. (2025) on the same scenes and seeds; report again with them

Layout under <drive>/m4_study_v1/: dev/ and paper/ (outputs/quickdraw/<arm>/<sid>_s<seed>.png|json, results/),
locked.json, status.json, logs/run.log, logs/LATEST_ERROR.txt.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .provenance import atomic_json, digest, file_hash

ROOT = Path(__file__).resolve().parents[1]
STUDY = "m4_study_v1"
SPLIT = "quickdraw"
FINAL = "ocsd_final"
COMBO = "ocsd_m4_combo"
STRUCT = ("opr", "miou", "ra")


# ---------------------------------------------------------------------------------------------- protocol
def protocol_path(name: str = STUDY) -> Path:
    return ROOT / "assets" / name / "protocol.json"


def dev_overrides(arm: str) -> dict:
    from .methods import VARIANT_OVERRIDES
    return dict(VARIANT_OVERRIDES[arm])


def arm_digest(arms, tier) -> str:
    """Digest of the full configurations (tier + frozen tuning + overrides) of the given OCSD-family arms."""
    from .config import ExperimentConfig
    from .stages import tier_variants
    variants, _ = tier_variants(ExperimentConfig(tier=tier))
    return digest({a: variants[a].to_dict() for a in arms})


def baseline_digest(arms, tier) -> str:
    from .config import ExperimentConfig
    from .stages import tier_baselines, tier_variants
    E = ExperimentConfig(tier=tier)
    variants, _ = tier_variants(E)
    bl = tier_baselines(E)
    return digest({a: (variants[a].to_dict() if a in variants else [bl[a][0], bl[a][1].to_dict()]) for a in arms})


def load_protocol(name: str = STUDY) -> dict:
    """Read the lock and refuse to run when the code's settings differ from it."""
    from .config import BACKBONES
    p = json.loads(protocol_path(name).read_text())
    p["name"] = name
    if p["dev"]["arm_config_sha256"] != arm_digest(p["dev"]["arms"], p["sd15_tier"]):
        raise ValueError("Development arm settings in the code differ from the locked protocol")
    if p["extras"]["config_sha256"] != baseline_digest(p["extras"]["arms"] + p["paper"]["fallback"], p["sd15_tier"]):
        raise ValueError("Baseline settings in the code differ from the locked protocol")
    if p["models"] != {k: BACKBONES["sd15"][k] for k in ("sd", "controlnet", "t2i_adapter", "gligen")}:
        raise ValueError("Model repositories differ from the locked protocol")
    return p


def dev_spec(p, combo: bool = False) -> dict:
    """The development stage as an xl_study-style protocol (so its evaluator, tables and sheets are reused).
    combo: include the combined arm of the second development step."""
    return dict(name=p["name"] + "_dev", title=p["title"] + " - development (tuning split)",
                drive_root=f"{p['drive_root']}/dev", scene_assets=f"assets/{p['name']}/dev",
                scenes=p["dev"]["scenes"], seeds=p["dev"]["seeds"], arms=[],
                sd15_arms=list(p["dev"]["arms"]) + ([COMBO] if combo else []),
                sd15_tier=p["sd15_tier"], reference_arm=p["dev"]["reference"], review_salt=p["review_salt"] + "-dev",
                review=dict(scenes=p["dev"]["scenes"], seeds=p["dev"]["seeds"][:1]))


def paper_spec(p, extras: bool) -> dict:
    q = p["paper"]
    return dict(name=p["name"] + "_paper", title=p["title"] + " - paper benchmark", drive_root=f"{p['drive_root']}/paper",
                scene_assets=q["scene_assets"], scene_import=q["scene_import"], scenes=q["scenes"], seeds=q["seeds"],
                arms=[], sd15_arms=[FINAL] + (list(p["extras"]["arms"]) if extras else []),
                imported_arms=dict(q["imported_arms"]), sd15_fallback=list(q["fallback"]), sd15_tier=p["sd15_tier"],
                reference_arm=q["reference"], review_salt=p["review_salt"], review=q["review"])


# ---------------------------------------------------------------------------------------------- scenes
def import_dev_scenes(drive_root, p, log=print) -> dict:
    """Copy the development scenes from the paper benchmark on Drive. Every scene must be in the frozen tuning split
    (data/benchmarks/tuning_split.json); the copied files are recorded on the first run and must match later."""
    src = Path(drive_root) / p["dev"]["source"]
    split = json.loads((Path(drive_root) / p["dev"]["tuning_split"]).read_text())
    not_tuning = [s for s in p["dev"]["scenes"] if s not in set(split.get("quickdraw", []))]
    if not_tuning:
        raise ValueError("Development scenes outside the tuning split: " + ", ".join(not_tuning))
    dst = ROOT / "assets" / p["name"] / "dev" / SPLIT
    files = {}
    for sid in p["dev"]["scenes"]:
        meta = json.loads((src / sid / "scene.json").read_text())
        names = ["scene.json", "sketch.png", "masks.npz"] + [f"obj_{i:02d}.png" for i in range(meta["n_obj"])]
        (dst / sid).mkdir(parents=True, exist_ok=True)
        for n in names:
            if not (src / sid / n).exists():
                raise FileNotFoundError(f"{sid}/{n} missing in {src}")
            if not (dst / sid / n).exists() or file_hash(dst / sid / n) != file_hash(src / sid / n):
                shutil.copyfile(src / sid / n, dst / sid / n)
            files[f"{SPLIT}/{sid}/{n}"] = file_hash(dst / sid / n)
    record = Path(drive_root) / p["drive_root"] / "dev" / "scene_manifest.json"
    if record.exists():
        if json.loads(record.read_text()) != files:
            raise ValueError("Development scenes on Drive changed since the first run of this study")
    else:
        record.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(record, files)
    log(f"[m4] development scenes: {len(p['dev']['scenes'])} from the tuning split, manifest {digest(files)[:12]}")
    return dict(dataset_id=digest(files), n=len(p["dev"]["scenes"]))


def import_paper_scenes(drive_root, spec, log=print) -> dict:
    from . import xl_study
    ident = xl_study.import_scenes(spec, drive_root)
    log(f"[m4] paper scenes: {len(spec['scenes'])}, same files as the 2026-11-05 run: {ident['same_as_earlier_run']}")
    return ident


# ---------------------------------------------------------------------------------------------- generation
class Models:
    """SD 1.5 engine (with ControlNet and T2I-Adapter) + vision models, loaded once and shared by every stage."""

    def __init__(self, eng=None, vis=None):
        self.eng, self.vis = eng, vis

    def get(self, tier):
        if self.eng is None:
            from .config import ExperimentConfig
            from .engine import Engine
            E = ExperimentConfig(tier=tier)
            self.eng = Engine.from_pretrained(E.backbone, "cuda", E.fp16, cache_dir=os.environ.get("HF_HOME"),
                                              load_adapter=True)
        if self.vis is None:
            from .vision import Vision
            self.vis = Vision(str(self.eng.device), cache_dir=os.environ.get("HF_HOME"))
        return self.eng, self.vis


def configs(tier):
    """(ExperimentConfig, OCSD variants, base config, baselines) of the SD 1.5 methods for a tier (CPU tests patch
    this to shrink every configuration)."""
    from .config import ExperimentConfig
    from .stages import tier_baselines, tier_variants
    E = ExperimentConfig(tier=tier)
    variants, base = tier_variants(E)
    return E, variants, base, tier_baselines(E)


def todo_images(drive_root, spec, arms, missing=()):
    from .xl_study import Dirs
    dirs = Dirs(drive_root, spec)
    want = {(m, sid, s) for m in arms for sid in spec["scenes"] for s in spec["seeds"]}
    want |= {tuple(m) for m in missing if m[0] in spec.get("sd15_fallback", [])}
    return sorted(x for x in want if not (dirs.png(*x).exists() and dirs.png(*x).with_suffix(".json").exists()))


def generate(drive_root, spec, arms, models: Models, missing=(), max_minutes=None, log=print) -> int:
    """Generate the SD 1.5 arms of `spec` with the thesis runner (signatures, M2/M3 caches, resume). Returns how many
    images are still missing afterwards."""
    from .runner import Runner
    from .xl_study import Dirs, scene_dirs
    want = todo_images(drive_root, spec, arms, missing)
    if not want:
        return 0
    log(f"[m4] {spec['name']}: {len(want)} images to generate ({sorted({m for m, _, _ in want})})")
    E, variants, base, baselines = configs(spec["sd15_tier"])
    eng, vis = models.get(spec["sd15_tier"])
    dirs = Dirs(drive_root, spec)
    runner = Runner(eng, vis, str(dirs.outputs), E.backbone, variants=variants, base_cfg=base,
                    baselines=baselines, cache_dir=os.environ.get("HF_HOME"))
    items = [(SPLIT, sdir, sorted((m, s) for m, sid, s in want if sid == Path(sdir).name)) for sdir in scene_dirs(spec)]
    runner.run_items([x for x in items if x[2]], max_minutes=max_minutes)
    return len(todo_images(drive_root, spec, arms, missing))


# ---------------------------------------------------------------------------------------------- score reuse
def _eval_identity(dirs, arm, sid, seed, sdir):
    """The identity ocsd.metrics.evaluate_images gives this image (same rule, same default settings)."""
    import glob
    from .metrics import evaluation_identity
    png = str(dirs.png(arm, sid, seed))
    extra = [png[:-4] + ".json"]
    if os.path.isfile(extra[0]):
        k = json.load(open(extra[0])).get("m2_key")
        if k:
            extra += glob.glob(os.path.join(str(dirs.outputs), SPLIT, "_objects", k, sid, "*"))
    return evaluation_identity(png, sdir, ("owlv2", 0.3, 0.1), extra)


def reuse_scores(drive_root, spec, src, log=print) -> int:
    """Scores of imported images from the earlier locked run that produced them (same evaluator code and models):
    a row is reused only when the image here has the same SHA-256 as the image that run scored; everything else is
    scored again by xl_study.evaluate. Writes nothing for an arm that already has its own table here."""
    import pandas as pd
    from .xl_study import Dirs, scene_dirs
    dirs = Dirs(drive_root, spec)
    srcd = Path(drive_root) / src / SPLIT
    ex_src = srcd / "extra_metrics.csv"
    if not ex_src.exists():
        log(f"[m4] no earlier scores under {srcd}; every image is scored here")
        return 0
    extra = pd.read_csv(ex_src)
    sdirs = {Path(d).name: d for d in scene_dirs(spec)}
    (dirs.results / SPLIT).mkdir(parents=True, exist_ok=True)
    n, extra_rows = 0, []
    for arm in spec.get("imported_arms", {}):
        f, dst = srcd / f"per_image_{arm}.csv", dirs.results / SPLIT / f"per_image_{arm}.csv"
        if dst.exists() or not f.exists():
            continue
        keep = []
        for r in pd.read_csv(f).to_dict("records"):
            sid, seed = r["sid"], int(r["seed"])
            if sid not in sdirs or seed not in spec["seeds"] or not dirs.png(arm, sid, seed).exists():
                continue
            e = extra[(extra.method == arm) & (extra.sid == sid) & (extra.seed == seed)]
            if len(e) != 1 or e.iloc[0]["png_sha256"] != file_hash(dirs.png(arm, sid, seed)) \
                    or pd.isna(e.iloc[0].get("pickscore")):
                continue
            r["input_id"] = _eval_identity(dirs, arm, sid, seed, sdirs[sid])
            keep.append(r)
            extra_rows.append(e.iloc[0].to_dict())
        if keep:
            pd.DataFrame(keep).to_csv(dst, index=False)
            n += len(keep)
    if extra_rows:
        mine = dirs.results / SPLIT / "extra_metrics.csv"
        old = pd.read_csv(mine) if mine.exists() else pd.DataFrame(columns=["method", "sid", "seed"])
        have = {(r["method"], r["sid"], int(r["seed"])) for r in old.to_dict("records")}
        add = [r for r in extra_rows if (r["method"], r["sid"], int(r["seed"])) not in have]
        pd.concat([old, pd.DataFrame(add)], ignore_index=True).to_csv(mine, index=False)
    log(f"[m4] reused {n} earlier scores of imported images (identical files); the rest is scored here")
    return n


# ---------------------------------------------------------------------------------------------- tables
def scene_means(df, arm, metric):
    d = df[df.method == arm]
    return d.groupby("sid")[metric].mean()


def arm_means(df, arms, metrics):
    out = {}
    for a in arms:
        r = {}
        for k in metrics:
            if k in df:
                v = scene_means(df, a, k).to_numpy(dtype=float)
                v = v[~np.isnan(v)]
                r[k] = float(v.mean()) if len(v) else float("nan")
        r["images"] = int((df.method == a).sum())
        out[a] = r
    return out


# ---------------------------------------------------------------------------------------------- selection
def score_arms(df, p, arms, expected):
    """The pre-registered score of every arm in `arms` (see the protocol's selection text) and the means it uses."""
    rule, ref = p["dev"]["selection"], p["dev"]["reference"]
    means = arm_means(df, [ref] + list(arms), list(STRUCT) + ["pickscore", "clip", "count_acc", "sketch_chamfer"])
    # PickScore is part of the rule; if it could not be computed for some arm, the rule falls back to structure only
    no_pick = any(np.isnan(means[a].get("pickscore", float("nan"))) for a in [ref] + list(arms))
    rows = {}
    for a in arms:
        m = means[a]
        struct = [m[k] for k in STRUCT if not np.isnan(m.get(k, float("nan")))]
        sc = float(np.mean(struct)) if struct else float("-inf")
        if not no_pick:
            sc += rule["pickscore_weight"] * (m["pickscore"] - means[ref]["pickscore"])
        complete = m["images"] >= rule["min_complete"] * expected
        natural = True if no_pick else m["pickscore"] >= means[ref]["pickscore"] + rule["min_pickscore_gain"]
        rows[a] = dict(means=m, score=sc, complete=bool(complete), not_less_natural=bool(natural),
                       eligible=bool(complete and natural))
    return rows, means, no_pick


def plan_combo(drive_root, p, log=print):
    """Second development step (pre-registered): every one-factor arm that is eligible and scores above the centre
    contributes its change (alpha: only the better of the two alpha arms); when at least two changes qualify, their
    union is the combined arm, evaluated on the same development scenes and seeds. Written once to dev/combo.json."""
    from .xl_study import Dirs, load_table
    path = Path(drive_root) / p["drive_root"] / "dev" / "combo.json"
    if path.exists():
        return json.loads(path.read_text())
    spec = dev_spec(p)
    df = load_table(Dirs(drive_root, spec), spec)
    c = p["dev"]["combo"]
    rows, _, _ = score_arms(df, p, p["dev"]["selection"]["candidates"], len(spec["scenes"]) * len(spec["seeds"]))
    centre = rows[c["centre"]]["score"]
    picked = []
    for group in c["factor_groups"]:
        good = [a for a in group if rows[a]["eligible"] and rows[a]["score"] > centre]
        if good:
            picked.append(max(good, key=lambda a: rows[a]["score"]))
    over = None
    if len(picked) >= 2:
        over = dev_overrides(c["centre"])
        for a in picked:
            over.update({k: v for k, v in c["factor_overrides"][a].items()})
    plan = dict(centre=c["centre"], centre_score=centre, factors=picked, overrides=over,
                scores={a: rows[a]["score"] for a in rows}, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(path, _jsonable(plan))
    log(f"[m4] combined arm: {'from ' + ', '.join(picked) if over else 'not needed (fewer than two improving changes)'}")
    return plan


def register_combo(plan) -> bool:
    if plan and plan.get("overrides"):
        from .methods import register_variant
        register_variant(COMBO, plan["overrides"])
        return True
    return False


def select(drive_root, p, log=print) -> dict:
    """Apply the pre-registered selection rule to the development tables; write locked.json once."""
    from .xl_study import Dirs, load_table
    lock = Path(drive_root) / p["drive_root"] / "locked.json"
    if lock.exists():
        L = json.loads(lock.read_text())
        if L["protocol_sha256"] != file_hash(protocol_path(p["name"])):
            raise ValueError("locked.json was written under a different protocol file")
        log(f"[m4] using the frozen selection: {L['chosen']} ({L['time']})")
        return L
    plan_path = Path(drive_root) / p["drive_root"] / "dev" / "combo.json"
    has_combo = register_combo(json.loads(plan_path.read_text())) if plan_path.exists() else False
    spec = dev_spec(p, combo=has_combo)
    df = load_table(Dirs(drive_root, spec), spec)
    rule = p["dev"]["selection"]
    ref = p["dev"]["reference"]
    expected = len(spec["scenes"]) * len(spec["seeds"])
    candidates = list(rule["candidates"]) + ([COMBO] if has_combo else [])
    rows, means, no_pick = score_arms(df, p, candidates, expected)
    eligible = [a for a in rows if rows[a]["eligible"]] or [a for a in rows if rows[a]["complete"]]
    if not eligible:
        raise RuntimeError("No development arm has enough images to be selected")
    chosen = max(eligible, key=lambda a: (rows[a]["score"], -candidates.index(a)))
    cfg = configs(p["sd15_tier"])[1][chosen]
    L = dict(chosen=chosen, overrides=dev_overrides(chosen), config=cfg.to_dict(), config_sha256=digest(cfg.to_dict()),
             fallback_rule_used=not any(rows[a]["eligible"] for a in rows), pickscore_missing=bool(no_pick),
             reference=ref, reference_means=means[ref], arms=rows, rule=rule, combo=has_combo,
             protocol_sha256=file_hash(protocol_path(p["name"])), dev_images=int(len(df)),
             time=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(lock, _jsonable(L))
    lines = ["# M4 study: development selection", "", f"Chosen: **{chosen}** (frozen {L['time']})", "",
             "| arm | OPR | mIoU | RA | PickScore | score | eligible |", "|---|---|---|---|---|---|---|"]
    for a in [ref] + candidates:
        m = means[a]
        r = rows.get(a, {})
        lines.append(f"| {a} | {m['opr']:.3f} | {m['miou']:.3f} | {m.get('ra', float('nan')):.3f} | "
                     f"{m['pickscore']:.2f} | {r.get('score', float('nan')):.3f} | {r.get('eligible', 'reference')} |")
    (Path(drive_root) / p["drive_root"] / "SELECTION.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return L


def register_final(L) -> None:
    from .methods import register_variant
    register_variant(FINAL, L["overrides"])


# ---------------------------------------------------------------------------------------------- statistics
def sign_test(d):
    from math import comb
    w, l = int((d > 0).sum()), int((d < 0).sum())
    n = w + l
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.0
    return w, l, len(d) - n, p


def wilcoxon_p(d):
    d = d[d != 0]
    if len(d) < 6:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return (float("nan"),) * 3
    m = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def paired(df, a, b, metric, better=1):
    t = df[df.method.isin([a, b])].groupby(["sid", "method"])[metric].mean().unstack("method")
    if a not in t or b not in t:
        return None
    t = t[[a, b]].dropna()
    d = (t[a] - t[b]).to_numpy(dtype=float)
    if not len(d):
        return None
    w, l, ties, p = sign_test(better * d)
    mu, lo, hi = boot(d)
    return dict(a=a, b=b, metric=metric, scenes=len(d), mean_a=float(t[a].mean()), mean_b=float(t[b].mean()),
                diff=mu, lo=lo, hi=hi, wins=w, losses=l, ties=ties, p_sign=p, p_wilcoxon=wilcoxon_p(better * d),
                favours=a if better * mu > 0 else (b if better * mu < 0 else "tie"))


# ---------------------------------------------------------------------------------------------- report
def report(drive_root, p, spec, log=print) -> dict:
    """Tables with bootstrap CIs, by object count, the pre-registered tests, blind sheets and two ZIPs."""
    import pandas as pd
    from .xl_study import METRICS, Dirs, all_arms, load_table, make_review, review_scenes, review_seeds
    dirs = Dirs(drive_root, spec)
    df = load_table(dirs, spec)
    arms = [a for a in all_arms(spec) if (df.method == a).any()]
    expected = len(spec["scenes"]) * len(spec["seeds"])
    metrics = [m for m in METRICS if m in df and m != "time_s"]
    rows = []
    for a in arms:
        d = df[df.method == a]
        r = dict(method=a, images=len(d), complete=len(d) == expected)
        for k in metrics:
            r[k], r[k + "_lo"], r[k + "_hi"] = boot(d.groupby("sid")[k].mean().to_numpy(dtype=float))
        rows.append(r)
    tests = []
    is_paper = spec["name"].endswith("_paper")
    if is_paper:
        prim = p["paper"]["primary"]
        res = [paired(df, t["a"], t["b"], t["metric"], t["better"]) for t in prim]
        ok = [r for r in res if r is not None]
        adj = holm([r["p_sign"] for r in ok]) if ok else []
        for t, r, q in zip(prim, ok, adj):
            r.update(id=t["id"], family="primary", p_holm=q)
            tests.append(r)
        sec = []
        for b in p["paper"]["secondary_vs"]:
            for k, better in p["paper"]["secondary_metrics"]:
                if (df.method == b).any() and k in df:
                    r = paired(df, FINAL, b, k, better)
                    if r is not None:
                        r.update(family="secondary")
                        sec.append(r)
        adj = holm([r["p_sign"] for r in sec]) if sec else []
        for r, q in zip(sec, adj):
            r["p_holm_secondary"] = q
        tests += sec
    else:
        ref = spec["reference_arm"]
        for a in arms:
            if a != ref:
                for k, better in (("opr", 1), ("miou", 1), ("ra", 1), ("pickscore", 1), ("sketch_chamfer", -1)):
                    if k in df:
                        r = paired(df, a, ref, k, better)
                        if r is not None:
                            r.update(family="development (not a result)")
                            tests.append(r)
    by_bin = df.groupby(["method", "count_bin"])[[k for k in ("opr", "miou", "ra", "count_acc", "pickscore") if k in df]] \
        .mean().reset_index()
    tables = dirs.results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables / "summary.csv", index=False)
    by_bin.to_csv(tables / "by_count_bin.csv", index=False)
    pd.DataFrame(tests).to_csv(tables / "tests.csv", index=False)
    df.to_csv(tables / "per_image.csv", index=False)
    key = make_review(dirs, spec)
    from .methods import LABELS
    lines = [f"# {spec['title']}", "",
             f"Protocol: assets/{p['name']}/protocol.json (docs/M4_STUDY_PROTOCOL.md). {len(spec['scenes'])} scenes x seeds "
             f"{spec['seeds']}; per-scene values averaged over seeds; means with 95% bootstrap CIs over scenes. "
             "SD 1.5 at 512 px for every arm. sketch_chamfer: lower is better.", ""]
    if is_paper:
        L = json.loads((Path(drive_root) / p["drive_root"] / "locked.json").read_text())
        lines += [f"`{FINAL}` = development arm `{L['chosen']}` frozen on {L['time']} (config {L['config_sha256'][:12]}).", ""]
    lines += ["| arm | images | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
    for r in rows:
        lines.append(f"| {r['method']} | {r['images']}/{expected} | " + " | ".join(
            f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]" for k in metrics) + " |")
    lines += ["", "Arms: " + "; ".join(f"`{a}` {LABELS.get(a, '')}" for a in arms), "",
              "## By object count", "", by_bin.round(3).to_markdown(index=False), "", "## Paired tests", "",
              "| family | id | a | b | metric | mean a | mean b | diff [95% CI] | W/L/T | p sign | p Holm | p Wilcoxon |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in tests:
        q = r.get("p_holm", r.get("p_holm_secondary", float("nan")))
        lines.append(f"| {r['family']} | {r.get('id', '')} | {r['a']} | {r['b']} | {r['metric']} | {r['mean_a']:.3f} | "
                     f"{r['mean_b']:.3f} | {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['wins']}/{r['losses']}/"
                     f"{r['ties']} | {r['p_sign']:.4f} | {q:.4f} | {r['p_wilcoxon']:.4f} |")
    if is_paper:
        lines += ["", "## Pre-registered claims", ""] + [f"- {c}" for c in p["paper"]["claims"]] + ["", "## Verdict", ""]
        lines += verdict(p, tests)
    (dirs.results / "summary.md").write_text("\n".join(lines) + "\n")
    atomic_json(dirs.results / "summary.json", _jsonable(dict(study=p["name"], stage=spec["name"], summary=rows,
                                                              tests=tests, by_count_bin=by_bin.to_dict("records"))))
    bundle, review_zip = dirs.results / f"ocsd_{spec['name']}_results.zip", dirs.results / f"REVIEW_{spec['name']}.zip"
    with ZipFile(bundle, "w", ZIP_DEFLATED) as z:
        for path in sorted(dirs.results.rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".json", ".md", ".txt", ".log") and "review" not in path.parts:
                z.write(path, path.relative_to(dirs.results).as_posix())
        for extra in ("locked.json", "SELECTION.md"):
            f = Path(drive_root) / p["drive_root"] / extra
            if f.exists():
                z.write(f, extra)
        for path in sorted((dirs.outputs / SPLIT).glob("*/*.json")):
            z.write(path, "outputs/" + path.relative_to(dirs.outputs / SPLIT).as_posix())
    with ZipFile(review_zip, "w", ZIP_DEFLATED) as z:
        for path in sorted((dirs.results / "review").iterdir()):
            z.write(path, "review/" + path.name)
        z.write(dirs.results / "review_key.json", "review_key.json.DO_NOT_OPEN_BEFORE_RATING")
        shown = {f"{sid}_s{seed}" for sid in review_scenes(spec) for seed in review_seeds(spec)}
        for path in sorted((dirs.outputs / SPLIT).glob(f"{FINAL}/*.png")):
            if path.stem in shown:
                z.write(path, f"images/{FINAL}/{path.name}")
    log(f"[m4] {spec['name']}: {dirs.results / 'summary.md'}\n[m4] results ZIP {bundle}\n[m4] review ZIP {review_zip}")
    return dict(summary=str(dirs.results / "summary.md"), archive=str(bundle), review_archive=str(review_zip),
                sheets=len(key))


def verdict(p, tests):
    """Apply the claim rules of the protocol to the primary tests (automatic metrics only)."""
    prim = {t["id"]: t for t in tests if t.get("family") == "primary"}
    out = []
    for c in p["paper"]["claim_rules"]:
        need = [prim.get(i) for i in c["significant_any"]]
        guard = [prim.get(i) for i in c.get("not_worse", [])]
        sig = [t for t in need if t and t["p_holm"] < 0.05 and t["favours"] == FINAL]
        worse = [t for t in guard if t and t["p_holm"] < 0.05 and t["favours"] != FINAL]
        holds = bool(sig) and not worse
        out.append(f"- **{c['id']}** ({c['text']}): {'HOLDS' if holds else 'does not hold'} "
                   f"(significant in favour: {[t['id'] for t in sig]}; significantly worse: {[t['id'] for t in worse]})")
    return out


def _jsonable(o):
    if isinstance(o, dict): return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, float) and o != o: return None
    return o


# ---------------------------------------------------------------------------------------------- status
def _status(drive_root, p, **kw):
    path = Path(drive_root) / p["drive_root"] / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(path, _jsonable(state))
    return state


def stage_left(drive_root, name: str = STUDY, p: dict = None) -> dict:
    """What is left on Drive (no model loaded), for the notebook's first cell."""
    if p is None:
        p = json.loads(protocol_path(name).read_text())
        p["name"] = name
    root = Path(drive_root) / p["drive_root"]
    dev = dev_spec(p)
    out = dict(locked=(root / "locked.json").exists())
    out["dev_images_left"] = sum(not (root / "dev" / "outputs" / SPLIT / a / f"{s}_s{sd}.png").exists()
                                 for a in dev["sd15_arms"] for s in dev["scenes"] for sd in dev["seeds"])
    q = p["paper"]
    out["paper_images_left"] = sum(not (root / "paper" / "outputs" / SPLIT / FINAL / f"{s}_s{sd}.png").exists()
                                   for s in q["scenes"] for sd in q["seeds"])
    out["extras_images_left"] = sum(not (root / "paper" / "outputs" / SPLIT / a / f"{s}_s{sd}.png").exists()
                                    for a in p["extras"]["arms"] for s in q["scenes"] for sd in q["seeds"])
    return out


# ---------------------------------------------------------------------------------------------- entry point
def main(drive_root, name: str = STUDY, max_hours: float = 11.0, run_extras: bool = True, models: Models = None,
         p: dict = None, log=print):
    """Every stage in order; each one resumes from Drive. Stops cleanly when the time budget is used up (Run all
    again to continue). Returns a dict with the paths of the reports that exist. `p`: an already loaded protocol
    (CPU tests pass a cut-down copy)."""
    from .runlog import _Tee
    import gc
    p = p or load_protocol(name)
    root = Path(drive_root) / p["drive_root"]
    (root / "logs").mkdir(parents=True, exist_ok=True)
    log_file = (root / "logs" / "run.log").open("a")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, log_file), _Tee(old_err, log_file)
    from . import runlog
    if not runlog._STATE.get("log_dir"):       # per-scene errors of the runner go to <study>/logs/errors/
        (root / "logs" / "errors").mkdir(parents=True, exist_ok=True)
        runlog._STATE["log_dir"], runlog._STATE["log_path"] = str(root / "logs"), str(root / "logs" / "run.log")
    t0 = time.time()
    left = lambda: max_hours * 60 - (time.time() - t0) / 60
    models = models or Models()
    out = {}
    try:
        print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {name} =====", flush=True)
        try:
            import subprocess
            head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            head = None
        _status(drive_root, p, code=head, error=None, stage="start")
        import torch
        from . import xl_study
        # ---- 1. development
        dev = dev_spec(p)
        dev_id = import_dev_scenes(drive_root, p)
        lock = root / "locked.json"
        if not lock.exists():
            _status(drive_root, p, stage="dev_generate", dev_scenes=dev_id)
            n_left = generate(drive_root, dev, dev["sd15_arms"], models, max_minutes=left())
            if n_left and left() < 2:
                _status(drive_root, p, stage="paused", dev_images_left=n_left)
                print(f"[m4] {n_left} development images left: time budget used up. Run all again to continue.")
                return out
            if n_left:      # images that failed (see results/logs/LATEST_ERROR.txt); the rule needs 95% per arm
                print(f"[m4] WARNING: {n_left} development images failed; selecting among the arms that are complete")
                _status(drive_root, p, dev_failed_images=n_left)
            _status(drive_root, p, stage="dev_evaluate")
            models.get(p["sd15_tier"])
            xl_study.evaluate(drive_root, dev, vis=models.vis, picker=_picker(models))
            # second development step: the combined arm (planned once from the one-factor results)
            if register_combo(plan_combo(drive_root, p)):
                dev = dev_spec(p, combo=True)
                _status(drive_root, p, stage="dev_combo")
                n_left = generate(drive_root, dev, [COMBO], models, max_minutes=left())
                if n_left and left() < 2:
                    _status(drive_root, p, stage="paused", dev_images_left=n_left)
                    print(f"[m4] {n_left} development images left: time budget used up. Run all again to continue.")
                    return out
                xl_study.evaluate(drive_root, dev, vis=models.vis, picker=_picker(models))
            out["dev"] = report(drive_root, p, dev)
        # ---- 2. selection (frozen once)
        L = select(drive_root, p)
        register_final(L)
        out["chosen"] = L["chosen"]
        _status(drive_root, p, stage="selected", chosen=L["chosen"], config_sha256=L["config_sha256"])
        # ---- 3. paper benchmark: the frozen arm + imported earlier images
        spec = paper_spec(p, extras=False)
        ident = import_paper_scenes(drive_root, spec)
        _status(drive_root, p, stage="paper_generate", paper_scenes=ident)
        missing = xl_study.import_legacy(drive_root, spec)
        n_left = generate(drive_root, spec, [FINAL], models, missing=missing, max_minutes=left())
        if n_left and left() < 2:
            _status(drive_root, p, stage="paused", paper_images_left=n_left)
            print(f"[m4] {n_left} paper images left: time budget used up. Run all again to continue.")
            return out
        if n_left:
            print(f"[m4] WARNING: {n_left} paper images failed (see the logs); the report shows the image counts")
            _status(drive_root, p, paper_failed_images=n_left)
        _status(drive_root, p, stage="paper_evaluate")
        if p["paper"].get("reuse_scores_from"):
            reuse_scores(drive_root, spec, p["paper"]["reuse_scores_from"])
        models.get(p["sd15_tier"])
        picker = _picker(models)
        xl_study.evaluate(drive_root, spec, vis=models.vis, picker=picker)
        out["paper"] = report(drive_root, p, spec)
        print(open(out["paper"]["summary"]).read(), flush=True)
        # ---- 4. extra baselines on the same scenes and seeds
        if run_extras:
            spec = paper_spec(p, extras=True)
            if left() < 20:
                print("[m4] primary results are complete; extras skipped for lack of time (Run all again to add them).")
            else:
                _status(drive_root, p, stage="extras_generate")
                n_left = generate(drive_root, spec, p["extras"]["arms"], models, max_minutes=left())
                xl_study.evaluate(drive_root, spec, vis=models.vis, picker=picker)
                out["paper"] = report(drive_root, p, spec)
                if n_left:
                    print(f"[m4] {n_left} extra baseline images left; Run all again to complete them.")
        _status(drive_root, p, stage="done" if not run_extras or not stage_left(drive_root, name, p)["extras_images_left"]
                else "primary_done", minutes=(time.time() - t0) / 60)
        print(f"[m4] finished in {(time.time() - t0) / 60:.0f} min", flush=True)
        return out
    except Exception as exc:
        _status(drive_root, p, stage="failed", error=repr(exc), traceback=traceback.format_exc())
        (root / "logs" / "LATEST_ERROR.txt").write_text(traceback.format_exc())
        raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log_file.close()
        gc.collect()


def _picker(models: Models):
    if getattr(models, "picker", None) is None:
        try:
            import torch
            from .xl_study import PickScore
            models.picker = PickScore("cuda" if torch.cuda.is_available() else "cpu", cache_dir=os.environ.get("HF_HOME"))
        except Exception as exc:
            print(f"[m4] PickScore could not be loaded ({exc!r}); the column stays empty", flush=True)
            models.picker = None
    return models.picker


# ---------------------------------------------------------------------------------------------- preflight
def preflight(drive_root, name: str = STUDY) -> dict:
    """Checks that need no model and no GPU, run before anything is loaded: the lock matches the code, the inputs on
    Drive exist, and what is left to do. Raises with a readable message when the run could not complete."""
    p = load_protocol(name)
    root = Path(drive_root)
    problems, notes = [], []
    split = root / p["dev"]["tuning_split"]
    if not split.exists():
        problems.append(f"missing {split}")
    else:
        tun = set(json.loads(split.read_text()).get("quickdraw", []))
        bad = [s for s in p["dev"]["scenes"] if s not in tun]
        if bad:
            problems.append("development scenes outside the tuning split: " + ", ".join(bad))
    src = root / p["dev"]["source"]
    miss = [s for s in p["dev"]["scenes"] + p["paper"]["scenes"] if not (src / s / "scene.json").exists()]
    if miss:
        problems.append(f"{len(miss)} scenes not found under {src}: " + ", ".join(miss[:6]))
    for arm, folder in p["paper"]["imported_arms"].items():
        n = sum((root / folder / f"{s}_s{sd}.png").exists() for s in p["paper"]["scenes"] for sd in p["paper"]["seeds"])
        want = len(p["paper"]["scenes"]) * len(p["paper"]["seeds"])
        if n < want:
            notes.append(f"{arm}: {n}/{want} earlier images on Drive; the other {want - n} are generated again")
    left = stage_left(drive_root, name)
    if problems:
        raise RuntimeError("Preflight failed:\n  - " + "\n  - ".join(problems))
    return dict(left=left, notes=notes)
