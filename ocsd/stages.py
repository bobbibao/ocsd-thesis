"""Các giai đoạn của toàn bộ thực nghiệm, gọi từ notebook RUN_ALL. Mọi giai đoạn đều có thể chạy lại để tiếp tục.

  1. setup_data      tải QuickDraw + COCO val2017, dựng QuickDraw-Scenes và COCO-Sketch
  2. generate        sinh ảnh cho E3 (hai bộ dữ liệu), E4 (cắt bỏ) và khảo sát alpha
  3. evaluate        độ đo từng ảnh + FID/KID/LPIPS
  4. report          bảng (CSV/MD/LaTeX), hình, kiểm định, summary.json / summary.md cho khóa luận
"""
from __future__ import annotations

import glob
import json
import os
import platform
import subprocess
import time
from typing import Dict, List, Optional

from .config import TIERS, ExperimentConfig, Paths
from .data import (COCO_URLS, build_coco_sketch, build_quickdraw_scenes, download_quickdraw, list_scenes)
from .methods import BASELINES, OCSD_VARIANTS
from .report import ABLATION, ABLATION_V2, ABLATION_V2_COCO, ALPHAS, MAIN, PARETO_METHODS


def _sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


# ============================================================================ 1. data
def setup_data(P: Paths, tier: str = "paper", with_coco: bool = True, force: bool = False):
    """Build benchmarks for `tier`. Skipped entirely when benchmarks/.done_<tier>.json exists (use force=True to redo)."""
    from .runlog import mark_skipped
    P.makedirs()
    t = TIERS[tier]
    marker = os.path.join(P.benchmarks, f".done_{tier}.json")
    if os.path.exists(marker) and not force:
        done = json.load(open(marker))
        need = max(t["qd_per_cell"], t.get("power_per_cell", 0))
        if done.get("qd_per_cell", t["qd_per_cell"]) >= need:
            mark_skipped("A_data", f"benchmarks for tier '{tier}' already built ({marker})")
            return done
        # a larger power set: only the extra QuickDraw scenes are built; existing scenes and COCO are kept
        extra_qd = _max_per_cell(P, freeze_tuning_split(P).get("quickdraw", [])) if t.get("tune") else 0
        download_quickdraw(P.quickdraw_raw, per_class=3000)
        build_quickdraw_scenes(P.quickdraw_raw, os.path.join(P.benchmarks, "quickdraw"), per_cell=need + extra_qd)
        done.update(qd_per_cell=need, counts=preview_benchmark(P), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(done, open(marker, "w"), indent=1)
        return done
    extra_qd, extra_coco = 0, 0
    if t.get("tune"):
        # the pilot scenes become the tuning split; build extra scenes so the evaluation keeps its full size
        split = freeze_tuning_split(P)
        extra_qd = _max_per_cell(P, split.get("quickdraw", []))
        extra_coco = 4 * _max_per_cell(P, split.get("coco", []))
    download_quickdraw(P.quickdraw_raw, per_class=3000)
    build_quickdraw_scenes(P.quickdraw_raw, os.path.join(P.benchmarks, "quickdraw"),
                           per_cell=max(t["qd_per_cell"], t.get("power_per_cell", 0)) + extra_qd)
    if with_coco:
        local = "/content/coco"   # giải nén ra đĩa cục bộ (nhanh hơn Drive), chỉ lưu kết quả lên Drive
        os.makedirs(local, exist_ok=True)
        if not os.path.exists(f"{local}/annotations/instances_val2017.json"):
            _sh(f"wget -q -c {COCO_URLS['ann']} -O {local}/ann.zip && unzip -q -o {local}/ann.zip -d {local}")
        if not os.path.exists(f"{local}/val2017"):
            _sh(f"wget -q -c {COCO_URLS['val']} -O {local}/val.zip && unzip -q -o {local}/val.zip -d {local}")
        build_coco_sketch(local, os.path.join(P.benchmarks, "coco"), n_scenes=t["coco_n"] + extra_coco,
                          ref_dir=os.path.join(P.data, "coco_ref"), n_ref=2000)
    counts = preview_benchmark(P)
    json.dump(dict(tier=tier, counts=counts, qd_per_cell=max(t["qd_per_cell"], t.get("power_per_cell", 0)),
                   time=time.strftime("%Y-%m-%d %H:%M:%S")), open(marker, "w"), indent=1)
    return counts


def preview_benchmark(P: Paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .data import load_scene
    rows = [("1", "simple"), ("3", "medium"), ("5", "complex"), ("8+", "complex")]
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    cells = [(cb, cx) for cb in ("1", "3", "5", "8+") for cx in ("simple", "medium", "complex")]
    for ax, (cb, cx) in zip(axes.T.flat, cells):
        ds = list_scenes(P.benchmarks, "quickdraw", [cb], [cx], limit=1)
        ax.axis("off")
        if ds:
            sc = load_scene(ds[0])
            ax.imshow(sc.sketch, cmap="gray")
            ax.set_title(f"{cb} đối tượng / {cx}\n{sc.caption[:70]}", fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.join(P.results, "figures"), exist_ok=True)
    fig.savefig(os.path.join(P.results, "figures", "benchmark_examples.png"), dpi=150)
    counts = {}
    for d in glob.glob(os.path.join(P.benchmarks, "*", "*", "scene.json")):
        m = json.load(open(d))
        k = f"{m['split']}|{m['count_bin']}|{m['complexity']}"
        counts[k] = counts.get(k, 0) + 1
    json.dump(counts, open(os.path.join(P.results, "benchmark_counts.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps(counts, indent=1, sort_keys=True))
    return counts


# ============================================================================ tuning split
def _tuning_split_path(P: Paths) -> str:
    return os.path.join(P.benchmarks, "tuning_split.json")


def freeze_tuning_split(P: Paths) -> Dict[str, List[str]]:
    """The scenes that exist before the paper/full benchmark is built (the pilot scenes) become the tuning split.
    Written once and never changed, so the evaluation scenes stay disjoint from the scenes used for tuning."""
    p = _tuning_split_path(P)
    if os.path.exists(p):
        return json.load(open(p))
    split = {sp: sorted(os.path.basename(d) for d in list_scenes(P.benchmarks, sp)) for sp in ("quickdraw", "coco")}
    split["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(split, open(p, "w"), indent=1)
    # keep the pilot's metrics and tables apart, so the paper tables only contain the new evaluation scenes
    arch = os.path.join(P.results, "pilot")
    for name in ("quickdraw", "coco", "tables", "figures", "user_study", "summary.md", "summary.json",
                 "progress.json", "experiment_config.json"):
        src = os.path.join(P.results, name)
        if os.path.exists(src) and not os.path.exists(os.path.join(arch, name)):
            os.makedirs(arch, exist_ok=True)
            os.rename(src, os.path.join(arch, name))
    print(f"[tune] pilot results moved to {arch}")
    print(f"[tune] froze tuning split: {len(split['quickdraw'])} QuickDraw + {len(split['coco'])} COCO scenes")
    return split


def _max_per_cell(P: Paths, sids: List[str]) -> int:
    cells: Dict[tuple, int] = {}
    for sid in sids:
        for sp in ("quickdraw", "coco"):
            f = os.path.join(P.benchmarks, sp, sid, "scene.json")
            if os.path.exists(f):
                m = json.load(open(f))
                cells[(m["count_bin"], m["complexity"])] = cells.get((m["count_bin"], m["complexity"]), 0) + 1
    return max(cells.values()) if cells else 0


def excluded_scenes(P: Paths, E: ExperimentConfig) -> set:
    if not TIERS[E.tier].get("tune") or not os.path.exists(_tuning_split_path(P)):
        return set()
    s = json.load(open(_tuning_split_path(P)))
    return set(s.get("quickdraw", [])) | set(s.get("coco", []))


def _tuned_path(P: Paths) -> str:
    return os.path.join(P.results, "tuning", "tuned.json")


TUNE_PHASES = ("alpha_lora", "energy", "m5_alpha", "v2", "baselines")


def tune_phases(E: ExperimentConfig) -> List[str]:
    """Phases of stage T for this tier (the baseline phase only with tune_baselines)."""
    return [p for p in TUNE_PHASES if p != "baselines" or TIERS[E.tier].get("tune_baselines")]


def _phases_done(tuned: Optional[Dict]) -> List[str]:
    """Finished tuning phases; tuned.json files written before phases were recorded are read from their keys."""
    if not tuned:
        return []
    if "phases" in tuned:
        return list(tuned["phases"])
    c = tuned.get("chosen", {})
    return [p for p, k in zip(TUNE_PHASES, ("alpha", "use_energy", "use_region_attn")) if k in c]


def _load_tuned(P: Optional[Paths], E: ExperimentConfig) -> Dict:
    if P is None or not TIERS[E.tier].get("tune") or not os.path.exists(_tuned_path(P)):
        return {}
    return json.load(open(_tuned_path(P)))


def tuning_finished(P: Paths, E: ExperimentConfig) -> bool:
    return set(_phases_done(_load_tuned(P, E) or None)) >= set(tune_phases(E))


def _legacy_ocsd(E: ExperimentConfig):
    """OCSD with the tier settings and the pre-freeze defaults: the starting point of tuning phases 1-2."""
    from .config import LEGACY_DEFAULTS, OCSDConfig
    return OCSDConfig(**LEGACY_DEFAULTS).replace(**TIERS[E.tier].get("cfg", {}))


def _tune_variants(E: ExperimentConfig):
    from .config import TUNE_GRID
    base = _legacy_ocsd(E)
    return {f"tune_a{a:.1f}_l{l:.1f}": base.replace(alpha=a, lora_scale=l)
            for a in TUNE_GRID["alpha"] for l in TUNE_GRID["lora_scale"]}


def _tune_energy_variants(E: ExperimentConfig, chosen: Dict):
    from .config import TUNE_ENERGY
    base = _legacy_ocsd(E).replace(alpha=chosen["alpha"], lora_scale=chosen["lora_scale"])
    return {f"tune_a{chosen['alpha']:.1f}_l{chosen['lora_scale']:.1f}_{k}": base.replace(**v)
            for k, v in TUNE_ENERGY.items()}


def _tune_m5_variants(E: ExperimentConfig, chosen: Dict):
    """Phase 3: {M5(a)+M5(b), M5(a) only, M5(b) only} x alpha, on top of the phase 1-2 choices and the current
    defaults (caption in P_g). Returns (variants, reference name = the phase-2 setting at its alpha)."""
    from .config import OCSDConfig, TUNE_M5, TUNE_M5_ALPHA
    base = OCSDConfig().replace(**TIERS[E.tier].get("cfg", {})).replace(**chosen)
    alphas = sorted(set(TUNE_M5_ALPHA) | {round(float(chosen["alpha"]), 2)})
    var = {f"tune_m5_{k}_a{a:.1f}": base.replace(alpha=a, **v) for k, v in TUNE_M5.items() for a in alphas}
    mode = "both" if chosen.get("use_energy", True) else "region"
    return var, f"tune_m5_{mode}_a{float(chosen['alpha']):.1f}"


def _tune_v2_variants(E: ExperimentConfig):
    """Phase "v2": OCSD-v2 alpha x anchor_shrink (TUNE_V2). Reference: the methods.V2 defaults."""
    from .config import TUNE_V2
    var, _ = tier_variants(E, apply_tuned=False)
    base = var["ocsd_v2"]
    alphas = sorted(set(TUNE_V2["alpha"]) | {base.alpha})
    shrinks = sorted(set(TUNE_V2["anchor_shrink"]) | {base.anchor_shrink})
    grid = {f"tune_v2_a{a:.1f}_s{sh:.1f}": base.replace(alpha=a, anchor_shrink=sh) for a in alphas for sh in shrinks}
    return grid, f"tune_v2_a{base.alpha:.1f}_s{base.anchor_shrink:.1f}"


def _tune_baseline_grid(E: ExperimentConfig, bl: str, field: str, values, tuned_bl: Dict):
    """Phase "baselines", one baseline: (OCSD-family variants, baseline configs, reference name) for its grid; the
    reference is the current value of the knob (always part of the grid)."""
    if bl == "zhang2025":
        z = tier_variants(E, apply_tuned=False)[0]["zhang2025"]
        vals = sorted(set(values) | {getattr(z, field)})
        return ({f"tunebl_{bl}_{field}{v:g}": z.replace(**{field: v}) for v in vals}, {},
                f"tunebl_{bl}_{field}{getattr(z, field):g}")
    kind, cfg = tier_baselines(E, tuned_bl=tuned_bl)[bl]
    vals = sorted(set(values) | {getattr(cfg, field)})
    return ({}, {f"tunebl_{bl}_{field}{v:g}": (kind, cfg.replace(**{field: v})) for v in vals},
            f"tunebl_{bl}_{field}{getattr(cfg, field):g}")


def _tuning_jobs(P: Paths):
    split = json.load(open(_tuning_split_path(P)))
    jobs = []
    for sp, n in (("quickdraw", 12), ("coco", 4)):
        keep = set(split.get(sp, []))
        others = {os.path.basename(d) for d in list_scenes(P.benchmarks, sp)} - keep
        ds = list_scenes(P.benchmarks, sp, limit=n, seed=1, exclude=others)   # stratified sample of the split
        if ds:
            jobs.append((sp, ds))
    if not jobs:
        raise RuntimeError(f"tuning split {_tuning_split_path(P)} is empty: run the pilot tier before this tier")
    return jobs


def _run_grid(P, E, eng, vis, variants, ref, name, max_minutes, baselines=None):
    """Generate + evaluate every variant (OCSD family) / baseline config on the tuning split; return (table, best)."""
    import pandas as pd
    from .config import TUNE_CLIP_TOL, TUNE_SEEDS
    from .metrics import evaluate_images
    from .runner import Runner
    jobs = _tuning_jobs(P)
    baselines = baselines or {}
    names = sorted(variants) + sorted(baselines)
    out_res = os.path.join(P.results, "tuning")
    _, base = tier_variants(E, apply_tuned=False)
    runner = Runner(eng, vis, P.outputs, E.backbone, variants=variants, base_cfg=base, cache_dir=P.cache,
                    baselines=baselines)
    todo = [(sp, d, [(m, s) for m in names for s in TUNE_SEEDS
                     if not os.path.exists(runner.img_path(sp, m, os.path.basename(d), s))])
            for sp, ds in jobs for d in ds]
    todo = [x for x in todo if x[2]]
    print(f"[tune:{name}] {len(names)} settings, {sum(len(ds) for _, ds in jobs)} scenes, seeds {TUNE_SEEDS}: "
          f"{sum(len(x[2]) for x in todo)} images to generate")
    runner.run_items(todo, max_minutes=max_minutes)
    if any(not os.path.exists(runner.img_path(sp, m, os.path.basename(d), s))
           for sp, ds in jobs for d in ds for m in names for s in TUNE_SEEDS):
        raise RuntimeError("tuning images incomplete (time limit or errors); run this stage again")
    rows = []
    for sp, ds in jobs:
        df = evaluate_images(P.outputs, P.benchmarks, sp, names, TUNE_SEEDS, vis, out_res,
                             detector=E.eval_detector, det_thr=E.det_thr, iou_thr=E.match_iou, scene_dirs=ds,
                             include_real=False)
        rows.append(df[df.sid.isin({os.path.basename(d) for d in ds}) & df.seed.isin(TUNE_SEEDS)
                       & df.method.isin(names)])
    df = pd.concat(rows)
    per_scene = df.groupby(["method", "split", "sid"])[["opr", "miou", "ra", "oce_c", "clip", "id_sim"]].mean()
    tab = per_scene.groupby("method").mean()
    tab["score"] = tab["opr"] + tab["miou"] + tab["ra"].fillna(0)
    ok = tab[tab["clip"] >= tab.loc[ref, "clip"] - TUNE_CLIP_TOL] if ref in tab.index else tab
    best = ok["score"].idxmax()
    os.makedirs(out_res, exist_ok=True)
    tab = tab.round(4).sort_values("score", ascending=False)
    tab.to_csv(os.path.join(out_res, f"{name}_table.csv"))
    with open(os.path.join(out_res, f"{name}_table.md"), "w") as f:
        f.write(tab.round(3).to_markdown())
    print(f"[tune:{name}] results (scene means):\n" + tab.round(3).to_string() + f"\n[tune:{name}] best: {best}")
    return tab, best


def tune(P: Paths, E: ExperimentConfig, eng=None, vis=None, max_minutes: Optional[float] = None):
    """Tune on the tuning split (pilot scenes), with the tier's own M2/M3 settings, in phases:
    (1) alpha x lora_scale (TUNE_GRID), (2) how M5(b) energy guidance is applied (TUNE_ENERGY),
    (3) which of M5(a) / M5(b) to keep, jointly with alpha (TUNE_M5 x TUNE_M5_ALPHA) - thesis OCSD;
    (v2) OCSD-v2 alpha x anchor_shrink (TUNE_V2); (baselines, with tune_baselines) one knob per baseline
    (TUNE_BASELINES), so the baselines get the same rule as OCSD.
    Each grid keeps the highest OPR + mIoU + RA (scene means) among settings whose global CLIP score is within
    TUNE_CLIP_TOL of that grid's reference setting. Result: results/tuning/tuned.json; finished phases are skipped."""
    from .config import TUNE_CLIP_TOL, TUNE_SEEDS
    from .engine import Engine
    from .runlog import mark_skipped
    from .vision import Vision
    t = TIERS[E.tier]
    if not t.get("tune"):
        mark_skipped("T_tune", f"tier '{E.tier}' does not tune")
        return eng, vis
    tuned = json.load(open(_tuned_path(P))) if os.path.exists(_tuned_path(P)) else None
    done = _phases_done(tuned)
    if set(done) >= set(tune_phases(E)):
        mark_skipped("T_tune", f"already tuned: {tuned['chosen']}")
        return eng, vis
    if not os.path.exists(_tuning_split_path(P)):
        raise RuntimeError("no tuning split: run the pilot tier first, then stage A of this tier")
    if eng is None:
        eng = Engine.from_pretrained(E.backbone, "cuda", E.fp16, cache_dir=P.cache)
    if vis is None:
        vis = Vision("cuda", cache_dir=P.cache)
    os.makedirs(os.path.join(P.results, "tuning"), exist_ok=True)
    rule = f"max OPR+mIoU+RA with CLIP >= CLIP(reference) - {TUNE_CLIP_TOL}"

    def save(phase):
        done.append(phase)
        tuned["phases"] = list(done)
        json.dump(tuned, open(_tuned_path(P), "w"), indent=1)

    if "alpha_lora" not in done:
        variants = _tune_variants(E)
        _, best = _run_grid(P, E, eng, vis, variants, "tune_a0.5_l1.0", "tuning", max_minutes)
        tuned = dict(chosen=dict(alpha=variants[best].alpha, lora_scale=variants[best].lora_scale), method=best,
                     rule=rule, n_scenes={sp: len(ds) for sp, ds in _tuning_jobs(P)}, seeds=TUNE_SEEDS,
                     time=time.strftime("%Y-%m-%d %H:%M:%S"))
        save("alpha_lora")
    if "energy" not in done:
        variants = _tune_energy_variants(E, tuned["chosen"])
        ref = [k for k in variants if k.endswith("_e_id20")][0]
        _, best = _run_grid(P, E, eng, vis, variants, ref, "tuning_energy", max_minutes)
        v = variants[best]
        tuned["chosen"].update(use_energy=v.use_energy, energy_tokens=v.energy_tokens, eta=v.eta)
        tuned.update(method_energy=best, time_energy=time.strftime("%Y-%m-%d %H:%M:%S"))
        save("energy")
    if "m5_alpha" not in done:
        variants, ref = _tune_m5_variants(E, tuned["chosen"])
        _, best = _run_grid(P, E, eng, vis, variants, ref, "tuning_m5", max_minutes)
        v = variants[best]
        tuned["chosen"].update(alpha=v.alpha, use_region_attn=v.use_region_attn, use_energy=v.use_energy)
        tuned.update(method_m5=best, reference_m5=ref, time_m5=time.strftime("%Y-%m-%d %H:%M:%S"))
        save("m5_alpha")
    if "v2" not in done:
        variants, ref = _tune_v2_variants(E)
        _, best = _run_grid(P, E, eng, vis, variants, ref, "tuning_v2", max_minutes)
        v = variants[best]
        tuned.update(chosen_v2=dict(alpha=v.alpha, anchor_shrink=v.anchor_shrink), method_v2=best, reference_v2=ref,
                     time_v2=time.strftime("%Y-%m-%d %H:%M:%S"))
        save("v2")
    if "baselines" in tune_phases(E) and "baselines" not in done:
        from .config import TUNE_BASELINES
        bl_done = tuned.setdefault("baselines", {})
        for bl, field, values in TUNE_BASELINES:
            if bl in bl_done or (bl in ("gligen", "t2i_adapter") and E.backbone != "sd15"):
                continue
            variants, cfgs, ref = _tune_baseline_grid(E, bl, field, values, bl_done)
            _, best = _run_grid(P, E, eng, vis, variants, ref, f"tuning_bl_{bl}", max_minutes, baselines=cfgs)
            c = variants[best] if best in variants else cfgs[best][1]
            bl_done[bl] = {field: getattr(c, field)}
            json.dump(tuned, open(_tuned_path(P), "w"), indent=1)
        tuned.update(time_baselines=time.strftime("%Y-%m-%d %H:%M:%S"))
        save("baselines")
    print(f"[tune] chosen: {tuned['chosen']}; OCSD-v2: {tuned.get('chosen_v2')}; baselines: {tuned.get('baselines')}")
    return eng, vis


def _gen_signatures(E: ExperimentConfig, tuned: Optional[Dict] = None, defaults: Optional[Dict] = None,
                    P: Optional[Paths] = None, legacy: bool = False) -> Dict[str, Dict[str, Dict]]:
    """split -> method -> the settings its images depend on: the whole config of every OCSD-family variant, and the
    BASELINE_FIELDS of every baseline. use_caption is left out on QuickDraw, where the caption never adds anything to
    P_g (sketch.caption_adds_info). legacy=True describes outputs made before the baselines had their own settings."""
    from .methods import BASELINE_FIELDS
    var, _ = tier_variants(E, P=P, tuned=tuned, defaults=defaults,
                           tuned_v2={} if legacy else None, tuned_zhang={} if legacy else None)
    bls = tier_baselines(E, P=P, defaults=defaults, baseline_defaults={} if legacy else None,
                         tuned_bl={} if legacy else None)
    out = {}
    for sp in ("quickdraw", "coco"):
        sig = {m: c.to_dict() for m, c in var.items()}
        for m, (kind, c) in bls.items():
            d = c.to_dict()
            sig[m] = {f: d[f] for f in BASELINE_FIELDS[kind]}
        for d in sig.values():
            if sp == "quickdraw":
                d.pop("use_caption", None)
        out[sp] = json.loads(json.dumps(sig))   # tuples -> lists, as read back from the marker
    return out


def _complete(old: Dict, new: Dict) -> Dict:
    """`old` with every key of `new` it lacks filled with the OCSDConfig default (new fields default to the behaviour
    from before they existed, see config.LEGACY_DEFAULTS)."""
    import dataclasses
    from .config import OCSDConfig
    dflt = json.loads(json.dumps({f.name: f.default for f in dataclasses.fields(OCSDConfig)}))
    return {k: old.get(k, dflt.get(k)) for k in new}


def _archive_stale(P: Paths, E: ExperimentConfig):
    """Images of the OCSD family (and of the P_g-based baselines) depend on the tuned settings and the code defaults:
    when the settings of a (split, method) change, move its images, metrics and FID row aside (outputs/_stale,
    results/_stale) so they are generated and scored again. Everything else is kept.

    The settings each folder was generated with are kept in outputs/.gen_configs.json. Outputs from before that file
    existed are described by outputs/.ocsd_tuned.json (the tuned values then) plus config.LEGACY_DEFAULTS."""
    from .config import LEGACY_DEFAULTS
    marker = os.path.join(P.outputs, ".gen_configs.json")
    legacy = os.path.join(P.outputs, ".ocsd_tuned.json")
    new = _gen_signatures(E, P=P)
    if os.path.exists(marker):
        old = json.load(open(marker))
    elif os.path.exists(legacy):
        old = _gen_signatures(E, tuned=json.load(open(legacy)), defaults=LEGACY_DEFAULTS, legacy=True)
    else:
        old = new   # nothing generated with a tuned configuration yet
    # a method without stored settings was generated with the defaults of its fields (baselines before this marker)
    stale = [(sp, m) for sp in new for m in new[sp]
             if _complete(old.get(sp, {}).get(m, {}), new[sp][m]) != new[sp][m]
             and (os.path.isdir(os.path.join(P.outputs, sp, m)) or m in old.get(sp, {}))]
    ts = time.strftime("%Y%m%d_%H%M%S")
    moved = []
    for sp, m in stale:
        src = os.path.join(P.outputs, sp, m)
        if os.path.isdir(src):
            dst = os.path.join(P.outputs, "_stale", ts, sp, m)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
            moved.append(f"{sp}/{m}")
        # its scores: main evaluator and every extra evaluator (results/<split>/det_<name>/)
        for csv in [os.path.join(P.results, sp, f"per_image_{m}.csv")] + \
                glob.glob(os.path.join(P.results, sp, "det_*", f"per_image_{m}.csv")):
            if os.path.exists(csv):
                sub = os.path.relpath(os.path.dirname(csv), os.path.join(P.results, sp))
                dst = os.path.normpath(os.path.join(P.results, "_stale", ts, sp, sub))
                os.makedirs(dst, exist_ok=True)
                os.rename(csv, os.path.join(dst, os.path.basename(csv)))
    for sp in {sp for sp, _ in stale}:
        q = os.path.join(P.results, sp, "quality_fid_kid.csv")
        if os.path.exists(q):
            import pandas as pd
            d = pd.read_csv(q)
            d[~d.method.isin([m for s, m in stale if s == sp])].to_csv(q, index=False)
    json.dump(new, open(marker, "w"), indent=1)
    if moved:
        print(f"[tune] settings changed for {len(moved)} output folders, moved to outputs/_stale/{ts} and generated "
              f"again: {', '.join(sorted(moved))}")


# ============================================================================ 2. generate
def experiment_plan(P: Paths, E: ExperimentConfig) -> List[Dict]:
    """Danh sách công việc (E3 QuickDraw, E3 COCO, E4 cắt bỏ, alpha, OCSD-v2 ablations, power, pareto). Tập cảnh của
    E4/alpha nằm trong tập cảnh đã học định danh để dùng lại M2/M3 (tiết kiệm GPU). priority: 0 = main tables, then
    the power job (1) and the sweeps (2), which run after everything else. E.skip_jobs drops jobs by name prefix."""
    t = TIERS[E.tier]
    main = [m for m in MAIN if not (m in ("gligen", "t2i_adapter", "gligen_bon") and E.backbone != "sd15")]
    free = [m for m in main if m not in ("ocsd", "zhang2025")]
    ex = excluded_scenes(P, E)
    # QuickDraw scenes are numbered per cell; the main jobs only see the first qd_per_cell (+ tuning scenes) of each
    # cell, so scenes built later for a larger power set never change the E3 / E4 scenes
    n_tune = _max_per_cell(P, sorted(s for s in ex if s.startswith("qd_")))
    ex_main = ex | _qd_from(P, t["qd_per_cell"] + n_tune)
    qd_all = list_scenes(P.benchmarks, "quickdraw", limit=12 * t["qd_per_cell"], exclude=ex_main)
    pp = t.get("power_per_cell", t["qd_per_cell"])
    qd_power = list_scenes(P.benchmarks, "quickdraw", limit=12 * pp, exclude=ex | _qd_from(P, pp + n_tune)) \
        if pp > t["qd_per_cell"] else qd_all
    qd_tr = list_scenes(P.benchmarks, "quickdraw", limit=t["trained_n"], seed=1, exclude=ex_main)
    qd_tr = [d for d in qd_tr if d in qd_all] or qd_all[: t["trained_n"]]
    coco_all = list_scenes(P.benchmarks, "coco", limit=t["coco_n"], seed=1, exclude=ex)
    coco_tr = list_scenes(P.benchmarks, "coco", limit=t["coco_trained_n"], seed=1, exclude=ex)
    coco_tr = [d for d in coco_tr if d in coco_all] or coco_all[: t["coco_trained_n"]]
    mid = [d for d in qd_tr if os.path.basename(d).split("_")[1] in ("3", "5")]
    # ablation rows that tuning made identical to their method (e.g. "no M5(b)" when M5(b) is off) only repeat its row
    var, _ = tier_variants(E, P=P)
    abl = [m for m in ABLATION if m in OCSD_VARIANTS and m not in ("ocsd", "zhang2025", "ocsd_lite")
           and var[m] != var["ocsd"]]
    abl_v2 = [m for m in ABLATION_V2[1:] if var[m] != var["ocsd_v2"]]
    abl_v2_coco = [m for m in ABLATION_V2_COCO[1:] if var[m] != var["ocsd_v2"]]
    power = [m for m in t.get("power", []) if m in main]
    pareto = [m for m in PARETO_METHODS if not (m.startswith("pgl_") and E.backbone != "sd15")] if t.get("pareto") else []
    plan = [
        dict(name="E3_quickdraw_free", split="quickdraw", scenes=qd_all, seeds=E.seeds[: t.get("seeds_all", 99)],
             methods=free),
        # the remaining seeds on the trained subset, so the main comparison (E3) has every seed for every method
        dict(name="E3_quickdraw_free_seeds", split="quickdraw", scenes=qd_tr,
             seeds=E.seeds[t.get("seeds_all", 99):], methods=free),
        dict(name="E3_quickdraw_trained", split="quickdraw", scenes=qd_tr, seeds=E.seeds, methods=["ocsd", "zhang2025"]),
        dict(name="E3_coco_free", split="coco", scenes=coco_all, seeds=E.seeds[:1], methods=free),
        dict(name="E3_coco_trained", split="coco", scenes=coco_tr, seeds=E.seeds[:1], methods=["ocsd", "zhang2025"]),
        dict(name="E4_ablation", split="quickdraw", scenes=mid[: t["ablation_n"]], seeds=E.seeds[:1], methods=abl),
        dict(name="E4v2_ablation", split="quickdraw", scenes=mid[: t["ablation_n"]], seeds=E.seeds[:1], methods=abl_v2),
        dict(name="E4v2_coco", split="coco", scenes=coco_all, seeds=E.seeds[:1], methods=abl_v2_coco),
        dict(name="alpha", split="quickdraw", scenes=mid[: t["alpha_n"]], seeds=E.seeds[:1], methods=ALPHAS),
        # every seed on every QuickDraw scene for the pre-registered tests (docs/PREREGISTRATION.md)
        dict(name="E3_quickdraw_power", split="quickdraw", scenes=qd_power, seeds=E.seeds, methods=power, priority=1),
        # control-vs-quality sweeps: every QuickDraw scene, first seed (KID needs many images per point)
        dict(name="pareto", split="quickdraw", scenes=qd_all, seeds=E.seeds[:1], methods=pareto, priority=2),
    ]
    skip = [x.strip() for x in (E.skip_jobs or "").split(",") if x.strip()]
    return [dict(p, priority=p.get("priority", 0)) for p in plan
            if p["scenes"] and p["seeds"] and p["methods"] and not any(p["name"].startswith(k) for k in skip)]


def _qd_from(P: Paths, k: int) -> set:
    """QuickDraw scene ids whose number within their cell is k or more (qd_<count>_<complexity>_<number>)."""
    return {os.path.basename(d) for d in list_scenes(P.benchmarks, "quickdraw")
            if int(os.path.basename(d).rsplit("_", 1)[1]) >= k}


def tier_variants(E: ExperimentConfig, apply_tuned: bool = True, P: Optional[Paths] = None,
                  tuned: Optional[Dict] = None, defaults: Optional[Dict] = None, tuned_v2: Optional[Dict] = None,
                  tuned_zhang: Optional[Dict] = None):
    """Áp siêu tham số của tier cho mọi phương pháp.
    The tier settings apply to every field a variant does not set itself (methods.VARIANT_OVERRIDES). The tuned
    values (results/tuning/tuned.json, or the arguments) apply to the fields outside methods.VARIANT_OWN: "chosen" to
    the thesis family, "chosen_v2" to the OCSD-v2 family, and only the baseline phase's value to Zhang et al.
    `defaults` replaces OCSDConfig defaults (e.g. LEGACY_DEFAULTS)."""
    from .config import OCSDConfig
    from .methods import VARIANT_FAMILY, VARIANT_OVERRIDES, VARIANT_OWN
    ov = TIERS[E.tier].get("cfg", {})
    base = OCSDConfig(**(defaults or {}))
    t = _load_tuned(P, E) if apply_tuned else {}
    tuned = t.get("chosen", {}) if tuned is None else tuned
    tuned_v2 = t.get("chosen_v2", {}) if tuned_v2 is None else tuned_v2
    tuned_zhang = t.get("baselines", {}).get("zhang2025", {}) if tuned_zhang is None else tuned_zhang
    var = {}
    for k, over in VARIANT_OVERRIDES.items():
        own = VARIANT_OWN.get(k, over)
        c = base.replace(**{f: v for f, v in ov.items() if f not in over}).replace(**over)
        if k == "zhang2025":
            c = c.replace(**tuned_zhang)
        else:
            fam = tuned_v2 if VARIANT_FAMILY.get(k) == "v2" else tuned
            c = c.replace(**{f: v for f, v in fam.items() if f not in own})
        var[k] = c
    return var, base.replace(**ov)


def tier_baselines(E: ExperimentConfig, P: Optional[Paths] = None, tuned_bl: Optional[Dict] = None,
                   defaults: Optional[Dict] = None, baseline_defaults: Optional[Dict] = None) -> Dict[str, tuple]:
    """method -> (baseline kind, config) for every baseline and every baseline sweep point: the tier settings, the
    BASELINE_DEFAULTS fixes, and the values of the baseline tuning phase (tuned.json "baselines", or `tuned_bl`).
    cn_region / cn_energy / collage use ControlNet's tuned scale; bo3 / boN variants use their base method's values;
    the collage baselines draw their M2 objects like OCSD-v2 (per seed); sweep points are not tuned."""
    from .config import BASELINE_DEFAULTS
    from .methods import BASELINE_TUNED_FROM, COLLAGE_BASELINES, PARETO_BASELINES
    _, base = tier_variants(E, apply_tuned=False, defaults=defaults)
    base = base.replace(**(BASELINE_DEFAULTS if baseline_defaults is None else baseline_defaults))
    if tuned_bl is None:
        tuned_bl = _load_tuned(P, E).get("baselines", {})
    v2 = tier_variants(E, apply_tuned=False)[0]["ocsd_v2"]
    out = {}
    for b in BASELINES:
        src = BASELINE_TUNED_FROM.get(b, b)
        c = base
        if src in ("cn_region", "cn_energy", "collage"):
            c = c.replace(**tuned_bl.get("controlnet", {}))
        c = c.replace(**tuned_bl.get(src, {}))
        if b in COLLAGE_BASELINES:
            c = c.replace(m2_per_seed=v2.m2_per_seed)
        out[b] = (b, c)
    for name, (b, field, value) in PARETO_BASELINES.items():
        out[name] = (b, base.replace(**{field: value}))
    return out


def _count_todo(P, job):
    n = 0
    for sdir in job["scenes"]:
        sid = os.path.basename(sdir)
        for m in job["methods"]:
            for s in job["seeds"]:
                if not os.path.exists(os.path.join(P.outputs, job["split"], m, f"{sid}_s{s}.png")):
                    n += 1
    return n


def work_items(P: Paths, E: ExperimentConfig):
    """Gộp mọi công việc theo cảnh: mỗi cảnh được xử lý một lần với mọi phương pháp/seed của nó, nên M2/M3 chỉ làm
    một lần cho mỗi cảnh. Thứ tự: theo priority của job (bảng chính trước, rồi power, rồi pareto), trong mỗi mức cảnh
    có học định danh trước (bảng E3/E4 đầy đủ sớm nhất), rồi phần còn lại. Each image goes to its lowest priority."""
    prio: Dict[tuple, int] = {}
    plan = experiment_plan(P, E)
    trained = {d for j in plan if "trained" in j["name"] for d in j["scenes"]}
    for j in plan:
        for d in j["scenes"]:
            for m in j["methods"]:
                for s in j["seeds"]:
                    k = (j["split"], d, m, s)
                    prio[k] = min(prio.get(k, 99), j["priority"])
    items: Dict[tuple, List] = {}
    for (split, d, m, s), pr in prio.items():
        items.setdefault((pr, split, d), []).append((m, s))
    order = sorted(items, key=lambda k: (k[0], k[1] != "quickdraw", k[2] not in trained, k[2]))
    return [(split, d, items[(pr, split, d)]) for pr, split, d in order]


def generate(P: Paths, E: ExperimentConfig, max_minutes: Optional[float] = None, eng=None, vis=None):
    from .engine import Engine
    from .runner import Runner
    from .vision import Vision
    if eng is None:
        eng = Engine.from_pretrained(E.backbone, "cuda", E.fp16, cache_dir=P.cache)
    if vis is None:
        vis = Vision("cuda", cache_dir=P.cache)
    if TIERS[E.tier].get("tune"):
        if not tuning_finished(P, E):
            raise RuntimeError("stage T (tuning) has not finished: run it before generating images for this tier")
        _archive_stale(P, E)
    variants, base = tier_variants(E, P=P)
    runner = Runner(eng, vis, P.outputs, E.backbone, variants=variants, base_cfg=base, cache_dir=P.cache,
                    baselines=tier_baselines(E, P=P))
    items = work_items(P, E)
    todo = [(sp, d, [(m, s) for m, s in ms if not os.path.exists(runner.img_path(sp, m, os.path.basename(d), s))])
            for sp, d, ms in items]
    todo = [x for x in todo if x[2]]
    print(f"Còn {sum(len(x[2]) for x in todo)} ảnh trên {len(todo)} cảnh")
    runner.run_items(todo, max_minutes=max_minutes)
    log_progress(P, E)
    estimate(P, E)
    return eng, vis


def estimate(P: Paths, E: ExperimentConfig, tiers=("pilot", "paper", "full")) -> Dict:
    """Ước tính giờ GPU từ thời gian ĐO THỰC của các ảnh đã sinh (nhật ký .json), cho từng tier."""
    import numpy as np
    per_img: Dict[str, List[float]] = {}
    m2, m3 = [], []
    for f in glob.glob(os.path.join(P.outputs, "*", "*", "*.json")):
        if "/_objects/" in f:
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        per_img.setdefault(r["method"], []).append(r.get("scene_time") or 0)
        pt = r.get("prep_times") or {}
        if pt.get("m3"):
            m3.append(pt["m3"])
    for f in glob.glob(os.path.join(P.outputs, "*", "_objects", "*", "*", "objects.json")):
        m2.append(json.load(open(f)).get("m2_time") or 0)
    if not per_img:
        print("Chưa có ảnh nào để ước tính.")
        return {}
    t_img = {k: float(np.median(v)) for k, v in per_img.items()}
    t_m2 = float(np.median(m2)) if m2 else 60.0
    t_m3 = float(np.median(m3)) if m3 else 120.0
    default = float(np.median(list(t_img.values())))
    out = dict(sec_per_image=t_img, m2_sec_per_scene=t_m2, m3_sec_per_scene=t_m3, tiers={})
    main = [m for m in MAIN if not (m in ("gligen", "t2i_adapter") and E.backbone != "sd15")]
    free = [m for m in main if m not in ("ocsd", "zhang2025")]
    abl = [m for m in ABLATION if m in OCSD_VARIANTS and m not in ("ocsd", "zhang2025", "ocsd_lite")]
    T = lambda m: t_img.get(m, default)
    for tier in tiers:
        t = TIERS[tier]
        S = len(t["seeds"])
        n_qd = 12 * t["qd_per_cell"]
        Sa = min(S, t.get("seeds_all", S))
        sec = (n_qd * Sa + t["trained_n"] * (S - Sa)) * sum(map(T, free)) + t["trained_n"] * S * (T("ocsd") + T("zhang2025"))
        sec += t["coco_n"] * sum(map(T, free)) + t["coco_trained_n"] * (T("ocsd") + T("zhang2025"))
        sec += t["ablation_n"] * sum(map(T, abl)) + t["alpha_n"] * sum(map(T, ALPHAS))
        # M2: 1 lần/cảnh (khóa cụm từ) + cảnh có học định danh thêm khóa tên lớp (Zhang) + ablation K=1
        sec += t_m2 * (n_qd + t["coco_n"] + 0.5 * (t["trained_n"] + t["coco_trained_n"]) + 0.5 * t["ablation_n"])
        # M3: OCSD + Zhang trên cảnh học định danh, + 3 khóa cắt bỏ (không L_att, K=1, 200+200 bước ~ 2x)
        sec += t_m3 * (2 * (t["trained_n"] + t["coco_trained_n"]) + 4 * t["ablation_n"])
        # power job: the remaining seeds of its methods on the scenes outside the trained subset (+ M3 for OCSD)
        rest = n_qd - t["trained_n"]
        for m in t.get("power", []):
            if m == "ocsd":
                sec += rest * (S * T(m) + t_m3)
            elif m in free:
                sec += rest * (S - Sa) * T(m)
        # OCSD-v2 / collage draw M2 objects per seed; OCSD-v2 ablations; sweeps
        sec += t_m2 * (S - 1) * (n_qd + t["coco_n"])
        sec += t["ablation_n"] * sum(T(m) for m in ABLATION_V2[1:]) + t["coco_n"] * sum(T(m) for m in ABLATION_V2_COCO[1:])
        if t.get("pareto"):
            sec += n_qd * sum(T(m) for m in PARETO_METHODS)
        out["tiers"][tier] = dict(gpu_hours=round(sec / 3600, 1))
    json.dump(out, open(os.path.join(P.results, "budget_estimate.json"), "w"), indent=1)
    print("Ước tính giờ GPU (theo thời gian đo trên GPU hiện tại, chưa gồm đánh giá ~10-15%):",
          {k: v["gpu_hours"] for k, v in out["tiers"].items()})
    return out


def log_progress(P: Paths, E: ExperimentConfig):
    prog = {j["name"]: dict(total=len(j["scenes"]) * len(j["methods"]) * len(j["seeds"]), todo=_count_todo(P, j))
            for j in experiment_plan(P, E)}
    json.dump(prog, open(os.path.join(P.results, "progress.json"), "w"), indent=1)
    for k, v in prog.items():
        print(f"{k:24s} {v['total'] - v['todo']:6d}/{v['total']} ảnh")
    return prog


# ============================================================================ 3. evaluate
def evaluate(P: Paths, E: ExperimentConfig, vis=None, fid: bool = True):
    import torch
    from .metrics import evaluate_detectors, evaluate_images, image_level_quality
    from .vision import Vision
    vis = vis or Vision("cuda", cache_dir=P.cache)
    plan = experiment_plan(P, E)
    for split in ("quickdraw", "coco"):
        jobs = [j for j in plan if j["split"] == split]
        if not jobs:
            continue
        methods = sorted({m for j in jobs for m in j["methods"]})
        seeds = sorted({s for j in jobs for s in j["seeds"]})
        scenes = sorted({d for j in jobs for d in j["scenes"]})
        evaluate_images(P.outputs, P.benchmarks, split, methods, seeds, vis, P.results, detector=E.eval_detector,
                        det_thr=E.det_thr, iou_thr=E.match_iou, scene_dirs=scenes)
        e3 = [j for j in jobs if j["name"].startswith("E3_")]
        if E.extra_detectors and e3:   # robustness table: the main comparison's images (E3 jobs), other evaluators
            evaluate_detectors(P.outputs, P.benchmarks, split, sorted({m for j in e3 for m in j["methods"]}),
                               sorted({s for j in e3 for s in j["seeds"]}), vis, P.results,
                               detectors=E.extra_detectors, det_thr=E.det_thr, detr_thr=E.detr_thr,
                               iou_thr=E.match_iou, scene_dirs=sorted({d for j in e3 for d in j["scenes"]}))
        # (scene, seed) pairs of this plan per method: the report ignores per-image rows outside it
        # (e.g. pilot-tier rows of the tuning scenes)
        plan_pairs = {}
        for j in jobs:
            for m in j["methods"] + (["real"] if split == "coco" and j["name"].endswith("_free") else []):
                for d in j["scenes"]:
                    for s in (j["seeds"] if m != "real" else [0]):
                        plan_pairs.setdefault(m, set()).add((os.path.basename(d), int(s)))
        os.makedirs(os.path.join(P.results, split), exist_ok=True)
        json.dump({m: sorted(v) for m, v in plan_pairs.items()},
                  open(os.path.join(P.results, split, "plan.json"), "w"), indent=0)
        if fid:
            if split == "coco":
                ref = sorted(glob.glob(os.path.join(P.benchmarks, "coco", "*", "real.png")))
                ref += sorted(glob.glob(os.path.join(P.data, "coco_ref", "*.jpg")))
            else:
                ref = sorted(glob.glob(os.path.join(P.data, "coco_ref", "*.jpg")))
            if ref:
                vis.unload()
                # main comparison (E3): every method scored on the same images (trained scenes x their seeds),
                # since FID depends on the number of images; ablation / alpha variants on their own job
                tr = [j for j in jobs if j["name"] == f"E3_{split}_trained"]
                e3 = {(os.path.basename(d), int(s)) for j in tr for d in j["scenes"] for s in j["seeds"]}
                main = {m for j in jobs if j["name"].startswith("E3_") for m in j["methods"]}
                fid_pairs = {m: (e3 if m in main and e3 else plan_pairs.get(m, set())) for m in methods}
                image_level_quality(P.outputs, P.benchmarks, split, methods, seeds, ref, P.results,
                                    scene_ids={os.path.basename(d) for d in scenes}, pairs=fid_pairs)
    return vis


# ============================================================================ 4. report
def report(P: Paths, E: ExperimentConfig):
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    from . import report as R
    tabs = R.build_all(P.results)
    figs = []
    for split in ("quickdraw", "coco"):
        if not glob.glob(os.path.join(P.results, split, "per_image_*.csv")):
            continue
        if split == "quickdraw":
            figs.append(R.plot_curves(P.results, split))
            figs.append(R.plot_alpha(P.results, split))
            figs.append(R.plot_pareto(P.results, split))
        try:
            sids = R.pick_showcase(P.results, split, k=6)
            figs.append(R.qualitative_grid(P.outputs, P.benchmarks, split, sids,
                                           ["controlnet", "t2i_adapter", "gligen", "collage", "zhang2025", "ocsd",
                                            "ocsd_v2"],
                                           out=os.path.join(P.results, "figures", f"qualitative_{split}.png")))
            if split == "quickdraw":
                R.user_study_pack(P.outputs, P.benchmarks, split, R.pick_showcase(P.results, split, k=10),
                                  ["controlnet", "t2i_adapter", "gligen", "zhang2025", "ocsd", "ocsd_v2"],
                                  os.path.join(P.results, "user_study"))
                # realism A/B on randomly drawn scenes (not the showcase, which favours OCSD)
                R.realism_pairs_pack(P.outputs, split, _random_scenes(P, E, split, 12),
                                     [("ocsd_v2", "controlnet"), ("ocsd_v2", "gligen"), ("ocsd_v2", "ocsd"),
                                      ("ocsd_v2", "collage")], os.path.join(P.results, "user_study_realism"))
        except Exception as e:  # thiếu ảnh của một phương pháp -> bỏ qua hình
            print("Bỏ qua hình định tính:", e)
    summary = dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"), tier=E.tier, backbone=E.backbone,
                   seeds=E.seeds, eval_detector=E.eval_detector, hardware=hardware_info(),
                   progress=log_progress(P, E), tables=sorted(tabs), figures=[f for f in figs if f])
    json.dump(summary, open(os.path.join(P.results, "summary.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(P.results, "summary.md"), "w") as f:
        f.write(f"# Kết quả thực nghiệm OCSD ({summary['generated_at']})\n\n")
        f.write(f"Tier: {E.tier}; backbone: {E.backbone}; seeds: {E.seeds}; bộ phát hiện đánh giá: {E.eval_detector}\n\n")
        f.write(f"Phần cứng: {json.dumps(summary['hardware'], ensure_ascii=False)}\n\n")
        for name in sorted(tabs):
            p = os.path.join(P.results, "tables", name + ".md")
            if os.path.exists(p):
                f.write(f"## {name}\n\n" + open(p).read() + "\n\n")
    print(open(os.path.join(P.results, "summary.md")).read()[:6000])
    return summary


def _random_scenes(P: Paths, E: ExperimentConfig, split: str, k: int, seed: int = 0) -> List[str]:
    """k scenes drawn at random (fixed seed) from the scenes every main method generated."""
    import random
    jobs = [j for j in experiment_plan(P, E) if j["split"] == split and j["name"].startswith("E3_")]
    if not jobs:
        return []
    common = set.intersection(*[{os.path.basename(d) for d in j["scenes"]} for j in jobs])
    sids = sorted(common)
    random.Random(seed).shuffle(sids)
    return sorted(sids[:k])


def hardware_info():
    info = dict(python=platform.python_version())
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(p.total_memory / 2 ** 30, 1)
        import diffusers, transformers
        info["diffusers"], info["transformers"] = diffusers.__version__, transformers.__version__
    except Exception:
        pass
    try:
        info["cpu"] = subprocess.run("lscpu | grep 'Model name' | cut -d: -f2", shell=True, capture_output=True,
                                     text=True).stdout.strip()
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except Exception:
        pass
    return info
