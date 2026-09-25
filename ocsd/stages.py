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
from .report import ABLATION, ALPHAS, MAIN


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
        mark_skipped("A_data", f"benchmarks for tier '{tier}' already built ({marker})")
        return json.load(open(marker))
    extra_qd, extra_coco = 0, 0
    if t.get("tune"):
        # the pilot scenes become the tuning split; build extra scenes so the evaluation keeps its full size
        split = freeze_tuning_split(P)
        extra_qd = _max_per_cell(P, split.get("quickdraw", []))
        extra_coco = 4 * _max_per_cell(P, split.get("coco", []))
    download_quickdraw(P.quickdraw_raw, per_class=3000)
    build_quickdraw_scenes(P.quickdraw_raw, os.path.join(P.benchmarks, "quickdraw"),
                           per_cell=t["qd_per_cell"] + extra_qd)
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
    json.dump(dict(tier=tier, counts=counts, time=time.strftime("%Y-%m-%d %H:%M:%S")), open(marker, "w"), indent=1)
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


def _tune_variants(E: ExperimentConfig):
    from .config import TUNE_GRID
    var, _ = tier_variants(E, apply_tuned=False)
    return {f"tune_a{a:.1f}_l{l:.1f}": var["ocsd"].replace(alpha=a, lora_scale=l)
            for a in TUNE_GRID["alpha"] for l in TUNE_GRID["lora_scale"]}


def _tune_energy_variants(E: ExperimentConfig, chosen: Dict):
    from .config import TUNE_ENERGY
    var, _ = tier_variants(E, apply_tuned=False)
    base = var["ocsd"].replace(alpha=chosen["alpha"], lora_scale=chosen["lora_scale"])
    return {f"tune_a{chosen['alpha']:.1f}_l{chosen['lora_scale']:.1f}_{k}": base.replace(**v)
            for k, v in TUNE_ENERGY.items()}


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


def _run_grid(P, E, eng, vis, variants, ref, name, max_minutes):
    """Generate + evaluate every variant on the tuning split; return (table, best variant name)."""
    import pandas as pd
    from .config import TUNE_CLIP_TOL, TUNE_SEEDS
    from .metrics import evaluate_images
    from .runner import Runner
    jobs = _tuning_jobs(P)
    names = sorted(variants)
    out_res = os.path.join(P.results, "tuning")
    _, base = tier_variants(E, apply_tuned=False)
    runner = Runner(eng, vis, P.outputs, E.backbone, variants=variants, base_cfg=base, cache_dir=P.cache)
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
    """Tune OCSD on the tuning split (pilot scenes), with the tier's own M2/M3 settings, in two phases:
    (1) alpha x lora_scale (TUNE_GRID), (2) how M5(b) energy guidance is applied (TUNE_ENERGY).
    Each phase keeps the highest OPR + mIoU + RA (scene means) among settings whose global CLIP score is within
    TUNE_CLIP_TOL of that phase's reference setting. Result: results/tuning/tuned.json; finished phases are skipped."""
    from .config import TUNE_CLIP_TOL, TUNE_SEEDS
    from .engine import Engine
    from .runlog import mark_skipped
    from .vision import Vision
    t = TIERS[E.tier]
    if not t.get("tune"):
        mark_skipped("T_tune", f"tier '{E.tier}' does not tune")
        return eng, vis
    tuned = json.load(open(_tuned_path(P))) if os.path.exists(_tuned_path(P)) else None
    if tuned and "use_energy" in tuned["chosen"]:
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
    if tuned is None:
        variants = _tune_variants(E)
        _, best = _run_grid(P, E, eng, vis, variants, "tune_a0.5_l1.0", "tuning", max_minutes)
        tuned = dict(chosen=dict(alpha=variants[best].alpha, lora_scale=variants[best].lora_scale), method=best,
                     rule=rule, n_scenes={sp: len(ds) for sp, ds in _tuning_jobs(P)}, seeds=TUNE_SEEDS,
                     time=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(tuned, open(_tuned_path(P), "w"), indent=1)
    variants = _tune_energy_variants(E, tuned["chosen"])
    ref = [k for k in variants if k.endswith("_e_id20")][0]
    _, best = _run_grid(P, E, eng, vis, variants, ref, "tuning_energy", max_minutes)
    v = variants[best]
    tuned["chosen"].update(use_energy=v.use_energy, energy_tokens=v.energy_tokens, eta=v.eta)
    tuned.update(method_energy=best, time_energy=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(tuned, open(_tuned_path(P), "w"), indent=1)
    print(f"[tune] chosen: {tuned['chosen']}")
    return eng, vis


def _archive_stale_ocsd(P: Paths):
    """OCSD-family images depend on the tuned settings: when those change, move the old images, their metrics and
    FID rows aside (outputs/_stale, results/_stale) so they are generated and scored again. Baselines are kept."""
    chosen = json.load(open(_tuned_path(P)))["chosen"]
    marker = os.path.join(P.outputs, ".ocsd_tuned.json")
    prev = json.load(open(marker)) if os.path.exists(marker) else \
        {k: v for k, v in chosen.items() if k in ("alpha", "lora_scale")}   # runs before energy tuning existed
    if prev == chosen:
        return
    fam = [m for m in OCSD_VARIANTS if m != "zhang2025"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    moved = 0
    for sp in ("quickdraw", "coco"):
        for m in fam:
            src = os.path.join(P.outputs, sp, m)
            if os.path.isdir(src):
                dst = os.path.join(P.outputs, "_stale", ts, sp, m)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.rename(src, dst)
                moved += 1
            csv = os.path.join(P.results, sp, f"per_image_{m}.csv")
            if os.path.exists(csv):
                dst = os.path.join(P.results, "_stale", ts, sp)
                os.makedirs(dst, exist_ok=True)
                os.rename(csv, os.path.join(dst, os.path.basename(csv)))
        q = os.path.join(P.results, sp, "quality_fid_kid.csv")
        if os.path.exists(q):
            import pandas as pd
            d = pd.read_csv(q)
            d[~d.method.isin(fam)].to_csv(q, index=False)
    json.dump(chosen, open(marker, "w"), indent=1)
    if moved:
        print(f"[tune] settings changed {prev} -> {chosen}: moved {moved} OCSD-family output folders to "
              f"outputs/_stale/{ts}; they will be generated again")


# ============================================================================ 2. generate
def experiment_plan(P: Paths, E: ExperimentConfig) -> List[Dict]:
    """Danh sách công việc (E3 QuickDraw, E3 COCO, E4 cắt bỏ, alpha). Tập cảnh của E4/alpha nằm trong tập cảnh đã
    học định danh để dùng lại M2/M3 (tiết kiệm GPU)."""
    t = TIERS[E.tier]
    main = [m for m in MAIN if not (m in ("gligen", "t2i_adapter") and E.backbone != "sd15")]
    free = [m for m in main if m not in ("ocsd", "zhang2025")]
    ex = excluded_scenes(P, E)
    qd_all = list_scenes(P.benchmarks, "quickdraw", limit=12 * t["qd_per_cell"], exclude=ex)
    qd_tr = list_scenes(P.benchmarks, "quickdraw", limit=t["trained_n"], seed=1, exclude=ex)
    qd_tr = [d for d in qd_tr if d in qd_all] or qd_all[: t["trained_n"]]
    coco_all = list_scenes(P.benchmarks, "coco", limit=t["coco_n"], seed=1, exclude=ex)
    coco_tr = list_scenes(P.benchmarks, "coco", limit=t["coco_trained_n"], seed=1, exclude=ex)
    coco_tr = [d for d in coco_tr if d in coco_all] or coco_all[: t["coco_trained_n"]]
    mid = [d for d in qd_tr if os.path.basename(d).split("_")[1] in ("3", "5")]
    plan = [
        dict(name="E3_quickdraw_free", split="quickdraw", scenes=qd_all, seeds=E.seeds[: t.get("seeds_all", 99)],
             methods=free),
        # the remaining seeds on the trained subset, so the main comparison (E3) has every seed for every method
        dict(name="E3_quickdraw_free_seeds", split="quickdraw", scenes=qd_tr,
             seeds=E.seeds[t.get("seeds_all", 99):], methods=free),
        dict(name="E3_quickdraw_trained", split="quickdraw", scenes=qd_tr, seeds=E.seeds, methods=["ocsd", "zhang2025"]),
        dict(name="E3_coco_free", split="coco", scenes=coco_all, seeds=E.seeds[:1], methods=free),
        dict(name="E3_coco_trained", split="coco", scenes=coco_tr, seeds=E.seeds[:1], methods=["ocsd", "zhang2025"]),
        dict(name="E4_ablation", split="quickdraw", scenes=mid[: t["ablation_n"]], seeds=E.seeds[:1],
             methods=[m for m in ABLATION if m in OCSD_VARIANTS and m not in ("ocsd", "zhang2025", "ocsd_lite")]),
        dict(name="alpha", split="quickdraw", scenes=mid[: t["alpha_n"]], seeds=E.seeds[:1], methods=ALPHAS),
    ]
    return [p for p in plan if p["scenes"] and p["seeds"]]


def tier_variants(E: ExperimentConfig, apply_tuned: bool = True, P: Optional[Paths] = None):
    """Áp siêu tham số của tier cho mọi phương pháp (Zhang et al. giữ K = 1).
    With a tuned.json (stages.tune), its alpha / lora_scale replace the defaults in every OCSD variant that does
    not set them itself (the alpha sweep and alpha ablations keep their alpha; Zhang et al. keeps its own setup)."""
    from .config import OCSDConfig
    ov = TIERS[E.tier].get("cfg", {})
    base = OCSDConfig()
    tuned = {}
    if apply_tuned and P is not None and TIERS[E.tier].get("tune") and os.path.exists(_tuned_path(P)):
        tuned = json.load(open(_tuned_path(P)))["chosen"]
    var = {}
    for k, v in OCSD_VARIANTS.items():
        o = dict(ov)
        if v.K == 1:
            o.pop("K", None)
        if k != "zhang2025":
            for f, val in tuned.items():
                if getattr(v, f) == getattr(base, f):
                    o[f] = val
        var[k] = v.replace(**o)
    return var, base.replace(**ov)


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
    một lần cho mỗi cảnh. Thứ tự: cảnh có học định danh trước (bảng E3/E4 đầy đủ sớm nhất), rồi phần còn lại."""
    items: Dict[tuple, List] = {}
    order: List[tuple] = []
    plan = experiment_plan(P, E)
    trained = {d for j in plan if "trained" in j["name"] for d in j["scenes"]}
    for j in plan:
        for d in j["scenes"]:
            key = (j["split"], d)
            if key not in items:
                items[key] = []
                order.append(key)
            for m in j["methods"]:
                for s in j["seeds"]:
                    if (m, s) not in items[key]:
                        items[key].append((m, s))
    order.sort(key=lambda k: (k[0] != "quickdraw", k[1] not in trained))
    return [(split, d, items[(split, d)]) for split, d in order]


def generate(P: Paths, E: ExperimentConfig, max_minutes: Optional[float] = None, eng=None, vis=None):
    from .engine import Engine
    from .runner import Runner
    from .vision import Vision
    if eng is None:
        eng = Engine.from_pretrained(E.backbone, "cuda", E.fp16, cache_dir=P.cache)
    if vis is None:
        vis = Vision("cuda", cache_dir=P.cache)
    if TIERS[E.tier].get("tune"):
        if not os.path.exists(_tuned_path(P)) or "use_energy" not in json.load(open(_tuned_path(P)))["chosen"]:
            raise RuntimeError("stage T (tuning) has not finished: run it before generating images for this tier")
        _archive_stale_ocsd(P)
    variants, base = tier_variants(E, P=P)
    runner = Runner(eng, vis, P.outputs, E.backbone, variants=variants, base_cfg=base, cache_dir=P.cache)
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
        # M3: OCSD + Zhang trên cảnh học định danh, + 2 khóa cắt bỏ (không L_att, K=1)
        sec += t_m3 * (2 * (t["trained_n"] + t["coco_trained_n"]) + 2 * t["ablation_n"])
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
    from .metrics import evaluate_images, image_level_quality
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
        if fid:
            if split == "coco":
                ref = sorted(glob.glob(os.path.join(P.benchmarks, "coco", "*", "real.png")))
                ref += sorted(glob.glob(os.path.join(P.data, "coco_ref", "*.jpg")))
            else:
                ref = sorted(glob.glob(os.path.join(P.data, "coco_ref", "*.jpg")))
            if ref:
                vis.unload()
                image_level_quality(P.outputs, P.benchmarks, split, methods, seeds, ref, P.results,
                                    scene_ids={os.path.basename(d) for d in scenes})
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
        try:
            sids = R.pick_showcase(P.results, split, k=6)
            figs.append(R.qualitative_grid(P.outputs, P.benchmarks, split, sids,
                                           ["controlnet", "t2i_adapter", "gligen", "zhang2025", "ocsd"],
                                           out=os.path.join(P.results, "figures", f"qualitative_{split}.png")))
            if split == "quickdraw":
                R.user_study_pack(P.outputs, P.benchmarks, split, R.pick_showcase(P.results, split, k=10),
                                  ["controlnet", "t2i_adapter", "gligen", "zhang2025", "ocsd"],
                                  os.path.join(P.results, "user_study"))
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
