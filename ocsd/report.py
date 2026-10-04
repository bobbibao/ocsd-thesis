"""Tổng hợp kết quả: trung bình + khoảng tin cậy 95% (bootstrap theo cảnh), kiểm định Wilcoxon có ghép cặp
(hiệu chỉnh Holm), bảng E1-E4/alpha/thời gian (CSV + Markdown + LaTeX) và hình vẽ cho khóa luận."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import PARETO
from .methods import LABELS, V2_ABLATIONS, V2_COCO_ABLATIONS

METRICS = ["opr", "oce", "oce_c", "count_acc", "miou", "ra", "clip", "obj_clip", "id_sim"]
HIGHER = dict(opr=True, oce=False, oce_c=False, count_acc=True, miou=True, ra=True, clip=True, obj_clip=True,
              id_sim=True, fid=False, kid=False, lpips_div=True, ra_cond=True, miou_matched=True)
PCT = {"opr", "count_acc", "ra", "ra_cond"}


def load_per_image(results_dir: str, split: str, sub: str = "") -> pd.DataFrame:
    """All per-image rows of `split`, limited to the (scene, seed) pairs of the current experiment plan when
    results/<split>/plan.json exists (rows left over from another tier, e.g. pilot scenes, are ignored).
    sub: a sub-folder with another evaluator's rows, e.g. "det_detr"."""
    import glob
    import json
    fs = glob.glob(os.path.join(results_dir, split, sub, "per_image_*.csv"))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    pp = os.path.join(results_dir, split, "plan.json")
    if os.path.exists(pp):
        plan = json.load(open(pp))
        ok = {(m, sid, int(seed)) for m, pairs in plan.items() for sid, seed in pairs}
        df = df[[(m, sid, int(seed)) in ok for m, sid, seed in zip(df.method, df.sid, df.seed)]]
    return df.reset_index(drop=True)


def per_scene(df: pd.DataFrame) -> pd.DataFrame:
    """Trung bình trên các seed của cùng một cảnh -> đơn vị thống kê là cảnh (tránh đếm trùng)."""
    keys = ["method", "sid", "split", "n_obj", "count_bin", "complexity"]
    num = [c for c in df.columns if c not in keys + ["seed", "missing"] and pd.api.types.is_numeric_dtype(df[c])]
    return df.groupby(keys, as_index=False)[num].mean()


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def summarize(ps: pd.DataFrame, by: Sequence[str] = ("method",), metrics=METRICS) -> pd.DataFrame:
    rows = []
    for key, g in ps.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key), n_scenes=g["sid"].nunique())
        for m in metrics:
            if m not in g:
                continue
            mu, lo, hi = bootstrap_ci(g[m].to_numpy(dtype=float))
            r[m], r[m + "_lo"], r[m + "_hi"] = mu, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def holm(df: pd.DataFrame, by: Sequence[str] = ("metric",)) -> pd.DataFrame:
    """Holm-adjusted p ("p_holm") within each group of rows (a family of comparisons for one metric)."""
    df = df.copy()
    df["p_holm"] = np.nan
    for _, g in df.groupby(list(by)):
        order = g.sort_values("p").index
        k = len(order)
        prev = 0.0
        for rank, idx in enumerate(order):
            v = min(1.0, max(prev, (k - rank) * df.loc[idx, "p"]))
            df.loc[idx, "p_holm"] = v
            prev = v
    return df


def paired_tests(ps: pd.DataFrame, ref: str = "ocsd", metrics=("opr", "oce_c", "miou", "ra", "obj_clip"),
                 others: Optional[Sequence[str]] = None, alternative: str = "two-sided") -> pd.DataFrame:
    """Wilcoxon signed-rank giữa `ref` và từng phương pháp trên cùng tập cảnh; p hiệu chỉnh Holm theo độ đo.
    alternative="greater" tests that `ref` is better (lower for metrics where lower is better)."""
    from scipy.stats import wilcoxon
    rows = []
    a = ps[ps.method == ref].set_index("sid")
    for m in (others if others is not None else sorted(set(ps.method) - {ref})):
        b = ps[ps.method == m].set_index("sid")
        common = a.index.intersection(b.index)
        for met in metrics:
            if met not in a or len(common) < 6:
                continue
            x, y = a.loc[common, met].to_numpy(float), b.loc[common, met].to_numpy(float)
            ok = ~(np.isnan(x) | np.isnan(y))
            x, y = x[ok], y[ok]
            if len(x) < 6 or np.allclose(x, y):
                p = 1.0
            else:
                sx, sy = (x, y) if HIGHER.get(met, True) else (y, x)   # orient so "greater" means ref is better
                p = float(wilcoxon(sx, sy, zero_method="zsplit", alternative=alternative).pvalue)
            rows.append(dict(ref=ref, method=m, metric=met, n=len(x), mean_ref=x.mean(), mean_other=y.mean(),
                             diff=x.mean() - y.mean(), p=p))
    df = pd.DataFrame(rows)
    return holm(df) if len(df) else df


def common_seed_rows(df: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    """Rows of `methods` limited to the (scene, seed) pairs that every one of them has, so per-scene means of
    different methods average the same seeds."""
    d = df[df.method.isin(methods)]
    pairs = [set(zip(d[d.method == m].sid, d[d.method == m].seed)) for m in methods if (d.method == m).any()]
    if not pairs:
        return d
    keep = set.intersection(*pairs)
    return d[[(s, int(e)) in keep for s, e in zip(d.sid, d.seed.astype(int))]]


# Pre-registered hypotheses (docs/PREREGISTRATION.md), tested on the E3_quickdraw_power job: every QuickDraw scene of
# the power set, every seed, scene means. Each family is Holm-corrected over its (ref, other) comparisons, per metric.
# H1 and its secondary tests were committed in 0889623 and are unchanged; amendment 1 added H2 (OCSD-v2) and its tests
# before OCSD-v2 existed.
_8P = ("count_bin", "8+")
PREREG = dict(
    methods=["ocsd_v2", "ocsd", "gligen", "controlnet", "ocsd_lite", "gligen_bon", "controlnet_bon"],
    families=[
        dict(name="H1", kind="primary", subset=_8P, metrics=["opr"], refs=["ocsd"], others=["gligen", "controlnet"],
             alternative="greater"),
        # the 8+ scenes that were not part of the first paper-tier comparison (E3), where the effect was first seen
        dict(name="new_scenes_8plus", kind="secondary", subset=_8P, new_only=True, metrics=["opr"], refs=["ocsd"],
             others=["gligen", "controlnet"], alternative="greater"),
        dict(name="all_scenes", kind="secondary", subset=None, metrics=["opr", "oce_c", "ra"], refs=["ocsd"],
             others=["gligen", "controlnet"], alternative="two-sided"),
        dict(name="identity_learning", kind="secondary", subset=None, metrics=["opr", "oce_c", "id_sim"], refs=["ocsd"],
             others=["ocsd_lite"], alternative="two-sided"),
        # ---- amendment 1
        dict(name="H2", kind="primary", subset=_8P, metrics=["opr"], refs=["ocsd_v2"], others=["gligen", "controlnet"],
             alternative="greater"),
        dict(name="new_scenes_8plus_v2", kind="secondary", subset=_8P, new_only=True, metrics=["opr"],
             refs=["ocsd_v2"], others=["gligen", "controlnet"], alternative="greater"),
        dict(name="all_scenes_v2", kind="secondary", subset=None, metrics=["opr", "oce_c", "ra"], refs=["ocsd_v2"],
             others=["gligen", "controlnet"], alternative="two-sided"),
        dict(name="compute_matched", kind="secondary", subset=None, metrics=["opr"], refs=["ocsd_v2"],
             others=["gligen_bon", "controlnet_bon"], alternative="greater"),
        dict(name="v2_vs_thesis", kind="secondary", subset=None, metrics=["opr", "oce_c", "ra"], refs=["ocsd_v2"],
             others=["ocsd"], alternative="two-sided"),
    ],
)


def prereg_tests(ps: pd.DataFrame, new_sids: Optional[set] = None) -> pd.DataFrame:
    """Run the PREREG families on per-scene means `ps` (already limited to the power job's scenes and seeds).
    `new_sids`: scenes outside the first E3 comparison (for the families with new_only). Column family =
    "<kind>:<name>", e.g. "primary:H1"."""
    out = []
    for f in PREREG["families"]:
        sub = ps if f["subset"] is None else ps[ps[f["subset"][0]] == f["subset"][1]]
        label = "all" if f["subset"] is None else f"{f['subset'][0]}={f['subset'][1]}"
        if f.get("new_only"):
            sub = sub[sub.sid.isin(new_sids or set())]
            label += ", new scenes"
        rows = [paired_tests(sub, r, f["metrics"], f["others"], f["alternative"]) for r in f["refs"]
                if r in set(sub.method)]
        rows = [r for r in rows if len(r)]
        if rows:
            t = holm(pd.concat(rows, ignore_index=True))
            out.append(t.assign(family=f"{f['kind']}:{f['name']}", subset=label, alternative=f["alternative"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------------------------------------------------------- formatting
def fmt(v, metric, ci=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if metric in PCT:
        s = f"{100 * v:.1f}"
        if ci is not None and not np.isnan(ci[0]):
            s += f" ± {100 * (ci[1] - ci[0]) / 2:.1f}"
        return s
    d = 2 if metric in ("oce", "oce_c", "fid", "kid", "time_s", "m2_s", "m3_s", "lpips_div") else 3
    if metric in ("clip", "obj_clip"):
        d = 2
    s = f"{v:.{d}f}"
    if ci is not None and not np.isnan(ci[0]):
        s += f" ± {(ci[1] - ci[0]) / 2:.{d}f}"
    return s


def table(summary: pd.DataFrame, methods: Sequence[str], metrics: Sequence[str], row_key="method",
          with_ci=True, bold_best=True) -> pd.DataFrame:
    rows = []
    sub = summary.set_index(row_key)
    best = {}
    for m in metrics:
        if m in sub:
            col = sub.loc[[x for x in methods if x in sub.index], m].astype(float)
            if len(col.dropna()):
                best[m] = col.max() if HIGHER.get(m, True) else col.min()
    for meth in methods:
        if meth not in sub.index:
            continue
        r = sub.loc[meth]
        row = {"Phương pháp": LABELS.get(meth, meth)}
        for m in metrics:
            if m not in sub:
                continue
            ci = (r.get(m + "_lo", np.nan), r.get(m + "_hi", np.nan)) if with_ci else None
            s = fmt(r[m], m, ci)
            if bold_best and m in best and np.isclose(r[m], best[m]):
                s = f"**{s}**"
            row[m] = s
        rows.append(row)
    return pd.DataFrame(rows)


HEAD = dict(opr="OPR (%) ↑", oce="OCE ↓", oce_c="OCE-lớp ↓", count_acc="Đếm đúng (%) ↑", miou="mIoU ↑",
            ra="RA (%) ↑", clip="CLIP ↑", obj_clip="CLIP-đt ↑", id_sim="ID-Sim ↑", fid="FID ↓", kid="KID×10³ ↓",
            lpips_div="LPIPS-đa dạng ↑", time_s="Thời gian (s)", m2_s="M2 (s)", m3_s="M3 (s)", peak_gb="VRAM (GB)",
            tries="Số lần sinh")


def save_table(df: pd.DataFrame, path_noext: str, caption: str = ""):
    os.makedirs(os.path.dirname(path_noext), exist_ok=True)
    df2 = df.rename(columns=HEAD)
    df2.to_csv(path_noext + ".csv", index=False)
    with open(path_noext + ".md", "w") as f:
        if caption:
            f.write(f"**{caption}**\n\n")
        f.write(df2.to_markdown(index=False))
    with open(path_noext + ".tex", "w") as f:
        f.write(df2.to_latex(index=False, escape=False, caption=caption or None))
    return df2


# ----------------------------------------------------------------------------- experiment tables
MAIN = ["controlnet", "t2i_adapter", "gligen", "cn_region", "cn_energy", "controlnet_bo3", "controlnet_bon", "gligen_bon",
        "collage", "collage_bo3", "zhang2025", "ocsd_lite", "ocsd", "ocsd_v2"]
ABLATION = ["ocsd", "abl_no_blend", "abl_alpha0", "ocsd_lite", "abl_no_region", "abl_no_energy", "abl_m5ab_both",
            "abl_no_scenecn", "abl_no_verify", "abl_no_m5", "abl_no_attsep", "abl_k1", "abl_m3_long", "abl_bg_only",
            "abl_global_only", "zhang2025"]
ALPHAS = [f"alpha_{a:.1f}" for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
ABLATION_V2 = ["ocsd_v2"] + list(V2_ABLATIONS)
ABLATION_V2_COCO = ["ocsd_v2"] + list(V2_COCO_ABLATIONS)
# control-vs-quality curves (job "pareto"): label -> methods along the curve
PARETO_CURVES = {
    "OCSD-v2 (α)": [f"pv2_a{a:.1f}" for a in PARETO["pv2"]],
    "OCSD-lite, thesis sampler (α)": [f"pv1lite_a{a:.1f}" for a in PARETO["pv1lite"]],
    "ControlNet (scale)": [f"pcn_s{v:.1f}" for v in PARETO["pcn"]],
    "GLIGEN (β)": [f"pgl_b{v:.1f}" for v in PARETO["pgl"]],
}
PARETO_METHODS = [m for ms in PARETO_CURVES.values() for m in ms]
EXTRA_DETECTORS = {"owlv2": "OWLv2", "owlv2d": "OWLv2 + competing queries", "detr": "DETR (COCO, closed set)"}


def build_all(results_dir: str, splits=("quickdraw", "coco"), quality: Optional[Dict[str, pd.DataFrame]] = None):
    """Sinh toàn bộ bảng cho Chương 4. Trả về dict tên -> DataFrame."""
    out = {}
    tdir = os.path.join(results_dir, "tables")
    for split in splits:
        df = load_per_image(results_dir, split)
        if not len(df):
            continue
        ps = per_scene(df)
        q = quality.get(split) if quality else None
        if q is None and os.path.exists(os.path.join(results_dir, split, "quality_fid_kid.csv")):
            q = pd.read_csv(os.path.join(results_dir, split, "quality_fid_kid.csv"))
        # ---- E3: so sánh tổng thể (chỉ trên các cảnh mà mọi phương pháp chính đều có ảnh)
        main = [m for m in MAIN if m in set(ps.method)]
        common = set.intersection(*[set(ps[ps.method == m].sid) for m in main]) if main else set()
        e3 = summarize(per_scene(common_seed_rows(df[df.sid.isin(common)], main)))
        if q is not None and len(q):
            e3 = e3.merge(q[["method", "fid", "kid"] + (["lpips_div"] if "lpips_div" in q else [])], on="method", how="left")
        mets = ["opr", "oce_c", "count_acc", "miou", "ra", "clip", "obj_clip", "id_sim", "fid", "kid"]
        out[f"E3_{split}"] = save_table(table(e3, main, [m for m in mets if m in e3]),
                                        os.path.join(tdir, f"E3_{split}"),
                                        f"So sánh định lượng trên {split} ({len(common)} cảnh, trung bình ± nửa KTC 95%)")
        # ---- E3b: các phương pháp không cần huấn luyện trên TOÀN BỘ tập (nhiều cảnh hơn)
        free = [m for m in main if m not in ("ocsd", "zhang2025")]
        if free:
            cf = set.intersection(*[set(ps[ps.method == m].sid) for m in free])
            if len(cf) > len(common):
                # same seeds for every method (the power job gives some of them a second seed)
                s = summarize(per_scene(common_seed_rows(df[df.sid.isin(cf)], free)))
                out[f"E3all_{split}"] = save_table(table(s, free, [m for m in mets if m in s]),
                                                   os.path.join(tdir, f"E3all_{split}"),
                                                   f"Các phương pháp không huấn luyện trên toàn bộ {split} ({len(cf)} cảnh)")
        # ---- power: PREREG methods on every scene with every seed, and the pre-registered tests
        pw = [m for m in PREREG["methods"] if m in set(ps.method)]
        if split == "quickdraw" and "ocsd" in pw and len(pw) > 1:
            pp = per_scene(common_seed_rows(df, pw))
            cp = set.intersection(*[set(pp[pp.method == m].sid) for m in pw])
            if len(cp) > len(common):
                pp = pp[pp.sid.isin(cp)]
                s = summarize(pp)
                out[f"E3power_{split}"] = save_table(
                    table(s, pw, [m for m in mets if m in s]), os.path.join(tdir, f"E3power_{split}"),
                    f"Pre-registered comparison on {split} ({len(cp)} scenes, every seed)")
                s8 = summarize(pp[pp.count_bin == _8P[1]])
                if len(s8):
                    out[f"E3power8_{split}"] = save_table(
                        table(s8, pw, [m for m in mets if m in s8]), os.path.join(tdir, f"E3power8_{split}"),
                        f"Pre-registered comparison on {split}, 8+ objects ({int(s8.n_scenes.max())} scenes)")
                pr = prereg_tests(pp, new_sids=cp - common)
                if len(pr):
                    out[f"prereg_{split}"] = save_table(
                        pr.round(4), os.path.join(tdir, f"prereg_{split}"),
                        "Pre-registered Wilcoxon tests (docs/PREREGISTRATION.md); Holm within each family and metric")
        # ---- E1 / E2: theo số đối tượng và độ phức tạp
        for dim, name in (("count_bin", "E1"), ("complexity", "E2")):
            s = summarize(ps[ps.method.isin(main) & ps.sid.isin(common)], by=("method", dim))
            if not len(s):
                continue
            wide = []
            for meth in main:
                row = {"Phương pháp": LABELS.get(meth, meth)}
                for v in _order(s[dim].unique()):
                    r = s[(s.method == meth) & (s[dim] == v)]
                    if len(r):
                        row[f"{v}"] = f"{fmt(r.opr.iloc[0], 'opr')} / {fmt(r.oce_c.iloc[0], 'oce_c')}"
                wide.append(row)
            out[f"{name}_{split}"] = save_table(pd.DataFrame(wide), os.path.join(tdir, f"{name}_{split}"),
                                                f"OPR (%) / OCE-lớp theo {dim} trên {split}")
            s.to_csv(os.path.join(tdir, f"{name}_{split}_long.csv"), index=False)
        # ---- kiểm định thống kê: one Holm family per question, each pair on the seeds both methods have
        pt = stats_families(df, main, common)
        if len(pt):
            pt.to_csv(os.path.join(tdir, f"stats_{split}.csv"), index=False)
            out[f"stats_{split}"] = pt
        # ---- E4: cắt bỏ
        ab = [m for m in ABLATION if m in set(ps.method)]
        if len(ab) > 1:
            cm = set.intersection(*[set(ps[ps.method == m].sid) for m in ab])
            # same seeds for every row: OCSD has every seed on these scenes, the ablation variants only the first
            s = summarize(per_scene(common_seed_rows(df[df.sid.isin(cm)], ab)))
            out[f"E4_{split}"] = save_table(table(s, ab, ["opr", "oce_c", "miou", "ra", "clip", "obj_clip", "id_sim"]),
                                            os.path.join(tdir, f"E4_{split}"),
                                            f"Nghiên cứu cắt bỏ trên {split} ({len(cm)} cảnh)")
        # ---- E4v2: OCSD-v2 ablations (QuickDraw) and the caption ablations (COCO)
        abv = [m for m in (ABLATION_V2 if split == "quickdraw" else ABLATION_V2_COCO) if m in set(ps.method)]
        if len(abv) > 1:
            cm = set.intersection(*[set(ps[ps.method == m].sid) for m in abv])
            s = summarize(per_scene(common_seed_rows(df[df.sid.isin(cm)], abv)))
            out[f"E4v2_{split}"] = save_table(table(s, abv, ["opr", "oce_c", "miou", "ra", "clip", "obj_clip", "id_sim"]),
                                              os.path.join(tdir, f"E4v2_{split}"),
                                              f"OCSD-v2 ablation on {split} ({len(cm)} scenes)")
        # ---- robustness: the same images under other evaluators
        rb = robustness_table(results_dir, split, df, main, common)
        if rb is not None:
            out[f"robustness_{split}"] = save_table(rb, os.path.join(tdir, f"robustness_{split}"),
                                                    f"OPR under three evaluators on {split} ({len(common)} scenes); "
                                                    "RA-cond / matched IoU only count detected objects")
        # ---- control-vs-quality sweeps
        pm = [m for m in PARETO_METHODS if m in set(ps.method)]
        if pm:
            cm = set.intersection(*[set(ps[ps.method == m].sid) for m in pm])
            s = summarize(per_scene(common_seed_rows(df[df.sid.isin(cm)], pm)), metrics=["opr", "oce_c", "miou", "clip"])
            if q is not None and len(q):
                s = s.merge(q[["method", "fid", "kid"]], on="method", how="left")
            out[f"pareto_{split}"] = save_table(table(s, pm, ["opr", "oce_c", "miou", "clip", "kid", "fid"], bold_best=False),
                                                os.path.join(tdir, f"pareto_{split}"),
                                                f"Control vs quality sweeps on {split} ({len(cm)} scenes)")
        # ---- alpha
        al = [m for m in ALPHAS if m in set(ps.method)]
        if al:
            cm = set.intersection(*[set(ps[ps.method == m].sid) for m in al])
            s = summarize(ps[ps.method.isin(al) & ps.sid.isin(cm)])
            if q is not None and len(q):
                s = s.merge(q[["method", "fid", "kid"]], on="method", how="left")
            out[f"alpha_{split}"] = save_table(table(s, al, ["opr", "oce_c", "miou", "clip", "fid", "kid"], bold_best=False),
                                               os.path.join(tdir, f"alpha_{split}"), f"Ảnh hưởng của alpha trên {split}")
        # ---- thời gian
        tcols = [c for c in ("time_s", "m2_s", "m3_s", "peak_gb", "tries") if c in ps]
        if tcols:
            rt = ps[ps.method.isin(main) & ps.sid.isin(common)].groupby("method")[tcols].mean().reindex(main).reset_index()
            rt["method"] = rt["method"].map(lambda m: LABELS.get(m, m))
            out[f"runtime_{split}"] = save_table(rt.round(2).rename(columns={"method": "Phương pháp"}),
                                                 os.path.join(tdir, f"runtime_{split}"), "Thời gian và bộ nhớ trung bình mỗi ảnh")
        # ---- trần của bộ phát hiện (ảnh thật)
        if "real" in set(ps.method):
            s = summarize(ps[ps.method == "real"])
            out[f"detector_ceiling_{split}"] = save_table(table(s, ["real"], ["opr", "oce_c", "miou", "ra", "clip"]),
                                                          os.path.join(tdir, f"detector_ceiling_{split}"),
                                                          "Độ đo trên chính ảnh thật (trần của bộ phát hiện)")
    return out


def stats_families(df: pd.DataFrame, main: Sequence[str], common: set) -> pd.DataFrame:
    """Wilcoxon tests in separate Holm families: every main method vs OCSD and vs OCSD-v2 on the common scenes, the
    thesis ablation vs OCSD, the v2 ablation vs OCSD-v2 and the alpha sweep vs OCSD (each pair on shared seeds)."""
    fams = [("main_vs_ocsd", "ocsd", [m for m in main if m != "ocsd"], common),
            ("main_vs_ocsd_v2", "ocsd_v2", [m for m in main if m != "ocsd_v2"], common),
            ("ablation_vs_ocsd", "ocsd", [m for m in ABLATION if m not in ("ocsd", *main)], None),
            ("ablation_v2_vs_ocsd_v2", "ocsd_v2", [m for m in ABLATION_V2 + ABLATION_V2_COCO if m != "ocsd_v2"], None),
            ("alpha_vs_ocsd", "ocsd", ALPHAS, None)]
    have = set(df.method)
    out = []
    for name, ref, others, scenes in fams:
        others = [m for m in others if m in have]
        if ref not in have or not others:
            continue
        d = df if scenes is None else df[df.sid.isin(scenes)]
        rows = [paired_tests(per_scene(common_seed_rows(d, [ref, m])), ref, others=[m]) for m in others]
        rows = [r for r in rows if len(r)]
        if rows:
            out.append(holm(pd.concat(rows, ignore_index=True)).assign(family=name))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def robustness_table(results_dir: str, split: str, df: pd.DataFrame, main: Sequence[str], common: set):
    """OPR of the main methods (and real photos on COCO) under the main evaluator and each extra evaluator
    (results/<split>/det_<name>), on the same images; RA-cond and matched IoU (only detected objects) from the first
    extra evaluator found; and Kendall's tau between each evaluator's method ranking and the main one.
    None when no extra evaluator has rows yet."""
    from scipy.stats import kendalltau
    keep = list(main) + ["real"]
    rows = {m: {"Phương pháp": LABELS.get(m, m)} for m in keep}
    pick = lambda d: d[d.method.isin(keep) & (d.sid.isin(common) | (d.method == "real"))]
    ref = per_scene(pick(df)).groupby("method").opr.mean()
    for m, v in ref.items():
        rows[m]["OPR (OWLv2)"] = fmt(v, "opr")
    taus, geo, found = {}, False, False
    for name, label in EXTRA_DETECTORS.items():
        d = load_per_image(results_dir, split, f"det_{name}")
        if not len(d):
            continue
        found = True
        g = per_scene(pick(d)).groupby("method")
        opr = g.opr.mean()
        o50 = g.opr50.mean() if "opr50" in d else pd.Series(dtype=float)
        for m, v in opr.items():
            if name != "owlv2":     # same detections as the main evaluator: only the stricter IoU is new
                rows[m][f"OPR ({label})"] = fmt(v, "opr")
            if m in o50.index:
                rows[m][f"OPR@0.5 ({label})"] = fmt(o50[m], "opr")
        if name == "owlv2":
            continue
        if not geo:
            geo = True
            for m, v in g.ra_cond.mean().items():
                rows[m][f"RA-cond ({label})"] = fmt(v, "ra_cond")
            for m, v in g.miou_matched.mean().items():
                rows[m][f"matched IoU ({label})"] = fmt(v, "miou")
        both = [m for m in main if m in ref.index and m in opr.index]
        if len(both) >= 3:
            taus[label] = kendalltau([ref[m] for m in both], [opr[m] for m in both])[0]
    if not found:
        return None
    tab = pd.DataFrame([r for r in rows.values() if len(r) > 1])
    if taus:
        tab = pd.concat([tab, pd.DataFrame([{"Phương pháp": "Kendall τ vs the OWLv2 ranking",
                                             **{f"OPR ({k})": f"{v:.2f}" for k, v in taus.items()}}])],
                        ignore_index=True)
    return tab


def _order(vals):
    pref = ["1", "2-3", "3", "4-5", "5", "6-8", "8+", "simple", "medium", "complex", "real"]
    return sorted(vals, key=lambda v: pref.index(v) if v in pref else 99)


# ----------------------------------------------------------------------------- figures
def plot_curves(results_dir: str, split: str = "quickdraw",
                methods=("controlnet", "gligen", "zhang2025", "ocsd_lite", "ocsd", "ocsd_v2"),
                out: Optional[str] = None):
    import matplotlib.pyplot as plt
    ps = per_scene(load_per_image(results_dir, split))
    present = [m for m in methods if m in set(ps.method)]
    if present:   # same scenes for every curve
        ps = ps[ps.sid.isin(set.intersection(*[set(ps[ps.method == m].sid) for m in present]))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dim, title in ((axes[0], "count_bin", "Số lượng đối tượng"), (axes[1], "complexity", "Độ phức tạp phác thảo")):
        s = summarize(ps[ps.method.isin(methods)], by=("method", dim), metrics=["opr"])
        for m in methods:
            g = s[s.method == m]
            if not len(g):
                continue
            order = _order(g[dim].unique())
            g = g.set_index(dim).loc[order]
            ax.plot(order, 100 * g.opr, marker="o", label=LABELS.get(m, m))
            ax.fill_between(order, 100 * g.opr_lo, 100 * g.opr_hi, alpha=0.15)
        ax.set_xlabel(title)
        ax.set_ylabel("OPR (%)")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = out or os.path.join(results_dir, "figures", f"opr_curves_{split}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    return out


def plot_alpha(results_dir: str, split: str = "quickdraw", out: Optional[str] = None):
    import matplotlib.pyplot as plt
    ps = per_scene(load_per_image(results_dir, split))
    s = summarize(ps[ps.method.isin(ALPHAS)], metrics=["opr", "miou", "clip"])
    if not len(s):
        return None
    s["alpha"] = s.method.str.replace("alpha_", "").astype(float)
    s = s.sort_values("alpha")
    q = os.path.join(results_dir, split, "quality_fid_kid.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s.alpha, 100 * s.opr, "o-", label="OPR (%)")
    ax.plot(s.alpha, 100 * s.miou, "s-", label="mIoU × 100")
    ax.set_xlabel("α")
    ax.grid(alpha=0.3)
    if os.path.exists(q):
        qq = pd.read_csv(q)
        qq = qq[qq.method.isin(ALPHAS)].copy()
        if len(qq):
            qq["alpha"] = qq.method.str.replace("alpha_", "").astype(float)
            qq = qq.sort_values("alpha")
            ax2 = ax.twinx()
            ax2.plot(qq.alpha, qq.fid, "d--", color="gray", label="FID")
            ax2.set_ylabel("FID ↓")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = out or os.path.join(results_dir, "figures", f"alpha_{split}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    return out


def plot_pareto(results_dir: str, split: str = "quickdraw", out: Optional[str] = None):
    """OPR against KID and against global CLIP along each sweep (PARETO_CURVES), same scenes and seed for every
    point. A method dominates where its curve is up-left (KID) / up-right (CLIP) of the others."""
    import matplotlib.pyplot as plt
    df = load_per_image(results_dir, split)
    if not len(df):
        return None
    pm = [m for m in PARETO_METHODS if m in set(df.method)]
    if not pm:
        return None
    cm = set.intersection(*[set(df[df.method == m].sid) for m in pm])
    s = per_scene(common_seed_rows(df[df.sid.isin(cm)], pm)).groupby("method")[["opr", "clip"]].mean()
    qp = os.path.join(results_dir, split, "quality_fid_kid.csv")
    kid = pd.read_csv(qp).set_index("method")["kid"] if os.path.exists(qp) else pd.Series(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, ms in PARETO_CURVES.items():
        ms = [m for m in ms if m in s.index]
        if not ms:
            continue
        tags = [m.split("_", 1)[1][1:] for m in ms]
        y = [100 * s.loc[m, "opr"] for m in ms]
        for ax, x in ((axes[0], [float(kid.get(m, np.nan)) for m in ms]), (axes[1], [s.loc[m, "clip"] for m in ms])):
            ax.plot(x, y, "o-", label=label)
            for xi, yi, tg in zip(x, y, tags):
                if not np.isnan(xi):
                    ax.annotate(tg, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel("KID ×10³ ↓")
    axes[1].set_xlabel("global CLIP ↑")
    for ax in axes:
        ax.set_ylabel("OPR (%) ↑")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Control vs quality on {split} ({len(cm)} scenes; labels = α / scale / β)", fontsize=9)
    fig.tight_layout()
    out = out or os.path.join(results_dir, "figures", f"pareto_{split}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    return out


def realism_pairs_pack(out_root: str, split: str, sids: Sequence[str], pairs: Sequence[tuple], out_dir: str,
                       seed: int = 0, rng_seed: int = 0):
    """Two-alternative realism study: for each scene and pair (a, b), the two images in random left/right order
    ("Which image looks more like a real photo?"), plus an answer key CSV."""
    import random
    import shutil
    rng = random.Random(rng_seed)
    os.makedirs(out_dir, exist_ok=True)
    key = []
    for sid in sids:
        for a, b in pairs:
            src = {m: os.path.join(out_root, split, m, f"{sid}_s{seed}.png") for m in (a, b)}
            if not all(os.path.exists(v) for v in src.values()):
                continue
            left, right = (a, b) if rng.random() < 0.5 else (b, a)
            qd = os.path.join(out_dir, f"P{len(key) + 1:03d}")
            os.makedirs(qd, exist_ok=True)
            shutil.copy(src[left], os.path.join(qd, "left.png"))
            shutil.copy(src[right], os.path.join(qd, "right.png"))
            key.append(dict(pair=len(key) + 1, sid=sid, left=left, right=right))
    pd.DataFrame(key).to_csv(os.path.join(out_dir, "answer_key.csv"), index=False)
    return out_dir


def qualitative_grid(out_root: str, bench_dir: str, split: str, sids: Sequence[str], methods: Sequence[str],
                     seed: int = 0, out: Optional[str] = None, cell: int = 256):
    """Lưới ảnh: mỗi hàng một cảnh (phác thảo + câu mô tả), mỗi cột một phương pháp."""
    import cv2
    import matplotlib.pyplot as plt
    from .data import load_scene
    cols = ["sketch"] + list(methods)
    fig, axes = plt.subplots(len(sids), len(cols), figsize=(2.2 * len(cols), 2.4 * len(sids)))
    axes = np.atleast_2d(axes)
    for r, sid in enumerate(sids):
        sc = load_scene(os.path.join(bench_dir, split, sid))
        for c, m in enumerate(cols):
            ax = axes[r, c]
            ax.axis("off")
            if m == "sketch":
                ax.imshow(sc.sketch, cmap="gray")
                ax.set_title("\n".join([sc.caption[i:i + 34] for i in range(0, min(len(sc.caption), 102), 34)]),
                             fontsize=6)
            else:
                p = os.path.join(out_root, split, m, f"{sid}_s{seed}.png")
                if os.path.exists(p):
                    ax.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
                if r == 0:
                    ax.set_title(LABELS.get(m, m).split(" (")[0], fontsize=7)
    fig.tight_layout()
    out = out or os.path.join(os.path.dirname(os.path.dirname(out_root.rstrip("/"))), "results", "figures",
                              f"qualitative_{split}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    return out


def pick_showcase(results_dir: str, split: str, a: str = "ocsd", b: str = "controlnet", k: int = 6) -> List[str]:
    """Chọn các cảnh mà `a` hơn `b` rõ nhất về OPR (dùng cho hình định tính / phân tích lỗi), phân tầng theo số đối tượng."""
    ps = per_scene(load_per_image(results_dir, split))
    x = ps[ps.method == a].set_index("sid")
    y = ps[ps.method == b].set_index("sid")
    common = x.index.intersection(y.index)
    d = (x.loc[common, "opr"] - y.loc[common, "opr"]).sort_values(ascending=False)
    bins = x.loc[common, "count_bin"]
    picked = []
    for cb in _order(bins.unique()):
        cand = [s for s in d.index if bins[s] == cb and s not in picked]
        if cand:
            picked.append(cand[0])
    for s in d.index:
        if len(picked) >= k:
            break
        if s not in picked:
            picked.append(s)
    return picked[:k]


def user_study_pack(out_root: str, bench_dir: str, split: str, sids: Sequence[str], methods: Sequence[str],
                    out_dir: str, seed: int = 0, rng_seed: int = 0):
    """Tạo bộ ảnh khảo sát người dùng ẩn danh (thứ tự phương pháp xáo trộn) + khóa giải mã CSV."""
    import random
    import shutil
    rng = random.Random(rng_seed)
    os.makedirs(out_dir, exist_ok=True)
    key = []
    for q, sid in enumerate(sids):
        qd = os.path.join(out_dir, f"Q{q + 1:02d}")
        os.makedirs(qd, exist_ok=True)
        shutil.copy(os.path.join(bench_dir, split, sid, "sketch.png"), os.path.join(qd, "sketch.png"))
        sc = json.load(open(os.path.join(bench_dir, split, sid, "scene.json")))
        open(os.path.join(qd, "caption.txt"), "w").write(sc["caption"])
        ms = list(methods)
        rng.shuffle(ms)
        for j, m in enumerate(ms):
            src = os.path.join(out_root, split, m, f"{sid}_s{seed}.png")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(qd, f"{chr(65 + j)}.png"))
                key.append(dict(question=q + 1, sid=sid, option=chr(65 + j), method=m))
    pd.DataFrame(key).to_csv(os.path.join(out_dir, "answer_key.csv"), index=False)
    return out_dir
