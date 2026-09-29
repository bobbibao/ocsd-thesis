"""Đánh giá: OPR, OCE, độ chính xác đếm, mIoU, RA (qua bộ phát hiện OWLv2 - độc lập với phương pháp),
CLIP score toàn ảnh và cấp đối tượng, ID-Sim (DINOv2), FID/KID, LPIPS (đa dạng giữa các seed).

Kết quả từng ảnh được ghi dần ra CSV (có thể tiếp tục khi bị ngắt)."""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch

from .data import load_scene
from .matching import consistency
from .methods import OCSD_VARIANTS, m2_key


def _read(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def _crop(img, box, margin=0.1, size=224):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, y0 = int(max(0, x0 - margin * w)), int(max(0, y0 - margin * h))
    x1, y1 = int(min(W, x1 + margin * w)), int(min(H, y1 + margin * h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(img[y0:y1, x0:x1], (size, size))


def evaluate_images(out_root: str, bench_dir: str, split: str, methods: Sequence[str], seeds: Sequence[int],
                    vis, results_dir: str, detector: str = "owlv2", det_thr: float = 0.3, iou_thr: float = 0.1,
                    include_real: bool = True, scene_dirs: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Tính độ đo cho từng ảnh; lưu results/<split>/per_image_<method>.csv (bỏ qua ảnh đã tính)."""
    os.makedirs(os.path.join(results_dir, split), exist_ok=True)
    scene_dirs = scene_dirs or sorted(glob.glob(os.path.join(bench_dir, split, "*")))
    scenes = {os.path.basename(d): d for d in scene_dirs if os.path.exists(os.path.join(d, "scene.json"))}
    det_fn = getattr(vis, detector)
    all_rows = []
    meths = list(methods) + (["real"] if include_real and split == "coco" else [])
    for m in meths:
        csv = os.path.join(results_dir, split, f"per_image_{m}.csv")
        old = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
        done = set(zip(old["sid"], old["seed"])) if len(old) else set()
        rows = []
        n_err = 0
        for sid, sdir in scenes.items():
            for s in (seeds if m != "real" else [0]):
                if (sid, s) in done:
                    continue
                p = os.path.join(sdir, "real.png") if m == "real" else os.path.join(out_root, split, m, f"{sid}_s{s}.png")
                if not os.path.exists(p):
                    continue
                try:
                    row = _eval_one(m, sid, s, sdir, p, split, vis, det_fn, det_thr, iou_thr, out_root)
                except Exception as e:
                    from .runlog import log_exception
                    log_exception("evaluate_image", e, context=f"{split}/{m}/{sid}_s{s}")
                    n_err += 1
                    if n_err >= 5 and not rows:
                        raise
                    continue
                rows.append(row)
        new = pd.DataFrame(rows)
        df = pd.concat([old, new], ignore_index=True) if len(old) else new
        if len(df):
            df.to_csv(csv, index=False)
        all_rows.append(df)
        print(f"{split}/{m}: {len(df)} ảnh đã đánh giá (+{len(new)})")
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def _eval_one(m, sid, s, sdir, p, split, vis, det_fn, det_thr, iou_thr, out_root):
    scene = load_scene(sdir)
    if hasattr(vis, "set_scene"):
        vis.set_scene(scene)
    img = _read(p)
    dets = det_fn(img, [o.cls for o in scene.objects], thr=det_thr)
    c = consistency(scene, dets, iou_thr)
    row = dict(method=m, sid=sid, seed=s, split=split, n_obj=scene.n,
               count_bin=scene.meta["count_bin"], complexity=scene.meta["complexity"],
               opr=c["opr"], oce=c["oce"], oce_c=c["oce_c"], count_acc=c["count_acc"], miou=c["miou"],
               ra=c["ra"], n_gen=c["n_gen"], n_preserved=c["n_preserved"],
               missing="|".join(c["missing"]))
    row["clip"] = float(vis.clip.score([img], [scene.caption])[0])
    crops = [_crop(img, o.box) for o in scene.objects]
    row["obj_clip"] = float(np.mean(vis.clip.score(crops, [f"a photo of a {o.phrase}" for o in scene.objects])))
    row["id_sim"] = _id_sim(out_root, split, m, scene, img, vis)
    lp = os.path.join(out_root, split, m, f"{sid}_s{s}.json")
    if os.path.exists(lp):
        lg = json.load(open(lp))
        row["time_s"] = lg.get("scene_time")
        pt = lg.get("prep_times") or {}
        row["m2_s"] = pt.get("m2")
        row["m3_s"] = pt.get("m3")
        row["tries"] = lg.get("tries", 1)
        row["peak_gb"] = lg.get("peak_gb")
    return row


def _id_sim(out_root, split, method, scene, img, vis) -> float:
    """Độ tương đồng DINOv2 giữa vùng đối tượng trong cảnh và ảnh đối tượng ở M2 (chỉ phương pháp hai nhánh)."""
    key = None
    lp = os.path.join(out_root, split, method, f"{scene.sid}_s0.json")
    if os.path.exists(lp):
        key = json.load(open(lp)).get("m2_key")
    if key is None and method in OCSD_VARIANTS:
        key = m2_key(OCSD_VARIANTS[method])
    if key is None:
        return float("nan")
    d = os.path.join(out_root, split, "_objects", key, scene.sid)
    if not os.path.exists(d):
        return float("nan")
    refs, gens = [], []
    for i, o in enumerate(scene.objects):
        p = os.path.join(d, f"obj_{i:02d}.png")
        if not os.path.exists(p):
            continue
        ref = _read(p)
        mask = cv2.imread(os.path.join(d, f"mask_{i:02d}.png"), cv2.IMREAD_GRAYSCALE) > 127
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        refs.append(_crop(np.where(mask[..., None], ref, 255).astype(np.uint8),
                          (xs.min(), ys.min(), xs.max(), ys.max()), 0.05))
        gens.append(_crop(img, o.box, 0.05))
    if not refs:
        return float("nan")
    a, b = vis.dino.emb(refs), vis.dino.emb(gens)
    return float((a * b).sum(-1).mean())


# ----------------------------------------------------------------------------- FID / KID / LPIPS
def _load_uint8_batch(paths, size=299):
    x = np.stack([cv2.resize(_read(p), (size, size), interpolation=cv2.INTER_AREA) for p in paths])
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def fid_kid(gen_paths: Sequence[str], ref_paths: Sequence[str], device="cuda", batch=50) -> Dict[str, float]:
    """FID và KID (torchmetrics, Inception-v3 2048 chiều). KID ổn định hơn với tập nhỏ."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    n = min(len(gen_paths), len(ref_paths))
    if n < 10:
        return dict(fid=float("nan"), kid=float("nan"), kid_std=float("nan"), n_gen=len(gen_paths))
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(subset_size=min(100, n), normalize=False).to(device)
    for paths, real in ((ref_paths, True), (gen_paths, False)):
        for i in range(0, len(paths), batch):
            x = _load_uint8_batch(paths[i:i + batch]).to(device)
            fid.update(x, real=real)
            kid.update(x, real=real)
    km, ks = kid.compute()
    return dict(fid=float(fid.compute()), kid=float(km) * 1000, kid_std=float(ks) * 1000, n_gen=len(gen_paths))


def lpips_diversity(out_root, split, method, sids, seeds, device="cuda") -> float:
    """LPIPS trung bình giữa các cặp ảnh cùng cảnh khác seed (cao = đa dạng hơn)."""
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    vals = []
    for sid in sids:
        ps = [os.path.join(out_root, split, method, f"{sid}_s{s}.png") for s in seeds]
        ps = [p for p in ps if os.path.exists(p)]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a = torch.from_numpy(_read(ps[i])).permute(2, 0, 1)[None].float().div(255).to(device)
                b = torch.from_numpy(_read(ps[j])).permute(2, 0, 1)[None].float().div(255).to(device)
                vals.append(float(lp(a, b)))
    return float(np.mean(vals)) if vals else float("nan")


def image_level_quality(out_root, bench_dir, split, methods, seeds, ref_paths, results_dir, device="cuda",
                        with_lpips=True, scene_ids=None, pairs=None) -> pd.DataFrame:
    """FID/KID (+ LPIPS diversity) per method over its images of `scene_ids` (all scenes if None) and `seeds`.
    `pairs` (method -> set of (sid, seed)) overrides both per method, so methods can be scored on the same images."""
    import hashlib
    rows = []
    prev_p = os.path.join(results_dir, split, "quality_fid_kid.csv")
    prev = pd.read_csv(prev_p).set_index("method") if os.path.exists(prev_p) else pd.DataFrame()
    for m in methods:
        gen = sorted(glob.glob(os.path.join(out_root, split, m, "*.png")))
        if pairs is not None and m in pairs:
            gen = [p for p in gen if tuple(os.path.basename(p)[:-4].rsplit("_s", 1)) in
                   {(sid, str(s)) for sid, s in pairs[m]}]
        else:
            keep = {f"_s{s}.png" for s in seeds}
            gen = [p for p in gen if any(p.endswith(k) for k in keep)
                   and (scene_ids is None or os.path.basename(p).rsplit("_s", 1)[0] in scene_ids)]
        if not gen:
            continue
        sel = hashlib.md5("|".join(os.path.basename(p) for p in gen).encode()).hexdigest()[:12]
        if m in prev.index and "sel" in prev.columns and prev.loc[m, "sel"] == sel:   # same images -> reuse
            rows.append(dict(prev.loc[m].to_dict(), method=m))
            continue
        r = dict(method=m, split=split, **fid_kid(gen, ref_paths, device))
        r["sel"] = sel
        sd = sorted({int(os.path.basename(p)[:-4].rsplit("_s", 1)[1]) for p in gen})
        if with_lpips and len(sd) > 1:
            sids = sorted({os.path.basename(p).rsplit("_s", 1)[0] for p in gen})
            r["lpips_div"] = lpips_diversity(out_root, split, m, sids, sd, device)
        rows.append(r)
        print(r)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(results_dir, split), exist_ok=True)
    df.to_csv(os.path.join(results_dir, split, "quality_fid_kid.csv"), index=False)
    return df
