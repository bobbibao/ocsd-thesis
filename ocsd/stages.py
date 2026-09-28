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
def setup_data(P: Paths, tier: str = "paper", with_coco: bool = True):
    P.makedirs()
    t = TIERS[tier]
    download_quickdraw(P.quickdraw_raw, per_class=3000)
    build_quickdraw_scenes(P.quickdraw_raw, os.path.join(P.benchmarks, "quickdraw"), per_cell=t["qd_per_cell"])
    if with_coco:
        local = "/content/coco"   # giải nén ra đĩa cục bộ (nhanh hơn Drive), chỉ lưu kết quả lên Drive
        os.makedirs(local, exist_ok=True)
        if not os.path.exists(f"{local}/annotations/instances_val2017.json"):
            _sh(f"wget -q -c {COCO_URLS['ann']} -O {local}/ann.zip && unzip -q -o {local}/ann.zip -d {local}")
        if not os.path.exists(f"{local}/val2017"):
            _sh(f"wget -q -c {COCO_URLS['val']} -O {local}/val.zip && unzip -q -o {local}/val.zip -d {local}")
        build_coco_sketch(local, os.path.join(P.benchmarks, "coco"), n_scenes=t["coco_n"],
                          ref_dir=os.path.join(P.data, "coco_ref"), n_ref=2000)
    preview_benchmark(P)


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


# ============================================================================ 2. generate
def experiment_plan(P: Paths, E: ExperimentConfig) -> List[Dict]:
    """Danh sách công việc. Thứ tự: E3 QuickDraw -> E3 COCO -> E4 -> alpha (kết quả quan trọng nhất có trước)."""
    t = TIERS[E.tier]
    main = [m for m in MAIN if not (m == "gligen" and E.backbone != "sd15")]
    main = [m for m in main if not (m == "t2i_adapter" and E.backbone != "sd15")]
    qd_all = list_scenes(P.benchmarks, "quickdraw")
    coco_sub = list_scenes(P.benchmarks, "coco", limit=t["full_identity_n"], seed=1)
    ocsd_full_qd = list_scenes(P.benchmarks, "quickdraw", limit=t["full_identity_n"], seed=1)
    plan = [
        # OCSD đầy đủ (có học định danh, tốn thời gian nhất) chạy trên tập con phân tầng; mọi phương pháp khác
        # chạy trên toàn bộ tập. Bảng E3 so sánh trên giao các cảnh chung (report.build_all tự xử lý).
        dict(name="E3_quickdraw_all", split="quickdraw", scenes=qd_all, seeds=E.seeds,
             methods=[m for m in main if m not in ("ocsd", "zhang2025")]),
        dict(name="E3_quickdraw_trained", split="quickdraw", scenes=ocsd_full_qd, seeds=E.seeds,
             methods=["ocsd", "zhang2025"]),
        dict(name="E3_coco", split="coco", scenes=coco_sub, seeds=E.seeds[:2], methods=main),
        dict(name="E4_ablation", split="quickdraw",
             scenes=list_scenes(P.benchmarks, "quickdraw", count_bins=["3", "5"], limit=t["ablation_n"], seed=2),
             seeds=E.seeds[:2], methods=[m for m in ABLATION if m in OCSD_VARIANTS]),
        dict(name="alpha", split="quickdraw",
             scenes=list_scenes(P.benchmarks, "quickdraw", count_bins=["3", "5"], limit=t["alpha_n"], seed=3),
             seeds=E.seeds[:1], methods=ALPHAS),
    ]
    return [p for p in plan if p["scenes"]]


def _count_todo(P, job):
    n = 0
    for sdir in job["scenes"]:
        sid = os.path.basename(sdir)
        for m in job["methods"]:
            for s in job["seeds"]:
                if not os.path.exists(os.path.join(P.outputs, job["split"], m, f"{sid}_s{s}.png")):
                    n += 1
    return n


def generate(P: Paths, E: ExperimentConfig, only: Optional[List[str]] = None, max_minutes: Optional[float] = None,
             eng=None, vis=None):
    import torch
    from .engine import Engine
    from .runner import Runner
    from .vision import Vision
    os.environ.setdefault("HF_HOME", P.cache)
    if eng is None:
        eng = Engine.from_pretrained(E.backbone, "cuda", E.fp16, cache_dir=None)
    if vis is None:
        vis = Vision("cuda")
    runner = Runner(eng, vis, P.outputs, E.backbone)
    t0 = time.time()
    for job in experiment_plan(P, E):
        if only and job["name"] not in only:
            continue
        todo = _count_todo(P, job)
        print(f"\n===== {job['name']}: {len(job['scenes'])} cảnh x {len(job['methods'])} phương pháp x "
              f"{len(job['seeds'])} seed, còn {todo} ảnh")
        if todo == 0:
            continue
        left = None if max_minutes is None else max_minutes - (time.time() - t0) / 60
        if left is not None and left <= 0:
            break
        runner.run(job["scenes"], job["methods"], job["seeds"], job["split"], max_minutes=left)
    log_progress(P, E)
    return eng, vis


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
    vis = vis or Vision("cuda")
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
                image_level_quality(P.outputs, P.benchmarks, split, methods, seeds, ref, P.results)
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
