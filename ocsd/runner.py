"""Chạy sinh ảnh cho (cảnh x phương pháp x seed), CÓ THỂ TIẾP TỤC sau khi Colab ngắt kết nối:
ảnh đã có trên Drive sẽ được bỏ qua. Các biến thể OCSD dùng chung bước chuẩn bị (M2, M3) của cùng một cảnh.

Cấu trúc đầu ra:
  outputs/<split>/<method>/<sid>_s<seed>.png  (+ .json nhật ký: thời gian, số lần sinh lại, cấu hình)
  outputs/<split>/_objects/<m2key>/<sid>/obj_XX.png, mask_XX.png, fg.png   (ảnh đối tượng M2, dùng cho ID-Sim)
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .config import OCSDConfig
from .data import load_scene
from .engine import Engine
from .method import ObjectResult, Prepared, compose_foreground, generate, learn_identity, object_branch
from .methods import BASELINES, COLLAGE_BASELINES, OCSD_VARIANTS, m2_dir_key, m3_key, run_baseline, run_collage
from .sketch import CropInfo, Scene


def _save_png(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


class Runner:
    def __init__(self, eng: Engine, vis, out_root: str, backbone: str = "sd15", cache_dir: Optional[str] = None,
                 variants: Optional[Dict[str, OCSDConfig]] = None, base_cfg: Optional[OCSDConfig] = None,
                 baselines: Optional[Dict[str, Tuple[str, OCSDConfig]]] = None):
        """variants: OCSD-family method -> config. baselines: method -> (baseline kind, config), e.g. a tuned
        baseline, a sweep point "pcn_s0.7" -> ("controlnet", cfg with cn_scale 0.7) or a tuning-grid entry
        (default: every kind in methods.BASELINES with base_cfg)."""
        self.eng, self.vis, self.out_root = eng, vis, out_root
        self.backbone, self.cache_dir = backbone, cache_dir
        self.variants = dict(OCSD_VARIANTS if variants is None else variants)
        self.base_cfg = base_cfg or OCSDConfig()
        self.baselines = dict(baselines) if baselines is not None else {b: (b, self.base_cfg) for b in BASELINES}

    # ------------------------------------------------------------------ paths
    def img_path(self, split, method, sid, seed):
        return os.path.join(self.out_root, split, method, f"{sid}_s{seed}.png")

    def obj_dir(self, split, key, sid):
        return os.path.join(self.out_root, split, "_objects", key, sid)

    # ------------------------------------------------------------------ M2 cache
    def _load_objs(self, split, key, scene: Scene) -> Optional[List[ObjectResult]]:
        d = self.obj_dir(split, key, scene.sid)
        meta = os.path.join(d, "objects.json")
        if not os.path.exists(meta):
            return None
        m = json.load(open(meta))
        out = []
        for i, r in enumerate(m["objects"]):
            img = cv2.cvtColor(cv2.imread(os.path.join(d, f"obj_{i:02d}.png")), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(d, f"mask_{i:02d}.png"), cv2.IMREAD_GRAYSCALE) > 127
            sk = cv2.imread(os.path.join(d, f"sk_{i:02d}.png"), cv2.IMREAD_GRAYSCALE)
            out.append(ObjectResult(img, mask, CropInfo(**r["info"]), r["score"], r["candidates"], sk))
        return out

    def _save_objs(self, split, key, scene: Scene, objs: List[ObjectResult], m2_time: float):
        d = self.obj_dir(split, key, scene.sid)
        os.makedirs(d, exist_ok=True)
        for i, r in enumerate(objs):
            _save_png(os.path.join(d, f"obj_{i:02d}.png"), r.img)
            cv2.imwrite(os.path.join(d, f"mask_{i:02d}.png"), (r.mask * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(d, f"sk_{i:02d}.png"), r.sketch_crop)
        json.dump(dict(m2_time=m2_time, objects=[dict(info=r.info.__dict__, score=r.score, candidates=r.candidates)
                                                 for r in objs]), open(os.path.join(d, "objects.json"), "w"))

    # ------------------------------------------------------------------ main loop
    def run(self, scene_dirs: Sequence[str], methods: Sequence[str], seeds: Sequence[int], split: str,
            max_minutes: Optional[float] = None, verbose: bool = True):
        items = [(split, d, [(m, s) for m in methods for s in seeds]) for d in scene_dirs]
        return self.run_items(items, max_minutes, verbose)

    def run_items(self, items, max_minutes: Optional[float] = None, verbose: bool = True):
        """items: [(split, scene_dir, [(method, seed), ...])]. Mỗi cảnh xử lý một lần cho mọi (method, seed)."""
        t_start = time.time()
        stats = dict(done=0, skipped=0, failed=0)
        for si, (split, sdir, pairs) in enumerate(items):
            if max_minutes and (time.time() - t_start) / 60 > max_minutes:
                print(f"Dừng do hết ngân sách thời gian ({max_minutes} phút). Chạy lại ô này để tiếp tục.")
                break
            scene = load_scene(sdir)
            todo = [(m, s) for m, s in pairs if not os.path.exists(self.img_path(split, m, scene.sid, s))]
            stats["skipped"] += len(pairs) - len(todo)
            if not todo:
                continue
            if verbose:
                el = (time.time() - t_start) / 60
                print(f"[{si + 1}/{len(items)}] {split}/{scene.sid} (n={scene.n}): {len(todo)} ảnh | {el:.0f} phút")
            try:
                self._run_scene(scene, todo, split, stats)
            except Exception as e:
                stats["failed"] += 1
                traceback.print_exc()
                from .runlog import log_exception
                log_exception("generate_scene", e, context=f"{split}/{scene.sid} todo={todo}")
                if stats["failed"] >= 3 and stats["done"] == 0:
                    raise RuntimeError("The first 3 scenes all failed; stopping so the error can be fixed "
                                       "(see results/logs/LATEST_ERROR.txt)") from e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        print("Tổng kết:", stats, f"- {(time.time() - t_start) / 60:.1f} phút")
        return stats

    def _write(self, split, method, scene, seed, img, log, cfg=None):
        _save_png(self.img_path(split, method, scene.sid, seed), img)
        rec = dict(method=method, sid=scene.sid, seed=seed, n_obj=scene.n, **log)
        if cfg is not None:
            rec["cfg"] = cfg.to_dict()
        json.dump(_jsonable(rec), open(self.img_path(split, method, scene.sid, seed)[:-4] + ".json", "w"))

    def _objects(self, split, scene: Scene, cfg: OCSDConfig, seed: int):
        """M2 objects of `cfg` for this scene (cached on disk); drawn again per seed when cfg.m2_per_seed.
        Returns (cache key, objects, M2 time of the cached set)."""
        key = m2_dir_key(cfg, seed)
        objs = self._load_objs(split, key, scene)
        if objs is None:
            t0 = time.time()
            objs = object_branch(self.eng, self.vis, scene, cfg, seed=seed if cfg.m2_per_seed else 0)
            m2_time = time.time() - t0
            self._save_objs(split, key, scene, objs, m2_time)
        else:
            m2_time = json.load(open(os.path.join(self.obj_dir(split, key, scene.sid), "objects.json"))).get("m2_time")
        return key, objs, m2_time

    def _run_scene(self, scene: Scene, todo, split, stats):
        eng, vis = self.eng, self.vis
        if hasattr(vis, "set_scene"):   # chỉ dùng trong kiểm thử với bộ phát hiện giả
            vis.set_scene(scene)
        unknown = [x for x in todo if x[0] not in self.variants and x[0] not in self.baselines]
        if unknown:
            raise KeyError(f"Phương pháp không xác định: {sorted({m for m, _ in unknown})}")
        two_branch = [x for x in todo if x[0] in self.baselines and self.baselines[x[0]][0] in COLLAGE_BASELINES]
        # ---- baseline một nhánh
        for m, s in [x for x in todo if x[0] in self.baselines and x not in two_branch]:
            kind, cfg = self.baselines[m]
            log = {}
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            img = run_baseline(kind, eng, vis, scene, s, cfg, self.backbone, self.cache_dir, log)
            self._write(split, m, scene, s, img, log, cfg)
            stats["done"] += 1
        # ---- họ OCSD: nhóm theo bộ đối tượng M2 (theo seed nếu m2_per_seed) rồi khóa M3
        by_m2: Dict[str, list] = {}
        for m, s in [x for x in todo if x[0] in self.variants]:
            by_m2.setdefault(m2_dir_key(self.variants[m], s), []).append((m, s))
        for k2, items in by_m2.items():
            cfg0 = self.variants[items[0][0]]
            _, objs, m2_time = self._objects(split, scene, cfg0, items[0][1])
            fg = compose_foreground(scene, objs, cfg0)
            if not os.path.exists(os.path.join(self.obj_dir(split, k2, scene.sid), "fg.png")):
                _save_png(os.path.join(self.obj_dir(split, k2, scene.sid), "fg.png"), fg[0])
            by_m3: Dict[str, list] = {}
            for m, s in items:
                by_m3.setdefault(m3_key(self.variants[m]), []).append((m, s))
            for k3, its in by_m3.items():
                c3 = self.variants[its[0][0]]
                prep_log = {}
                id_tokens = None
                m3_time = 0.0
                if c3.use_identity:
                    t0 = time.time()
                    id_tokens = learn_identity(eng, scene, objs, c3, seed=0, log=prep_log)
                    m3_time = time.time() - t0
                prep = Prepared(objs, fg, id_tokens, dict(m2=m2_time, m3=m3_time))
                for m, s in its:
                    cfg = self.variants[m]
                    log = dict(prep_times=prep.times, m2_key=k2, m3_key=k3, **prep_log)
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    img = generate(eng, vis, scene, prep, cfg, s, log=log)
                    self._write(split, m, scene, s, img, log, cfg)
                    stats["done"] += 1
                eng.reset_identity()
        # ---- baseline hai nhánh (collage): dùng bộ đối tượng M2 của chính cấu hình đó
        for m, s in two_branch:
            kind, cfg = self.baselines[m]
            k2, objs, m2_time = self._objects(split, scene, cfg, s)
            log = dict(prep_times=dict(m2=m2_time), m2_key=k2)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            img = run_collage(kind, eng, vis, scene, objs, s, cfg, log)
            self._write(split, m, scene, s, img, log, cfg)
            stats["done"] += 1
