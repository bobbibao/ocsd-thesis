"""Same sketch, different scene background ("đổi bối cảnh"), with OCSD on SD 1.5 and all five modules M1-M5 (M3 on).

Uses the code and settings of ocsd.trace_ocsd (frozen paper_v2 tuning, tier heldout_final) and the same order as
ocsd.runner. M1 parsing, M2 objects and M3 identity tokens do not depend on the background (the object prompt has no
background words), so they are made ONCE; then M4 + M5 run once per background with the same seed, only the background
words of the prompt (scene.bg and the caption) changed. The first background is the scene's own, so its image can be
compared with the traced image of the same scene and seed.
Output: <drive>/trace_ocsd/bgswap_<sid>_s<seed>/ : sketch.png, objects_m2.png, <i>_<bg>.png, SHEET.jpg, bgswap.json, ZIP.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

from .trace_ocsd import DEFAULT_SID, PROTOCOL, _w, ocsd_config

DEFAULT_BGS = ("on a city street", "on a snowy field", "on a sandy beach")


def _slug(s: str) -> str:
    return "_".join(s.split())


def run(drive_root: str, sid: str = DEFAULT_SID, seed: int = 0, bgs=DEFAULT_BGS, eng=None, vis=None,
        out_root: str | None = None, log=print, scene_dir: str | None = None, overrides: dict | None = None,
        size: int | None = None) -> dict:
    from . import method, xl_study
    from .data import load_scene

    if scene_dir is None:
        p = xl_study.load_protocol(PROTOCOL, drive_root)
        if sid not in p["scenes"]:
            raise ValueError(f"{sid} is not one of the 72 paper scenes")
        scene_dir = [d for d in xl_study.scene_dirs(p) if Path(d).name == sid][0]
    scene = load_scene(scene_dir)
    if size:
        from .ocsd_xl import upscale_scene
        scene = upscale_scene(scene, size)
    cfg = ocsd_config(overrides)
    out = Path(out_root or Path(drive_root) / "trace_ocsd") / f"bgswap_{sid}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if eng is None:
        from .engine import Engine
        eng = Engine.from_pretrained("sd15", "cuda", True, cache_dir=os.environ.get("HF_HOME"), load_adapter=False)
    if vis is None:
        from .vision import Vision
        vis = Vision(str(eng.device), cache_dir=os.environ.get("HF_HOME"))
    if hasattr(vis, "set_scene"):
        vis.set_scene(scene)
    bg0, cap0 = scene.bg, scene.caption
    if bg0 not in cap0:
        raise ValueError(f"background {bg0!r} not found in caption {cap0!r}")
    all_bgs = [bg0] + [b for b in bgs if b != bg0]
    log(f"[bgswap] {sid} seed {seed}: OCSD (SD 1.5, M1-M5), backgrounds {all_bgs}")
    meta = dict(sid=sid, seed=seed, method="OCSD (SD 1.5, M1-M5)", cfg=cfg.to_dict(), runs=[])
    t0 = time.time()
    try:
        objs = method.object_branch(eng, vis, scene, cfg, seed=seed if cfg.m2_per_seed else 0)   # M2 (once)
        fg = method.compose_foreground(scene, objs, cfg)
        id_tokens = method.learn_identity(eng, scene, objs, cfg, seed=0, log={}) if cfg.use_identity else None  # M3
        prep = method.Prepared(objs, fg, id_tokens, {})
        for i, bg in enumerate(all_bgs):                                                              # M4 + M5
            scene.bg, scene.caption = bg, cap0.replace(bg0, bg)
            rl = {}
            t1 = time.time()
            img = method.generate(eng, vis, scene, prep, cfg, seed, log=rl)
            f = _w(out / f"{i}_{_slug(bg)}.png", img, None)
            meta["runs"].append(dict(bg=bg, caption=scene.caption, global_prompt=rl.get("global_prompt"), file=f,
                                     seconds=round(time.time() - t1, 1),
                                     log={k: v for k, v in rl.items() if isinstance(v, (int, float, str, bool))}))
            log(f"[bgswap] {i + 1}/{len(all_bgs)} {bg}: {time.time() - t1:.0f} s")
    finally:
        scene.bg, scene.caption = bg0, cap0
        eng.reset_identity()
        eng.ctrl.clear_regions()
        eng.ctrl.reset_maps()
        eng.ctrl.store = False
    meta["seconds"] = round(time.time() - t0, 1)
    _w(out / "sketch.png", np.repeat(scene.sketch[..., None], 3, 2), None)
    _w(out / "objects_m2.png", fg[0] if isinstance(fg, (tuple, list)) else fg, None)
    ims = [cv2.cvtColor(cv2.imread(r["file"]), cv2.COLOR_BGR2RGB) for r in meta["runs"]]
    h = ims[0].shape[0]
    sk = cv2.resize(np.repeat(scene.sketch[..., None], 3, 2).astype(np.uint8), (h, h))
    sheet = np.concatenate([sk] + [cv2.resize(x, (h, h)) for x in ims], 1)
    _w(out / "SHEET.jpg", sheet, None)
    (out / "bgswap.json").write_text(json.dumps(meta, indent=1, default=str))
    z = out.parent / f"bgswap_{sid}_s{seed}.zip"
    with ZipFile(z, "w", ZIP_DEFLATED) as zf:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(out.parent))
    meta["zip"] = str(z)
    log(f"[bgswap] done in {meta['seconds']:.0f} s -> {z}")
    return meta
