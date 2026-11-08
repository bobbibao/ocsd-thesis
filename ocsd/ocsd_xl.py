"""OCSD-XL: the OCSD pipeline (M1-M5, as in the thesis and slides) on Stable Diffusion XL at 1024 px.

Nothing in the method is replaced by an external layout generator. The modules are the ones in `ocsd.method`:
  M1  sketch parsing (per-object sketch, mask, box, relations, grouped prompt)          ocsd.sketch
  M2  object branch: K candidates per object with ControlNet Scribble, Grounding DINO + SAM mask, selection
  M3  identity tokens + LoRA (optional arm; off in the centre configuration, as in OCSD-v2)
  M4  scene construction with latent blending while t > alpha*T
  M5  object-aware conditioning: (a) region-constrained cross-attention, (b) attention-energy guidance,
      (c) scene-level ControlNet, (d) post-generation check with object-level repair
What changes against the SD 1.5 runs is the backbone (SDXL + a scribble ControlNet trained for hand-drawn lines),
the working resolution (1024 px) and a scale-free energy step (unit-RMS gradient). The sampler is the OCSD-v2 "joint"
trajectory (`blend_mode="joint"`); the thesis sampler ("separate") is kept as an arm.

Scenes are upscaled from the 512 px benchmark frame to 1024 px for generation; images are scored at 512 px with the
same evaluator as every earlier run.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import OCSDConfig
from .methods import V2
from .sketch import Scene, SceneObject

SIZE = 1024

# Settings shared by every SDXL arm (baselines included), fixed before any SDXL output existed.
XL_COMMON = dict(height=SIZE, width=SIZE, steps=30, guidance=6.0, K=2, obj_steps=20, obj_cn_scale=1.0, cn_scale=1.0,
                 S1=0, S2=0, R=2, bo_n=3)
# Centre configuration of OCSD-XL = OCSD-v2 (methods.V2) + a stronger scene ControlNet (the SDXL scribble model keeps
# shape at 0.8 without the artefacts the SD 1.5 one shows) + unit-RMS energy steps (eta is then a latent step size and
# does not depend on the backbone's gradient scale).
XL_OCSD = dict(V2, omega=0.8, energy_grad_norm=True, eta=0.05, tau=10, lambda0=8.0, alpha=0.4, anchor_shrink=0.3)

# name -> (kind, overrides on XL_COMMON [+ XL_OCSD for kind "ocsd"])
ARMS: Dict[str, Tuple[str, dict]] = {
    # baselines on the same backbone and ControlNet
    "xl_controlnet": ("controlnet", {}),
    "xl_collage": ("collage", dict(collage_strength=0.3, m2_per_seed=True)),
    # OCSD-XL and its one-factor variants
    "ocsd_xl": ("ocsd", {}),
    "ocsd_xl_a02": ("ocsd", dict(alpha=0.2)),
    "ocsd_xl_a06": ("ocsd", dict(alpha=0.6)),
    "ocsd_xl_w04": ("ocsd", dict(omega=0.4)),
    "ocsd_xl_noenergy": ("ocsd", dict(use_energy=False)),
    "ocsd_xl_norepair": ("ocsd", dict(use_verify=False, repair=False)),
    "ocsd_xl_region": ("ocsd", dict(alpha=1.0)),              # no M2 anchoring: single-branch generation + M5
    "ocsd_xl_sep": ("ocsd", dict(blend_mode="separate", alpha=0.1, anchor_shrink=0.0, repair=False, R=1,
                                 omega=0.4)),                 # the thesis (v1) sampler on SDXL, without M3
    "ocsd_xl_id": ("ocsd", dict(use_identity=True, S1=100, S2=100, lora_scale=0.5)),   # + M3 identity learning
}


def arm_config(name: str) -> Tuple[str, OCSDConfig]:
    kind, over = ARMS[name]
    base = dict(XL_COMMON, **(XL_OCSD if kind == "ocsd" else {}))
    return kind, OCSDConfig().replace(**dict(base, **over))


# ---------------------------------------------------------------------------------------------- scene scaling
def upscale_scene(scene: Scene, size: int = SIZE) -> Scene:
    """The same scene in a `size` px frame (strokes stay black on white and become proportionally thicker)."""
    h = scene.sketch.shape[0]
    if h == size:
        return scene
    f = size / h

    def sketch(a):
        up = cv2.resize(a, (size, size), interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
        return np.where(up < (160 if f > 1 else 235), 0, 255).astype(np.uint8)

    def mask(m):
        return cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0

    objs = [SceneObject(cls=o.cls, phrase=o.phrase, box=tuple(float(v) * f for v in o.box), mask=mask(o.mask),
                        sketch=sketch(o.sketch)) for o in scene.objects]
    return replace(scene, sketch=sketch(scene.sketch), objects=objs)


def to_eval_size(img: np.ndarray, size: int = 512) -> np.ndarray:
    return img if img.shape[0] == size else cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------------------------- generation
def make_objects(eng, vis, scene_xl: Scene, seed: int, log: Optional[dict] = None):
    """M2 for one (scene, seed); every arm of that pair shares these objects."""
    from .method import object_branch
    _, cfg = arm_config("ocsd_xl")
    t0 = time.time()
    objs = object_branch(eng, vis, scene_xl, cfg, seed, log)
    if log is not None:
        log["m2_time"] = time.time() - t0
        log["m2_scores"] = [float(o.score) for o in objs]
    return objs


def run_arm(name: str, eng, vis, scene_xl: Scene, seed: int, objs, log: Optional[dict] = None) -> np.ndarray:
    """One image (SIZE px) of arm `name`. `objs`: the M2 objects of this (scene, seed)."""
    from .method import generate, prepare
    from .methods import run_baseline, run_collage
    log = log if log is not None else {}
    kind, cfg = arm_config(name)
    before = dict(eng.compute)
    try:
        if kind == "controlnet":
            img = run_baseline("controlnet", eng, vis, scene_xl, seed, cfg, log=log)
        elif kind == "collage":
            img = run_collage("collage", eng, vis, scene_xl, objs, seed, cfg, log)
        elif kind == "ocsd":
            prep = prepare(eng, vis, scene_xl, cfg, objs=objs, seed=seed, log=log)
            log["prep_times"] = dict(prep.times)
            img = generate(eng, vis, scene_xl, prep, cfg, seed, log=log)
        else:
            raise KeyError(kind)
    finally:
        # never leak identity tokens, LoRA, region masks or stored maps into the next image
        eng.reset_identity()
        eng.ctrl.clear_regions()
        eng.ctrl.reset_maps()
        eng.ctrl.store = False
    log["compute"] = eng.compute_delta(before)
    log["cfg"] = cfg.to_dict()
    return img


def compose_preview(scene_xl: Scene, objs) -> np.ndarray:
    """The M2 objects pasted at their sketch positions on a white canvas (what M4 anchors to)."""
    from .method import compose_foreground
    return compose_foreground(scene_xl, objs, arm_config("ocsd_xl")[1])[0]
