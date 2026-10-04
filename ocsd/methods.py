"""Danh mục phương pháp: baseline, cài đặt lại Zhang et al. (2025), OCSD và các biến thể cắt bỏ.

Mọi phương pháp dùng chung backbone, scheduler DDIM, số bước, CFG, câu lệnh âm và tập seed (so sánh công bằng).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .config import BACKBONES, LEGACY_DEFAULTS, PARETO, OCSDConfig
from .engine import Engine
from .matching import consistency, verify_pass, verify_score
from .method import _lat_mask, attn_masks, compose_foreground, sample
from .sketch import Scene, build_grouped_prompt, join_phrases, scene_global_prompt, sketch_to_control

BASE = OCSDConfig()

# ----------------------------------------------------------------------------- OCSD family (cấu hình)
M5_OFF = dict(use_region_attn=False, use_energy=False, use_scene_cn=False, use_verify=False)

# Each variant = BASE + its own overrides. stages.tier_variants applies the tier settings and the tuned values only to
# the fields a variant does not set itself, so an ablation or a sweep keeps what defines it.
VARIANT_OVERRIDES: Dict[str, dict] = {
    # phương pháp đề xuất
    "ocsd": {},
    "ocsd_lite": dict(use_identity=False),                          # không học định danh: không cần huấn luyện
    # Zhang et al. 2025 (cài đặt lại): M2 1 ứng viên, câu lệnh chỉ tên lớp, M3 không L_att, M4 alpha=0.5, không M5;
    # pinned to the pre-freeze defaults, so it is the same re-implementation as in the paper-tier run
    "zhang2025": dict(LEGACY_DEFAULTS, K=1, use_user_phrase=False, use_att_sep=False, **M5_OFF),
    # ---- nghiên cứu cắt bỏ (E4)
    "abl_no_blend": dict(use_blend=False),
    "abl_alpha0": dict(alpha=0.0),
    "abl_no_region": dict(use_region_attn=False),
    "abl_no_energy": dict(use_energy=False),
    "abl_m5ab_both": dict(use_region_attn=True, use_energy=True),   # both M5(a) and M5(b), whatever tuning kept
    "abl_no_scenecn": dict(use_scene_cn=False),
    "abl_no_verify": dict(use_verify=False),
    "abl_no_m5": dict(M5_OFF),
    "abl_no_attsep": dict(use_att_sep=False),
    "abl_k1": dict(K=1),
    "abl_m3_long": dict(S1=200, S2=200),                             # M3 at the full 200 + 200 steps (plan 2.7)
    "abl_bg_only": dict(prompt_mode="bg_only"),
    "abl_global_only": dict(prompt_mode="global_only"),
}
# ---- khảo sát alpha (không kiểm tra hậu sinh để thấy rõ tác động của riêng alpha)
for _a in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
    VARIANT_OVERRIDES[f"alpha_{_a:.1f}"] = dict(alpha=_a, use_verify=False)

# ---- OCSD-v2: one composite-aware trajectory (blend_mode "joint"), M5 from the first step, mean energy, the
# small-object mask fix, caption class words limited to their masks, object-level repair in M5(d), M2 objects drawn
# per seed and no per-scene training (M3 off). Tuned on the tuning split by stages.tune (phase "v2").
V2 = dict(blend_mode="joint", use_identity=False, alpha=0.4, anchor_shrink=0.3, m2_per_seed=True, repair=True,
          region_min_cell=True, caption_class_masks=True, energy_reduce="mean", energy_tokens="phrase", eta=20.0,
          use_region_attn=True, use_energy=True)
V2_ABLATIONS = {                                       # E4v2, QuickDraw
    "v2abl_separate": dict(blend_mode="separate"),     # the thesis sampler with the v2 settings
    "v2abl_no_repair": dict(repair=False),             # thesis M5(d): regenerate the whole image
    "v2abl_no_verify": dict(use_verify=False),
    "v2abl_no_region": dict(use_region_attn=False),
    "v2abl_no_energy": dict(use_energy=False),
    "v2abl_no_m5": dict(M5_OFF, repair=False),
    "v2abl_no_anchor": dict(alpha=1.0),                # M2 objects never re-imposed: single-branch generation + M5
    "v2abl_no_shrink": dict(anchor_shrink=0.0),
    "v2abl_no_mincell": dict(region_min_cell=False),
    "v2abl_energy_sum": dict(energy_reduce="sum"),
}
V2_COCO_ABLATIONS = {                                  # E4v2_coco: the caption only changes COCO prompts
    "v2abl_no_caption": dict(use_caption=False),
    "v2abl_no_capmask": dict(caption_class_masks=False),
}
VARIANT_OWN: Dict[str, dict] = {}       # fields tuning must not touch (default: all of VARIANT_OVERRIDES[name])
VARIANT_FAMILY: Dict[str, str] = {}     # "v1" (default) or "v2": which tuned values apply
for _k, _o in {"ocsd_v2": {}, **V2_ABLATIONS, **V2_COCO_ABLATIONS}.items():
    VARIANT_OVERRIDES[_k], VARIANT_OWN[_k], VARIANT_FAMILY[_k] = dict(V2, **_o), dict(_o), "v2"
# ---- control-vs-quality sweeps (job "pareto"): no M5(d), so only alpha changes along a curve
for _a in PARETO["pv2"]:
    _o = dict(alpha=_a, use_verify=False, repair=False)
    VARIANT_OVERRIDES[f"pv2_a{_a:.1f}"], VARIANT_OWN[f"pv2_a{_a:.1f}"] = dict(V2, **_o), _o
    VARIANT_FAMILY[f"pv2_a{_a:.1f}"] = "v2"
for _a in PARETO["pv1lite"]:
    VARIANT_OVERRIDES[f"pv1lite_a{_a:.1f}"] = dict(use_identity=False, alpha=_a, use_verify=False)

OCSD_VARIANTS: Dict[str, OCSDConfig] = {k: BASE.replace(**o) for k, o in VARIANT_OVERRIDES.items()}

# ----------------------------------------------------------------------------- baselines (catalogue)
SINGLE_BASELINES = ["controlnet", "t2i_adapter", "gligen", "cn_region", "cn_energy", "controlnet_bo3",
                    "controlnet_bon", "gligen_bon"]
COLLAGE_BASELINES = ["collage", "collage_bo3"]           # use the M2 objects of OCSD
BASELINES = SINGLE_BASELINES + COLLAGE_BASELINES
# tuned values of a baseline (stages.tune, phase "baselines") also apply to its best-of variants
BASELINE_TUNED_FROM = {"controlnet_bo3": "controlnet", "controlnet_bon": "controlnet", "gligen_bon": "gligen",
                       "collage_bo3": "collage"}
# control-vs-quality sweeps of the baselines: name -> (baseline, field, value)
PARETO_BASELINES = {f"pcn_s{v:.1f}": ("controlnet", "cn_scale", v) for v in PARETO["pcn"]}
PARETO_BASELINES.update({f"pgl_b{v:.1f}": ("gligen", "gligen_beta", v) for v in PARETO["pgl"]})
# the settings each baseline's images depend on (stages._gen_signatures; use_caption only matters on COCO)
BASELINE_FIELDS = {
    "controlnet": ("cn_scale",),
    "controlnet_bo3": ("cn_scale", "R", "verify_thr"),
    "controlnet_bon": ("cn_scale", "bo_n", "verify_thr"),
    "t2i_adapter": ("adapter_scale",),
    "gligen": ("gligen_beta",),
    "gligen_bon": ("gligen_beta", "bo_n", "verify_thr"),
    "cn_region": ("cn_scale", "use_caption", "lambda0", "gamma", "region_min_cell", "caption_class_masks"),
    "cn_energy": ("cn_scale", "use_caption", "beta", "eta", "tau", "region_min_cell", "energy_reduce",
                  "energy_grad_norm"),
    "collage": ("cn_scale", "collage_strength", "K", "obj_steps", "obj_cn_scale", "use_user_phrase", "m2_per_seed"),
    "collage_bo3": ("cn_scale", "collage_strength", "K", "obj_steps", "obj_cn_scale", "use_user_phrase",
                    "m2_per_seed", "R", "verify_thr"),
}

LABELS = {
    "controlnet": "SD + ControlNet (mô hình nền)",
    "t2i_adapter": "SD + T2I-Adapter",
    "gligen": "GLIGEN (hộp + văn bản)",
    "cn_region": "ControlNet + chú ý vùng (kiểu DenseDiffusion)",
    "cn_energy": "ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)",
    "controlnet_bo3": "ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra)",
    "controlnet_bon": "ControlNet, best of N (same check, compute-matched)",
    "gligen_bon": "GLIGEN, best of N (same check, compute-matched)",
    "collage": "Collage: M2 objects pasted + SDEdit",
    "collage_bo3": "Collage, best of 3 (same check)",
    "zhang2025": "Zhang et al. (2025) - cài đặt lại",
    "ocsd_lite": "OCSD-lite (không học định danh)",
    "ocsd": "OCSD (đề xuất)",
    "ocsd_v2": "OCSD-v2 (joint trajectory, training-free)",
    "abl_no_blend": "- bỏ suy luận trộn tiềm ẩn (alpha = 1)",
    "abl_alpha0": "- trộn tiềm ẩn toàn bộ quá trình (alpha = 0)",
    "abl_no_region": "- bỏ chú ý giới hạn theo vùng M5(a)",
    "abl_no_energy": "- bỏ dẫn hướng năng lượng M5(b)",
    "abl_m5ab_both": "+ both M5(a) and M5(b)",
    "abl_m3_long": "+ M3 with the full 200 + 200 steps",
    "abl_no_scenecn": "- bỏ ControlNet cấp cảnh M5(c)",
    "abl_no_verify": "- bỏ kiểm tra hậu sinh M5(d)",
    "abl_no_m5": "- bỏ toàn bộ M5",
    "abl_no_attsep": "- bỏ L_att ở M3",
    "abl_k1": "- M2 chỉ 1 ứng viên (không chọn)",
    "abl_bg_only": "- chỉ câu lệnh nền",
    "abl_global_only": "- chỉ câu lệnh toàn cục",
    "v2abl_separate": "v2 - separate background branch (thesis sampler)",
    "v2abl_no_repair": "v2 - whole-image regeneration instead of repair",
    "v2abl_no_verify": "v2 - no check / repair (M5(d))",
    "v2abl_no_region": "v2 - no region attention M5(a)",
    "v2abl_no_energy": "v2 - no energy guidance M5(b)",
    "v2abl_no_m5": "v2 - no M5",
    "v2abl_no_anchor": "v2 - objects never re-imposed (alpha = 1)",
    "v2abl_no_shrink": "v2 - anchor masks not shrunk",
    "v2abl_no_mincell": "v2 - without the small-object mask fix",
    "v2abl_energy_sum": "v2 - energy summed over objects",
    "v2abl_no_caption": "v2 - caption not in the prompt",
    "v2abl_no_capmask": "v2 - caption class words not limited to their masks",
    "real": "Ảnh thật (trần bộ phát hiện)",
}
LABELS.update({f"alpha_{a:.1f}": f"α = {a:.1f}" for a in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)})
LABELS.update({f"pv2_a{a:.1f}": f"OCSD-v2, α = {a:.1f}" for a in PARETO["pv2"]})
LABELS.update({f"pv1lite_a{a:.1f}": f"OCSD-lite (thesis sampler), α = {a:.1f}" for a in PARETO["pv1lite"]})
LABELS.update({k: f"{'ControlNet, scale' if b == 'controlnet' else 'GLIGEN, β'} = {v:.1f}"
               for k, (b, _, v) in PARETO_BASELINES.items()})


def m2_key(c: OCSDConfig) -> str:
    return f"K{c.K}_{'phrase' if c.use_user_phrase else 'cls'}"


def m2_dir_key(c: OCSDConfig, seed: int) -> str:
    """Cache folder of the M2 objects: one set per scene (drawn with seed 0), or one per seed with m2_per_seed
    (seed 0 shares the per-scene set, which was drawn with seed 0)."""
    return m2_key(c) if (not c.m2_per_seed or seed == 0) else f"{m2_key(c)}_s{seed}"


def m3_key(c: OCSDConfig) -> str:
    if not c.use_identity:
        return m2_key(c) + "_noid"
    return m2_key(c) + f"_id{c.S1}-{c.S2}_r{c.lora_rank}_{'att' if (c.use_att_sep and c.lambda_att > 0) else 'noatt'}"


# ----------------------------------------------------------------------------- baselines
_GLIGEN = {}


def _gligen_pipe(eng: Engine, backbone: str, cache_dir=None):
    if "pipe" not in _GLIGEN:
        from diffusers import DDIMScheduler, StableDiffusionGLIGENPipeline
        pid = BACKBONES[backbone]["gligen"]
        if not pid:
            raise RuntimeError("GLIGEN không có cho backbone này")
        p = StableDiffusionGLIGENPipeline.from_pretrained(pid, torch_dtype=eng.dtype, cache_dir=cache_dir,
                                                          safety_checker=None, requires_safety_checker=False)
        p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
        p.set_progress_bar_config(disable=True)
        _GLIGEN["pipe"] = p.to(eng.device)
    return _GLIGEN["pipe"]


def best_of(scene: Scene, vis, cfg: OCSDConfig, n: int, make: Callable[[int], np.ndarray], log: dict) -> np.ndarray:
    """Up to n samples (make(r)), each checked by Grounding DINO like M5(d); stops at the first one that passes and
    returns the best by verify_score."""
    best, best_s, tries = None, -1e9, []
    for r in range(n):
        im = make(r)
        c = consistency(scene, vis.gdino(im, [o.cls for o in scene.objects], thr=cfg.verify_thr))
        s = verify_score(c)
        tries.append(s)
        if s > best_s:
            best, best_s = im, s
        if verify_pass(c):
            break
    log["tries"] = len(tries)
    return best


def _gligen(eng: Engine, scene: Scene, seed: int, cfg: OCSDConfig, backbone: str, cache_dir) -> np.ndarray:
    H = cfg.height
    pipe = _gligen_pipe(eng, backbone, cache_dir)
    boxes = [[o.box[0] / H, o.box[1] / H, o.box[2] / H, o.box[3] / H] for o in scene.objects]
    g = torch.Generator("cpu").manual_seed(seed)
    return np.array(pipe(prompt=scene.caption, gligen_phrases=[o.phrase for o in scene.objects],
                         gligen_boxes=boxes, gligen_scheduled_sampling_beta=cfg.gligen_beta,
                         num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
                         negative_prompt=cfg.negative_prompt, generator=g, height=H, width=H).images[0])


def run_baseline(name: str, eng: Engine, vis, scene: Scene, seed: int, cfg: OCSDConfig = BASE,
                 backbone: str = "sd15", cache_dir=None, log: Optional[dict] = None) -> np.ndarray:
    log = log if log is not None else {}
    t0 = time.time()
    H = cfg.height
    lh = H // eng.vae_factor
    ar = lh // eng.attn_div
    if name in ("controlnet", "controlnet_bo3", "controlnet_bon"):
        control = eng.control_tensor(sketch_to_control(scene.sketch))
        make = lambda r: sample(eng, [scene.caption], [seed + 100003 * r], cfg, control, cfg.cn_scale)[0]
        if name == "controlnet":
            img = make(0)
        else:
            img = best_of(scene, vis, cfg, cfg.R + 1 if name == "controlnet_bo3" else cfg.bo_n, make, log)
    elif name == "t2i_adapter":
        st = eng.adapter_states(scene.sketch, cfg.adapter_scale)
        img = sample(eng, [scene.caption], [seed], cfg, adapter_states=st)[0]
    elif name == "gligen":
        img = _gligen(eng, scene, seed, cfg, backbone, cache_dir)
    elif name == "gligen_bon":
        img = best_of(scene, vis, cfg, cfg.bo_n,
                      lambda r: _gligen(eng, scene, seed + 100003 * r, cfg, backbone, cache_dir), log)
    elif name in ("cn_region", "cn_energy"):
        control = eng.control_tensor(sketch_to_control(scene.sketch))
        gp = scene_global_prompt(eng.tokenizer, scene, None, use_phrase=True, caption=cfg.use_caption,
                                 class_groups=cfg.caption_class_masks and name == "cn_region")
        masks = [o.mask for o in scene.objects]
        region = energy = None
        if name == "cn_region":
            om = [_lat_mask(m, lh, lh, eng.device) for m in masks]
            tokens = [gp.groups.get(f"obj{i}", []) for i in range(scene.n)]
            for c in sorted({o.cls for o in scene.objects}):
                if gp.groups.get(f"cls:{c}"):
                    om.append(_lat_mask(np.any([m for m, o in zip(masks, scene.objects) if o.cls == c], 0),
                                        lh, lh, eng.device))
                    tokens.append(gp.groups[f"cls:{c}"])
            region = dict(masks=torch.stack(om), tokens=tokens, bg_tokens=gp.groups.get("bg", []),
                          lambda0=cfg.lambda0, gamma=cfg.gamma, min_cell=cfg.region_min_cell)
        else:
            energy = dict(groups=[gp.groups.get(f"id{i}", []) for i in range(scene.n)],
                          masks_lr=attn_masks(masks, ar, eng.device, cfg.region_min_cell),
                          eot=eng.eot_index(gp.text), beta=cfg.beta, eta=cfg.eta, tau=cfg.tau,
                          reduce=cfg.energy_reduce, grad_norm=cfg.energy_grad_norm)
        img = sample(eng, [gp.text], [seed], cfg, control, cfg.cn_scale, region=region, energy=energy)[0]
        log["prompt"] = gp.text
    else:
        raise KeyError(name)
    log["scene_time"] = time.time() - t0
    log["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    return img


@torch.no_grad()
def sdedit(eng: Engine, img: np.ndarray, prompt: str, seed: int, cfg: OCSDConfig, control=None,
           cn_scale: float = 0.0, strength: float = 0.3) -> np.ndarray:
    """Re-noise `img` to the last `strength` fraction of the trajectory and denoise it with `prompt` (SDEdit)."""
    ts = eng.set_timesteps(cfg.steps)
    start = min(max(int(round((1 - strength) * len(ts))), 0), len(ts) - 1)
    z = eng.q_sample(eng.to_latent(img), eng.randn(seed, 1, cfg.height, cfg.width), ts[start])
    emb_c, emb_u = eng.encode([prompt]), eng.encode([cfg.negative_prompt])
    for t in ts[start:]:
        z = eng.step(eng.cfg_eps(z, t, emb_c, emb_u, cfg.guidance, control, cn_scale), t, z)
    return eng.to_image(z)[0]


def run_collage(name: str, eng: Engine, vis, scene: Scene, objs, seed: int, cfg: OCSDConfig,
                log: Optional[dict] = None) -> np.ndarray:
    """Naive two-branch baseline: the M2 objects are pasted (same placement as OCSD) onto a ControlNet image of the
    scene, then harmonised by SDEdit (strength cfg.collage_strength, caption + scene ControlNet). collage_bo3 adds the
    same Grounding DINO check as M5(d) over 3 seeds. No identity learning, no M5(a)/(b)."""
    log = log if log is not None else {}
    t0 = time.time()
    canvas, _, union = compose_foreground(scene, objs, cfg)
    control = eng.control_tensor(sketch_to_control(scene.sketch))

    def make(r):
        s = seed + 100003 * r
        bg = sample(eng, [scene.caption], [s], cfg, control, cfg.cn_scale)[0]
        comp = np.where(union[..., None], canvas, bg)
        return sdedit(eng, comp, scene.caption, s + 1, cfg, control, cfg.cn_scale, cfg.collage_strength)

    img = best_of(scene, vis, cfg, cfg.R + 1, make, log) if name == "collage_bo3" else make(0)
    log["scene_time"] = time.time() - t0
    log["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    return img
