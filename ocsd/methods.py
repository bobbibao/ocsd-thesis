"""Danh mục phương pháp: baseline, cài đặt lại Zhang et al. (2025), OCSD và các biến thể cắt bỏ.

Mọi phương pháp dùng chung backbone, scheduler DDIM, số bước, CFG, câu lệnh âm và tập seed (so sánh công bằng).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import BACKBONES, LEGACY_DEFAULTS, OCSDConfig
from .engine import Engine
from .matching import consistency, verify_pass, verify_score
from .method import _lat_mask, sample
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

OCSD_VARIANTS: Dict[str, OCSDConfig] = {k: BASE.replace(**o) for k, o in VARIANT_OVERRIDES.items()}

BASELINES = ["controlnet", "t2i_adapter", "gligen", "cn_region", "cn_energy", "controlnet_bo3"]

LABELS = {
    "controlnet": "SD + ControlNet (mô hình nền)",
    "t2i_adapter": "SD + T2I-Adapter",
    "gligen": "GLIGEN (hộp + văn bản)",
    "cn_region": "ControlNet + chú ý vùng (kiểu DenseDiffusion)",
    "cn_energy": "ControlNet + năng lượng chú ý (kiểu BoxDiff/A&E)",
    "controlnet_bo3": "ControlNet + chọn tốt nhất trong 3 (cùng bộ kiểm tra)",
    "zhang2025": "Zhang et al. (2025) - cài đặt lại",
    "ocsd_lite": "OCSD-lite (không học định danh)",
    "ocsd": "OCSD (đề xuất)",
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
    "real": "Ảnh thật (trần bộ phát hiện)",
}
LABELS.update({f"alpha_{a:.1f}": f"α = {a:.1f}" for a in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)})


def m2_key(c: OCSDConfig) -> str:
    return f"K{c.K}_{'phrase' if c.use_user_phrase else 'cls'}"


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


def run_baseline(name: str, eng: Engine, vis, scene: Scene, seed: int, cfg: OCSDConfig = BASE,
                 backbone: str = "sd15", cache_dir=None, log: Optional[dict] = None) -> np.ndarray:
    log = log if log is not None else {}
    t0 = time.time()
    H = cfg.height
    lh = H // eng.vae_factor
    ar = lh // eng.attn_div
    if name in ("controlnet", "controlnet_bo3"):
        control = eng.control_tensor(sketch_to_control(scene.sketch))
        if name == "controlnet":
            img = sample(eng, [scene.caption], [seed], cfg, control, 1.0)[0]
        else:
            best, best_s, tries = None, -1e9, []
            for r in range(cfg.R + 1):
                im = sample(eng, [scene.caption], [seed + 100003 * r], cfg, control, 1.0)[0]
                c = consistency(scene, vis.gdino(im, [o.cls for o in scene.objects], thr=cfg.verify_thr))
                s = verify_score(c)
                tries.append(s)
                if s > best_s:
                    best, best_s = im, s
                if verify_pass(c):
                    break
            img = best
            log["tries"] = len(tries)
    elif name == "t2i_adapter":
        st = eng.adapter_states(scene.sketch, 1.0)
        img = sample(eng, [scene.caption], [seed], cfg, adapter_states=st)[0]
    elif name == "gligen":
        pipe = _gligen_pipe(eng, backbone, cache_dir)
        boxes = [[o.box[0] / H, o.box[1] / H, o.box[2] / H, o.box[3] / H] for o in scene.objects]
        g = torch.Generator("cpu").manual_seed(seed)
        img = np.array(pipe(prompt=scene.caption, gligen_phrases=[o.phrase for o in scene.objects],
                            gligen_boxes=boxes, gligen_scheduled_sampling_beta=0.3,
                            num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
                            negative_prompt=cfg.negative_prompt, generator=g, height=H, width=H).images[0])
    elif name in ("cn_region", "cn_energy"):
        control = eng.control_tensor(sketch_to_control(scene.sketch))
        gp = scene_global_prompt(eng.tokenizer, scene, None, use_phrase=True, caption=cfg.use_caption)
        masks = [o.mask for o in scene.objects]
        region = energy = None
        if name == "cn_region":
            om = torch.stack([_lat_mask(m, lh, lh, eng.device) for m in masks])
            region = dict(masks=om, tokens=[gp.groups.get(f"obj{i}", []) for i in range(scene.n)],
                          bg_tokens=gp.groups.get("bg", []), lambda0=cfg.lambda0, gamma=cfg.gamma)
        else:
            mlr = torch.stack([_lat_mask(m, ar, ar, eng.device, thr=0.3) for m in masks])
            energy = dict(groups=[gp.groups.get(f"id{i}", []) for i in range(scene.n)], masks_lr=mlr,
                          eot=eng.eot_index(gp.text), beta=cfg.beta, eta=cfg.eta, tau=cfg.tau)
        img = sample(eng, [gp.text], [seed], cfg, control, 1.0, region=region, energy=energy)[0]
        log["prompt"] = gp.text
    else:
        raise KeyError(name)
    log["scene_time"] = time.time() - t0
    log["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    return img
