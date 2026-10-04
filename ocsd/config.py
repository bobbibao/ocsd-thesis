"""Cấu hình chung cho toàn bộ dự án OCSD (Object-Consistent Sketch-guided Diffusion).

Mọi đường dẫn đều nằm dưới DRIVE_ROOT (Google Drive) để kết quả không mất khi Colab ngắt kết nối.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


# ----------------------------------------------------------------------------- paths
@dataclass
class Paths:
    root: str = "/content/drive/MyDrive/KLTN_OCSD"

    @property
    def data(self):
        return os.path.join(self.root, "data")

    @property
    def quickdraw_raw(self):
        return os.path.join(self.data, "quickdraw_raw")

    @property
    def coco_raw(self):
        return os.path.join(self.data, "coco_raw")

    @property
    def benchmarks(self):
        return os.path.join(self.data, "benchmarks")

    @property
    def outputs(self):
        return os.path.join(self.root, "outputs")

    @property
    def results(self):
        return os.path.join(self.root, "results")

    @property
    def cache(self):
        # HuggingFace cache on local disk (Drive breaks the cache's symlinks), see ocsd/hfcache.py
        return os.environ.get("HF_HOME", "/content/hf_cache")

    def makedirs(self):
        for p in [self.data, self.quickdraw_raw, self.coco_raw, self.benchmarks,
                  self.outputs, self.results, self.cache]:
            os.makedirs(p, exist_ok=True)


# ----------------------------------------------------------------------------- models
# Backbone mặc định: SD 1.5. Lý do: ControlNet Scribble, T2I-Adapter Sketch và GLIGEN đều có bản
# chính thức cho SD 1.x, nên mọi baseline dùng CHUNG một backbone -> so sánh công bằng.
BACKBONES: Dict[str, Dict[str, str]] = {
    "sd15": dict(
        sd="stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet="lllyasviel/control_v11p_sd15_scribble",
        t2i_adapter="TencentARC/t2iadapter_sketch_sd15v2",
        gligen="masterful/gligen-1-4-generation-text-box",
        resolution="512",
    ),
    "sd21": dict(
        sd="stabilityai/stable-diffusion-2-1-base",
        controlnet="thibaud/controlnet-sd21-scribble-diffusers",
        t2i_adapter="",  # không có T2I-Adapter sketch chính thức cho SD 2.1
        gligen="",
        resolution="512",
    ),
}

AUX_MODELS = dict(
    gdino="IDEA-Research/grounding-dino-base",      # dùng TRONG phương pháp (M2 chọn ứng viên, M5(d) kiểm tra)
    owlv2="google/owlv2-base-patch16-ensemble",       # dùng để ĐÁNH GIÁ (khác bộ phát hiện của phương pháp)
    sam="facebook/sam-vit-huge",
    clip="openai/clip-vit-large-patch14",
    dinov2="facebook/dinov2-base",
    pidinet="lllyasviel/Annotators",
    detr="facebook/detr-resnet-50",                   # closed-set COCO detector, robustness of the evaluation only
)


# ----------------------------------------------------------------------------- method
@dataclass
class OCSDConfig:
    """Siêu tham số của OCSD (Bảng 3.x trong khóa luận). Các cờ use_* phục vụ nghiên cứu cắt bỏ."""
    steps: int = 50                  # T - số bước DDIM
    guidance: float = 7.5            # w - CFG
    height: int = 512
    width: int = 512
    negative_prompt: str = "lowres, blurry, deformed, bad anatomy, extra limbs, watermark, text"

    # M2 - sinh ảnh cấp đối tượng
    K: int = 4                        # số ứng viên cho mỗi đối tượng
    obj_steps: int = 30               # số bước khử nhiễu cho ảnh đối tượng
    obj_cn_scale: float = 1.0
    obj_prompt_tmpl: str = "a photo of a {phrase}, simple white background, high quality"
    use_user_phrase: bool = True      # False -> chỉ dùng tên lớp (như Zhang et al.)
    select_weights: tuple = (1.0, 1.0, 1.0)  # lambda_1..3 (CLIP, IoU, conf)

    # M3 - học định danh
    use_identity: bool = True
    S1: int = 200                     # bước học vector nhúng
    S2: int = 200                     # bước học kết hợp (vector nhúng + LoRA)
    lora_scale: float = 0.5           # LoRA strength at inference (training always uses 1.0); tuned value
    lr_emb1: float = 5e-3
    lr_emb2: float = 5e-5
    lr_lora: float = 1e-4
    lora_rank: int = 16
    lambda_att: float = 0.01
    use_att_sep: bool = True

    # M4 - xây dựng cảnh
    use_blend: bool = True
    alpha: float = 0.1                # t > alpha*T: trộn tiềm ẩn; t <= alpha*T: suy luận tùy biến; tuned value
    bg_prompt_tmpl: str = "a photo of {bg}, high quality"
    placement_search: bool = True
    # append the scene caption to the global prompt P_g when it says more than the object phrases + background
    # (COCO captions; a no-op on QuickDraw-Scenes, whose caption is built from the phrases)
    use_caption: bool = True
    # "separate" (thesis OCSD): the background is denoised alone and the objects are pasted back while t > alpha*T;
    # the scene prompt, ControlNet and M5 only act after that. "joint" (OCSD-v2): one trajectory - the U-Net sees the
    # composite at every step (scene prompt, scene ControlNet, M5 from the first step) and the objects are re-imposed
    # inside their masks while t > alpha*T.
    blend_mode: str = "separate"
    anchor_shrink: float = 0.0        # joint: by the last anchored step, the border band of each object (this fraction
                                      # of its inner radius) is left to the model, so contact and edges are redrawn
    m2_per_seed: bool = False         # draw the M2 objects again for every evaluation seed (else once per scene)
    collage_strength: float = 0.3     # "collage" baseline: SDEdit strength after pasting the M2 objects

    # M5 - điều kiện hóa nhận biết đối tượng
    use_region_attn: bool = True      # (a)
    lambda0: float = 8.0
    gamma: float = 1.0
    use_energy: bool = True           # (b)
    beta: float = 1.0
    eta: float = 20.0
    tau: int = 10                     # số bước đầu của giai đoạn tùy biến có dẫn hướng năng lượng
    energy_tokens: str = "phrase"      # tokens whose attention the energy steers: "id" (<o_i>) or "phrase" (object words)
    use_scene_cn: bool = True         # (c)
    omega: float = 0.4
    use_verify: bool = True           # (d)
    R: int = 2                        # số lần sinh lại tối đa
    lambda0_boost: float = 1.5
    verify_thr: float = 0.35
    repair: bool = False              # (d) joint mode: re-denoise only the regions of missing objects / extra detections
    repair_strength: float = 0.5      # fraction of the trajectory re-run inside the repair regions
    region_min_cell: bool = False     # (a)/(b): an object too small to cover 30% of any cell at a coarse attention
                                      # resolution keeps its strongest cell instead of vanishing (and being penalised
                                      # everywhere at that resolution)
    caption_class_masks: bool = False  # (a): caption words that name an object class only attend inside that class's masks
    energy_reduce: str = "sum"        # (b): "sum" over objects (thesis) or "mean" (scale independent of the object count)
    energy_grad_norm: bool = False    # (b): normalise the energy gradient to unit RMS (eta is then the RMS step size)

    # baselines (single-branch methods and the collage baseline)
    cn_scale: float = 1.0             # ControlNet conditioning scale
    adapter_scale: float = 1.0        # T2I-Adapter scale
    gligen_beta: float = 0.3          # GLIGEN scheduled-sampling beta (fraction of steps with grounding)
    bo_n: int = 8                     # best-of-N baselines: samples per image, same Grounding DINO check as M5(d)

    # biến thể câu lệnh (cho ablation "chỉ câu lệnh nền / chỉ câu lệnh toàn cục")
    prompt_mode: str = "both"         # both | bg_only | global_only

    def to_dict(self):
        return asdict(self)

    def replace(self, **kw) -> "OCSDConfig":
        d = self.to_dict()
        d.update(kw)
        return OCSDConfig(**d)


# Defaults before the final configuration was frozen into OCSDConfig (alpha 0.1, LoRA 0.5, energy on phrase tokens,
# caption in P_g). The paper-tier run of 2026-10-01 used these plus results/tuning/tuned.json. Zhang et al. (2025)
# and tuning phases 1-2 still start from them, and stages._archive_stale uses them to know how older images were made.
LEGACY_DEFAULTS = dict(alpha=0.5, lora_scale=1.0, energy_tokens="id", eta=20.0, use_caption=False)
# Convention for fields added later: their default reproduces the behaviour from before they existed, so stored
# settings without the field (outputs/.gen_configs.json) are read as that default (stages._archive_stale).

# Settings every baseline that uses region attention / energy gets (the fixes OCSD-v2 has), see stages.tier_baselines.
BASELINE_DEFAULTS = dict(region_min_cell=True, caption_class_masks=True)


# ----------------------------------------------------------------------------- experiment
@dataclass
class ExperimentConfig:
    backbone: str = "sd15"
    seeds: Optional[List[int]] = None   # None -> lấy theo tier
    # "pilot": kiểm tra nhanh (vài chục cảnh) ; "paper": đầy đủ cho khóa luận/bài báo
    tier: str = "paper"
    eval_detector: str = "owlv2"      # bộ phát hiện dùng để tính OPR/OCE/mIoU/RA
    det_thr: float = 0.30
    match_iou: float = 0.10           # ngưỡng IoU để coi là "bảo toàn" (OPR)
    rel_delta: float = 0.05           # ngưỡng delta cho quan hệ trái/phải/trên/dưới
    fp16: bool = True
    skip_jobs: str = ""               # comma-separated job-name prefixes left out of the plan, e.g. "pareto,E3_quickdraw_power"
    # extra evaluators for the robustness table (results/<split>/det_<name>/): the main OWLv2 again (OPR at stricter
    # IoU), OWLv2 with every benchmark class as a competing query, and a COCO-trained closed-set detector (DETR) on the
    # classes that exist in COCO
    extra_detectors: tuple = ("owlv2", "owlv2d", "detr")
    detr_thr: float = 0.5

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = list(TIERS[self.tier]["seeds"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


TIERS = {
    # Quy mô thực nghiệm, thiết kế cho Google Colab Pro (~100 compute unit/tháng: T4 ~1,8 unit/giờ, L4 ~4,8 unit/giờ).
    # qd_per_cell: số cảnh / ô của QuickDraw-Scenes (12 ô); trained_n: số cảnh chạy phương pháp cần học định danh
    # (OCSD, Zhang et al.) - ablation và alpha lấy trong chính các cảnh này để dùng lại M2/M3 đã học.
    # coco_n: số cảnh COCO-Sketch cho mọi phương pháp không huấn luyện; coco_trained_n: cho OCSD/Zhang.
    # cfg: ghi đè siêu tham số cho MỌI phương pháp (giống nhau -> vẫn công bằng).
    # seeds_all: how many of `seeds` run on EVERY QuickDraw scene; the rest run only on the trained subset.
    # tune: tune alpha and lora_scale on the pilot scenes first (stages.tune), and leave those scenes out.
    # power: methods that run on EVERY QuickDraw scene with EVERY seed (job E3_quickdraw_power), for the
    # pre-registered tests in docs/PREREGISTRATION.md (report.PREREG). power_per_cell (default qd_per_cell) enlarges
    # the power set only: stage A then builds the extra scenes, and the E3/E4 scenes stay exactly the same.
    # pareto: control-vs-quality sweeps on every QuickDraw scene, first seed (job pareto, report.PARETO).
    # tune_baselines: stage T also tunes one knob per baseline with the same rule (TUNE_BASELINES).
    "pilot": dict(qd_per_cell=2, trained_n=8, coco_n=8, coco_trained_n=4, ablation_n=4, alpha_n=4, seeds=[0],
                  seeds_all=1, tune=False, cfg=dict(steps=30, obj_steps=20, K=2, S1=100, S2=100)),
    "paper": dict(qd_per_cell=6, trained_n=36, coco_n=32, coco_trained_n=16, ablation_n=18, alpha_n=12, seeds=[0, 1],
                  seeds_all=1, tune=True, tune_baselines=True, pareto=True,
                  power=["ocsd_v2", "ocsd", "ocsd_lite", "gligen", "controlnet", "gligen_bon", "controlnet_bon"],
                  power_per_cell=6, cfg=dict(steps=30, obj_steps=20, K=2, S1=100, S2=100)),
    "full": dict(qd_per_cell=25, trained_n=180, coco_n=200, coco_trained_n=100, ablation_n=60, alpha_n=40,
                 seeds=[0, 1, 2], seeds_all=3, tune=True, tune_baselines=True, cfg=dict()),
}


# Tuning grid (stages.tune): run on the pilot scenes, which are then excluded from the paper/full evaluation.
# Phases 1-2 start from LEGACY_DEFAULTS, so they repeat the procedure reported in the thesis.
TUNE_GRID = dict(alpha=[0.0, 0.1, 0.2, 0.3, 0.5], lora_scale=[0.5, 1.0])
TUNE_SEEDS = [0, 1]
TUNE_CLIP_TOL = 1.0   # a setting may lose at most this much global CLIP score vs. the reference setting of its phase
# Phase 2 of the tuning (after alpha / lora_scale are fixed): how M5(b) energy guidance is applied.
TUNE_ENERGY = {
    "e_off": dict(use_energy=False),
    "e_id20": dict(use_energy=True, energy_tokens="id", eta=20.0),        # the original setting (reference)
    "e_id10": dict(use_energy=True, energy_tokens="id", eta=10.0),
    "e_ph20": dict(use_energy=True, energy_tokens="phrase", eta=20.0),
    "e_ph10": dict(use_energy=True, energy_tokens="phrase", eta=10.0),
}
# Phase 3 (after phases 1-2): which of M5(a) region attention / M5(b) energy guidance to keep, tuned jointly with
# alpha, since the ablation showed the two overlap. The phase-2 setting at its alpha is the reference and always
# part of the grid, so the phase can only move away from it when another setting scores higher.
TUNE_M5 = {
    "both": dict(use_region_attn=True, use_energy=True),
    "region": dict(use_region_attn=True, use_energy=False),
    "energy": dict(use_region_attn=False, use_energy=True),
}
TUNE_M5_ALPHA = [0.1, 0.3, 0.4, 0.5, 0.6]
# OCSD-v2 phase: the anchoring cut-off alpha (at least 20% of the trajectory is joint generation) x border release.
# Reference: the methods.V2 defaults (alpha 0.4, shrink 0.3).
TUNE_V2 = dict(alpha=[0.2, 0.4, 0.6], anchor_shrink=[0.0, 0.3])
# Baseline phase, in this order (cn_region / cn_energy / collage run with the ControlNet scale chosen first; the
# bo3 / boN variants inherit their base method's value). The reference of each grid is the current default.
TUNE_BASELINES = [
    ("controlnet", "cn_scale", [0.6, 0.8, 1.0, 1.2]),
    ("t2i_adapter", "adapter_scale", [0.6, 0.8, 1.0]),
    ("gligen", "gligen_beta", [0.2, 0.3, 0.5, 1.0]),
    ("cn_region", "lambda0", [4.0, 8.0, 12.0]),
    ("cn_energy", "eta", [10.0, 20.0, 40.0]),
    ("collage", "collage_strength", [0.2, 0.3, 0.5]),
    ("zhang2025", "alpha", [0.1, 0.3, 0.5]),
]
# Control-vs-quality sweeps (job "pareto"): value lists per sweep family, see report.PARETO.
PARETO = dict(pv2=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], pv1lite=[0.0, 0.3, 0.6, 1.0], pcn=[0.4, 0.7, 1.0, 1.3],
              pgl=[0.1, 0.3, 0.6, 1.0])
