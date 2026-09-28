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
        # cache mô hình HuggingFace trên Drive để lần sau không phải tải lại
        return os.path.join(self.root, "hf_cache")

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
    lr_emb1: float = 5e-3
    lr_emb2: float = 5e-5
    lr_lora: float = 1e-4
    lora_rank: int = 16
    lambda_att: float = 0.01
    use_att_sep: bool = True

    # M4 - xây dựng cảnh
    use_blend: bool = True
    alpha: float = 0.5                # t > alpha*T: trộn tiềm ẩn; t <= alpha*T: suy luận tùy biến
    bg_prompt_tmpl: str = "a photo of {bg}, high quality"
    placement_search: bool = True

    # M5 - điều kiện hóa nhận biết đối tượng
    use_region_attn: bool = True      # (a)
    lambda0: float = 8.0
    gamma: float = 1.0
    use_energy: bool = True           # (b)
    beta: float = 1.0
    eta: float = 20.0
    tau: int = 10                     # số bước đầu của giai đoạn tùy biến có dẫn hướng năng lượng
    use_scene_cn: bool = True         # (c)
    omega: float = 0.4
    use_verify: bool = True           # (d)
    R: int = 2                        # số lần sinh lại tối đa
    lambda0_boost: float = 1.5
    verify_thr: float = 0.35

    # biến thể câu lệnh (cho ablation "chỉ câu lệnh nền / chỉ câu lệnh toàn cục")
    prompt_mode: str = "both"         # both | bg_only | global_only

    def to_dict(self):
        return asdict(self)

    def replace(self, **kw) -> "OCSDConfig":
        d = self.to_dict()
        d.update(kw)
        return OCSDConfig(**d)


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

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = list(TIERS[self.tier]["seeds"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


TIERS = {
    # qd_per_cell: số cảnh mỗi ô (số đối tượng x độ phức tạp) của QuickDraw-Scenes (12 ô)
    # coco_n: số cảnh COCO-Sketch; full_identity_n: số cảnh chạy các phương pháp cần học định danh (OCSD, Zhang)
    # "pilot" ~1-2 giờ GPU để kiểm tra; "paper" ~ vừa ngân sách Colab Plus; "full" cho bài báo nếu có thêm GPU
    "pilot": dict(qd_per_cell=3, coco_n=16, ablation_n=8, alpha_n=6, full_identity_n=12, seeds=[0]),
    "paper": dict(qd_per_cell=15, coco_n=100, ablation_n=40, alpha_n=30, full_identity_n=96, seeds=[0, 1, 2]),
    "full": dict(qd_per_cell=25, coco_n=200, ablation_n=60, alpha_n=40, full_identity_n=180, seeds=[0, 1, 2]),
}
