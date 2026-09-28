"""Các mô hình thị giác phụ trợ: Grounding DINO, OWLv2, SAM, CLIP, DINOv2 (nạp lười, dùng chung)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .config import AUX_MODELS

Det = Dict  # {"cls": str, "box": (x0,y0,x1,y1), "score": float}


def _nms(dets: List[Det], iou: float = 0.5) -> List[Det]:
    import torchvision
    if not dets:
        return dets
    boxes = torch.tensor([d["box"] for d in dets], dtype=torch.float32)
    scores = torch.tensor([d["score"] for d in dets], dtype=torch.float32)
    classes = sorted({d["cls"] for d in dets})
    idx = torch.tensor([classes.index(d["cls"]) for d in dets])
    keep = torchvision.ops.batched_nms(boxes, scores, idx, iou).tolist()
    return [dets[k] for k in keep]


class GroundingDINO:
    """Dùng BÊN TRONG phương pháp (M2 chọn ứng viên, M5(d) kiểm tra hậu sinh)."""

    def __init__(self, device="cuda", model_id=AUX_MODELS["gdino"], cache_dir=None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir=cache_dir).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, img: np.ndarray, classes: Sequence[str], thr: float = 0.35, text_thr: float = 0.25) -> List[Det]:
        classes = sorted(set(classes))
        text = ". ".join(c.lower() for c in classes) + "."
        pil = Image.fromarray(img)
        inp = self.proc(images=pil, text=text, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        res = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=thr, text_threshold=text_thr, target_sizes=[pil.size[::-1]])[0]
        labels = res.get("text_labels") or res.get("labels")
        dets = []
        for box, score, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
            lab = str(lab).lower()
            hit = [c for c in classes if c.lower() in lab]
            if not hit:
                continue
            dets.append(dict(cls=max(hit, key=len), box=tuple(box), score=float(score)))
        return _nms(dets)


class OWLv2:
    """Dùng để ĐÁNH GIÁ (độc lập với bộ phát hiện của phương pháp -> tránh 'học tủ' độ đo)."""

    def __init__(self, device="cuda", model_id=AUX_MODELS["owlv2"], cache_dir=None):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.proc = Owlv2Processor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id, cache_dir=cache_dir).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, img: np.ndarray, classes: Sequence[str], thr: float = 0.3) -> List[Det]:
        classes = sorted(set(classes))
        queries = [f"a photo of a {c}" for c in classes]
        pil = Image.fromarray(img)
        inp = self.proc(text=[queries], images=pil, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        H, W = img.shape[:2]
        side = max(H, W)  # OWLv2 đệm ảnh thành hình vuông
        res = self.proc.post_process_grounded_object_detection(out, threshold=thr, target_sizes=[(side, side)])[0]
        dets = []
        for box, score, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()):
            x0, y0, x1, y1 = box
            dets.append(dict(cls=classes[int(lab)], box=(max(0, x0), max(0, y0), min(W, x1), min(H, y1)),
                             score=float(score)))
        return _nms(dets)


class SAM:
    def __init__(self, device="cuda", model_id=AUX_MODELS["sam"], cache_dir=None):
        from transformers import SamModel, SamProcessor
        self.proc = SamProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = SamModel.from_pretrained(model_id, cache_dir=cache_dir).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, img: np.ndarray, box) -> np.ndarray:
        pil = Image.fromarray(img)
        inp = self.proc(pil, input_boxes=[[list(map(float, box))]], return_tensors="pt").to(self.device)
        out = self.model(**inp, multimask_output=True)
        masks = self.proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
        scores = out.iou_scores[0, 0].cpu()
        return masks[0, int(scores.argmax())].numpy().astype(bool)


class CLIPScorer:
    def __init__(self, device="cuda", model_id=AUX_MODELS["clip"], cache_dir=None):
        from transformers import CLIPModel, CLIPProcessor
        self.proc = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).to(device).eval()
        self.device = device

    @torch.no_grad()
    def image_emb(self, imgs: Sequence[np.ndarray]) -> torch.Tensor:
        inp = self.proc(images=[Image.fromarray(i) for i in imgs], return_tensors="pt").to(self.device)
        e = self.model.get_image_features(**inp)
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        return torch.nn.functional.normalize(e.float(), dim=-1)

    @torch.no_grad()
    def text_emb(self, texts: Sequence[str]) -> torch.Tensor:
        inp = self.proc(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
        e = self.model.get_text_features(**inp)
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        return torch.nn.functional.normalize(e.float(), dim=-1)

    def score(self, imgs: Sequence[np.ndarray], texts: Sequence[str]) -> np.ndarray:
        """CLIPScore = 100 * max(cos, 0) cho từng cặp (ảnh_i, văn bản_i)."""
        a, b = self.image_emb(imgs), self.text_emb(texts)
        return (100 * (a * b).sum(-1).clamp(min=0)).cpu().numpy()


class DINOv2:
    def __init__(self, device="cuda", model_id=AUX_MODELS["dinov2"], cache_dir=None):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir).to(device).eval()
        self.device = device

    @torch.no_grad()
    def emb(self, imgs: Sequence[np.ndarray]) -> torch.Tensor:
        inp = self.proc(images=[Image.fromarray(i) for i in imgs], return_tensors="pt").to(self.device)
        e = self.model(**inp).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(e.float(), dim=-1)


class Vision:
    """Nạp lười: chỉ tải mô hình khi thực sự dùng, để tiết kiệm VRAM."""

    def __init__(self, device="cuda", cache_dir=None):
        self.device, self.cache_dir = device, cache_dir
        self._m = {}

    def _get(self, key, cls):
        if key not in self._m:
            self._m[key] = cls(self.device, cache_dir=self.cache_dir)
        return self._m[key]

    @property
    def gdino(self) -> GroundingDINO:
        return self._get("gdino", GroundingDINO)

    @property
    def owlv2(self) -> OWLv2:
        return self._get("owlv2", OWLv2)

    @property
    def sam(self) -> SAM:
        return self._get("sam", SAM)

    @property
    def clip(self) -> CLIPScorer:
        return self._get("clip", CLIPScorer)

    @property
    def dino(self) -> DINOv2:
        return self._get("dino", DINOv2)

    def unload(self, *keys):
        for k in keys or list(self._m):
            self._m.pop(k, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
