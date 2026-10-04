"""OCSD: M2 (sinh ảnh cấp đối tượng), M3 (học định danh), M4 (xây dựng cảnh) và M5 (điều kiện hóa nhận
biết đối tượng: (a) chú ý giới hạn theo vùng, (b) dẫn hướng năng lượng chú ý, (c) ControlNet cấp cảnh,
(d) kiểm tra hậu sinh). Cài đặt lại Zhang et al. (2025) = cùng khung nhưng tắt M5 và các cải tiến M2/M3."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .attention import binarize_masks, token_maps
from .config import OCSDConfig
from .engine import ID_TOKENS, Engine
from .matching import consistency, verify_pass, verify_score
from .sketch import (CropInfo, GroupedPrompt, Scene, bbox_of, build_grouped_prompt, mask_iou, paste_back,
                     scene_global_prompt, sketch_to_control, square_crop)


# ============================================================================ helpers
def _lat_mask(mask: np.ndarray, h: int, w: int, device, dtype=torch.float32, thr: Optional[float] = None):
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(h, w), mode="area")[0, 0]
    if thr is not None:
        t = (t > thr).float()
    return t.to(device, dtype)


def _on_white(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full_like(img, 255)
    out[mask] = img[mask]
    return out


def _cuda_peak():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2 ** 30
    return 0.0


# ============================================================================ plain sampling (dùng cho baseline và M2)
@torch.no_grad()
def sample(eng: Engine, prompts: Sequence[str], seeds: Sequence[int], cfg: OCSDConfig, control=None,
           cn_scale: float = 0.0, adapter_states=None, steps: Optional[int] = None,
           neg: Optional[str] = None, region: Optional[dict] = None, energy: Optional[dict] = None,
           size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Lấy mẫu DDIM + CFG cho một batch. `region`/`energy` cho baseline có điều khiển vùng/năng lượng.

    region = dict(masks, tokens, bg_tokens, lambda0, gamma[, min_cell]) ;
    energy = dict(groups, masks_lr, eot, beta, eta, tau[, reduce, grad_norm])
    """
    H, W = size or (cfg.height, cfg.width)
    B = len(prompts)
    z = torch.cat([eng.randn(s, 1, H, W) for s in seeds])
    emb_c = eng.encode(prompts)
    emb_u = eng.encode([neg if neg is not None else cfg.negative_prompt] * B)
    ts = eng.set_timesteps(steps or cfg.steps)
    if region is not None:
        eng.ctrl.set_regions(region["masks"], region["tokens"], region["bg_tokens"],
                             min_cell=region.get("min_cell", False))
    T0 = float(ts[0])
    try:
        for j, t in enumerate(ts):
            if region is not None:
                eng.ctrl.bias_enabled = True
                eng.ctrl.lambda_t = region["lambda0"] * (float(t) / T0) ** region["gamma"]
            if energy is not None and j < energy["tau"]:
                eta_j = energy["eta"] * (1 - 0.5 * j / max(energy["tau"], 1))
                zs = []
                for b in range(B):
                    zb, _ = eng.energy_update(z[b:b + 1], t, emb_c[b:b + 1], energy["groups"], energy["masks_lr"],
                                              energy["eot"], energy["beta"], eta_j,
                                              control[b:b + 1] if control is not None else None, cn_scale,
                                              reduce=energy.get("reduce", "sum"),
                                              grad_norm=energy.get("grad_norm", False))
                    zs.append(zb)
                z = torch.cat(zs)
            eps = eng.cfg_eps(z, t, emb_c, emb_u, cfg.guidance, control, cn_scale, adapter_states)
            z = eng.step(eps, t, z)
    finally:
        eng.ctrl.clear_regions()
    return eng.to_image(z)


# ============================================================================ M2
@dataclass
class ObjectResult:
    img: np.ndarray        # ảnh đối tượng 512x512 (khung crop)
    mask: np.ndarray       # mặt nạ SAM (khung crop)
    info: CropInfo         # ánh xạ crop <-> cảnh
    score: float
    candidates: int
    sketch_crop: np.ndarray


def object_branch(eng: Engine, vis, scene: Scene, cfg: OCSDConfig, seed: int = 0,
                  log: Optional[dict] = None) -> List[ObjectResult]:
    """M2: với mỗi phác thảo đối tượng, sinh K ứng viên bằng SD + ControlNet Scribble, tách mặt nạ bằng
    Grounding DINO + SAM, chọn ứng viên theo CLIP(vùng, cụm từ) + IoU(mặt nạ, mặt nạ phác thảo) + độ tin cậy."""
    eng.lora(False)
    out = []
    l1, l2, l3 = cfg.select_weights
    for i, o in enumerate(scene.objects):
        sk_crop, info = square_crop(o.sketch, o.box, cfg.height)
        m_crop, _ = square_crop((o.mask * 255).astype(np.uint8), o.box, cfg.height, fill=0)
        m_crop = m_crop > 127
        phrase = o.phrase if cfg.use_user_phrase else o.cls
        prompt = cfg.obj_prompt_tmpl.format(phrase=phrase)
        control = eng.control_tensor(sketch_to_control(sk_crop), cfg.K)
        seeds = [seed * 1000 + 17 * i + k for k in range(cfg.K)]
        imgs = sample(eng, [prompt] * cfg.K, seeds, cfg, control, cfg.obj_cn_scale, steps=cfg.obj_steps)
        best, best_s = None, -1e9
        for k in range(cfg.K):
            img = imgs[k]
            dets = vis.gdino(img, [o.cls], thr=0.25) if vis is not None else []
            if dets:
                d = max(dets, key=lambda d: d["score"])
                box, conf = d["box"], d["score"]
            else:
                box, conf = bbox_of(m_crop), 0.0
            mask = vis.sam(img, box) if vis is not None else m_crop.copy()
            if mask.sum() < 50:
                mask = m_crop.copy()
            if cfg.K == 1:
                s = 0.0
            else:
                clip = float(vis.clip.score([_on_white(img, mask)], [f"a photo of a {phrase}"])[0]) / 100 \
                    if vis is not None else 0.0
                s = l1 * clip + l2 * mask_iou(mask, m_crop) + l3 * conf
            if s > best_s:
                best, best_s = (img, mask), s
        out.append(ObjectResult(best[0], best[1], info, float(best_s), cfg.K, sk_crop))
    return out


# ============================================================================ M3
def learn_identity(eng: Engine, scene: Scene, objs: List[ObjectResult], cfg: OCSDConfig, seed: int = 0,
                   log: Optional[dict] = None) -> List[str]:
    """M3: học token định danh <o_i> (giai đoạn 1: chỉ vector nhúng; giai đoạn 2: vector nhúng + LoRA) với hàm
    mất mát khuếch tán có mặt nạ L_id và số hạng tách biệt chú ý L_att trên ảnh ghép nhiều đối tượng."""
    eng.reset_identity()
    n = scene.n
    tokens = ID_TOKENS[:n]
    tok = eng.tokenizer
    emb_layer = eng.text_encoder.get_input_embeddings()
    # khởi tạo <o_i> bằng vector nhúng của tên lớp
    with torch.no_grad():
        for i, o in enumerate(scene.objects):
            ids = tok(o.cls, add_special_tokens=False).input_ids
            emb_layer.weight[eng.id_token_ids[i]] = emb_layer.weight[ids].mean(0)
    if cfg.S1 + cfg.S2 == 0:
        return tokens
    rng = random.Random(seed)
    H = cfg.height
    lh = H // eng.vae_factor
    ar = lh // eng.attn_div
    # ---------------- dữ liệu huấn luyện: ảnh đơn + ảnh ghép
    samples = []
    for i, r in enumerate(objs):
        img = _on_white(r.img, r.mask)
        samples.append(dict(z0=eng.to_latent(img), masks=[r.mask], ids=[i],
                            prompt=f"a photo of {tokens[i]} {scene.objects[i].cls}"))
    if n > 1:
        for _ in range(min(6, n * (n - 1))):
            pick = rng.sample(range(n), k=min(n, rng.choice([2, 2, 3])))
            canvas = np.full((H, H, 3), 255, np.uint8)
            cell = H // len(pick)
            masks = []
            for c, i in enumerate(pick):
                r = objs[i]
                ys, xs = np.where(r.mask)
                if len(xs) == 0:
                    continue
                crop = r.img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                mcrop = r.mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                s = min((cell - 8) / crop.shape[1], (H * 0.8) / crop.shape[0])
                cw, chh = max(int(crop.shape[1] * s), 2), max(int(crop.shape[0] * s), 2)
                crop = cv2.resize(crop, (cw, chh))
                mcrop = cv2.resize(mcrop.astype(np.uint8), (cw, chh), interpolation=cv2.INTER_NEAREST) > 0
                x0, y0 = c * cell + (cell - cw) // 2, (H - chh) // 2
                full = np.zeros((H, H), bool)
                full[y0:y0 + chh, x0:x0 + cw] = mcrop
                canvas[full] = crop[mcrop]
                masks.append(full)
            if len(masks) != len(pick):
                continue
            prompt = "a photo of " + " and ".join(f"{tokens[i]} {scene.objects[i].cls}" for i in pick)
            samples.append(dict(z0=eng.to_latent(canvas), masks=masks, ids=pick, prompt=prompt))
    for s in samples:
        s["m_lat"] = torch.stack([_lat_mask(m, lh, lh, eng.device) for m in s["masks"]])
        s["m_att"] = torch.stack([_lat_mask(m, ar, ar, eng.device) for m in s["masks"]])
        s["gp"] = _token_positions(tok, s["prompt"], [tokens[i] for i in s["ids"]])
        s["eot"] = eng.eot_index(s["prompt"])
    # ---------------- tối ưu
    id_rows = torch.tensor(eng.id_token_ids[:n], device=emb_layer.weight.device)
    emb_layer.weight.requires_grad_(True)
    keep = emb_layer.weight.data.clone()
    row_mask = torch.zeros(emb_layer.weight.shape[0], 1, device=emb_layer.weight.device)
    row_mask[id_rows] = 1
    use_lora = cfg.S2 > 0
    if use_lora:
        eng.add_lora(cfg.lora_rank)
    opt = torch.optim.AdamW([emb_layer.weight], lr=cfg.lr_emb1, weight_decay=0.0)
    sch = eng.scheduler
    T_train = sch.config.num_train_timesteps
    losses = []
    t0 = time.time()
    g = torch.Generator("cpu").manual_seed(seed)
    try:
        for step in range(cfg.S1 + cfg.S2):
            if step == cfg.S1 and use_lora:
                eng.lora(True)
                for p in eng.lora_params():
                    p.requires_grad_(True)
                opt = torch.optim.AdamW([{"params": [emb_layer.weight], "lr": cfg.lr_emb2, "weight_decay": 0.0},
                                         {"params": eng.lora_params(), "lr": cfg.lr_lora, "weight_decay": 1e-2}])
            multi = [s for s in samples if len(s["ids"]) > 1]
            s = rng.choice(multi) if (multi and rng.random() < 0.35) else rng.choice(samples[:n])
            noise = torch.randn(s["z0"].shape, generator=g).to(eng.device, eng.dtype)
            t = torch.randint(0, T_train, (1,), generator=g).to(eng.device)
            zt = sch.add_noise(s["z0"], noise, t)
            need_att = cfg.use_att_sep and cfg.lambda_att > 0 and len(s["ids"]) > 1
            eng.ctrl.store, eng.ctrl.store_res = need_att, ar * ar
            eng.ctrl.reset_maps()
            eng.ctrl.bias_enabled = False
            with eng.autocast():
                emb = eng.encode_grad([s["prompt"]])
                pred = eng.unet(zt, t, encoder_hidden_states=emb, return_dict=False)[0]
            m = s["m_lat"].amax(0)[None, None]
            l_id = (((pred.float() - noise.float()) ** 2) * m).sum() / (m.sum() * pred.shape[1] + 1e-6)
            loss = l_id
            if need_att:
                A = eng.ctrl.aggregated_map(0)
                maps = token_maps(A, s["gp"], s["eot"], smooth=False)
                maps = maps / (maps.amax(dim=(1, 2), keepdim=True) + 1e-6)
                l_att = ((maps - s["m_att"].to(maps.dtype)) ** 2).mean()
                loss = loss + cfg.lambda_att * l_att
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if emb_layer.weight.grad is not None:
                emb_layer.weight.grad.mul_(row_mask.to(emb_layer.weight.grad.dtype))
            opt.step()
            with torch.no_grad():   # giữ nguyên các hàng không phải token định danh
                emb_layer.weight.data = torch.where(row_mask.bool(), emb_layer.weight.data, keep)
            losses.append(loss.item())
    finally:
        eng.ctrl.store = False
        eng.ctrl.reset_maps()
        emb_layer.weight.requires_grad_(False)
        for p in eng.lora_params():
            p.requires_grad_(False)
        eng.lora(False)
    if log is not None:
        log["m3_loss_first"] = float(np.mean(losses[:20])) if losses else None
        log["m3_loss_last"] = float(np.mean(losses[-20:])) if losses else None
        log["m3_time"] = time.time() - t0
    return tokens


def _token_positions(tok, prompt: str, words: Sequence[str]) -> List[List[int]]:
    ids = tok(prompt).input_ids
    out = []
    for w in words:
        wid = tok(w, add_special_tokens=False).input_ids
        pos = [p for p in range(len(ids) - len(wid) + 1) if ids[p:p + len(wid)] == wid]
        out.append(list(range(pos[0], pos[0] + len(wid))) if pos else [])
    return out


# ============================================================================ M4
def compose_foreground(scene: Scene, objs: List[ObjectResult], cfg: OCSDConfig):
    """Đặt các ảnh đối tượng vào đúng vị trí phác thảo (tìm tỉ lệ/độ dịch tối đa IoU với m_i), xa trước gần sau."""
    H = cfg.height
    canvas = np.full((H, H, 3), 255, np.uint8)
    order = sorted(range(scene.n), key=lambda i: scene.objects[i].box[3])
    placed = [np.zeros((H, H), bool) for _ in range(scene.n)]
    grid = [(s, dx, dy) for s in (0.9, 1.0, 1.1) for dx in (-8, 0, 8) for dy in (-8, 0, 8)] \
        if cfg.placement_search else [(1.0, 0, 0)]
    for i in order:
        r = objs[i]
        best, best_iou = (1.0, 0, 0), -1
        for s, dx, dy in grid:
            _, pm = paste_back(np.zeros((H, H, 3), np.uint8), None, r.img, r.mask, r.info, s, dx, dy)
            iou = mask_iou(pm, scene.objects[i].mask)
            if iou > best_iou:
                best, best_iou = (s, dx, dy), iou
        canvas, pm = paste_back(canvas, None, r.img, r.mask, r.info, *best)
        for j in range(scene.n):     # đối tượng gần hơn che khuất đối tượng xa hơn
            if j != i:
                placed[j] &= ~pm
        placed[i] = pm
    m_init = np.any(placed, 0)
    return canvas, placed, m_init


# ============================================================================ M4 + M5
@dataclass
class SceneCond:
    """Prompts, sketch control and region masks of one scene for one configuration (shared by both samplers)."""
    text: str
    emb_g: torch.Tensor
    emb_u: torch.Tensor
    emb_bg: torch.Tensor
    emb_ubg: torch.Tensor
    eot: int
    control: Optional[torch.Tensor]
    omega: float
    region_masks: List[np.ndarray]     # m_i OR the placed object, scene resolution
    om: torch.Tensor                   # M5(a) masks at latent resolution: objects, then caption class unions
    att_tokens: List[List[int]]        # token groups aligned with om
    bg_tok: List[int]
    en_groups: List[List[int]]         # M5(b) token groups (objects)
    masks_lr: torch.Tensor             # M5(b) masks at attention-map resolution


def attn_masks(masks: Sequence[np.ndarray], res: int, device, min_cell: bool = False) -> torch.Tensor:
    """Binary masks at attention-map resolution (cell > 30% covered, or the strongest cell with min_cell)."""
    soft = torch.stack([_lat_mask(m, res, res, device) for m in masks])
    return binarize_masks(soft, min_cell).float()


def scene_cond(eng: Engine, scene: Scene, cfg: OCSDConfig, id_tokens: Optional[List[str]], fg=None) -> SceneCond:
    H = cfg.height
    lh = H // eng.vae_factor
    ar = lh // eng.attn_div
    gp = scene_global_prompt(eng.tokenizer, scene, id_tokens, use_phrase=cfg.use_user_phrase, caption=cfg.use_caption,
                             class_groups=cfg.caption_class_masks)
    bg_text = cfg.bg_prompt_tmpl.format(bg=scene.bg)
    neg_bg = cfg.negative_prompt + ", " + ", ".join(sorted({o.cls for o in scene.objects}))
    if cfg.prompt_mode == "bg_only":
        custom = build_grouped_prompt(eng.tokenizer, [(bg_text, "bg")])
    else:
        custom = gp
    bg_branch_text = gp.text if cfg.prompt_mode == "global_only" else bg_text
    control = eng.control_tensor(sketch_to_control(scene.sketch)) if cfg.use_scene_cn else None
    # vùng cho M5: mặt nạ phác thảo m_i hợp với vùng đã đặt đối tượng
    placed = fg[1] if fg is not None else [o.mask for o in scene.objects]
    region_masks = [np.logical_or(o.mask, p) for o, p in zip(scene.objects, placed)]
    obj_tok = [custom.groups.get(f"obj{i}", []) for i in range(scene.n)]
    id_groups = [custom.groups.get(f"id{i}", []) for i in range(scene.n)]
    om = [_lat_mask(m, lh, lh, eng.device) for m in region_masks]
    att_tokens = list(obj_tok)
    for c in sorted({o.cls for o in scene.objects}):      # caption words of a class: inside that class's masks
        toks = custom.groups.get(f"cls:{c}", [])
        if toks:
            om.append(_lat_mask(np.any([m for m, o in zip(region_masks, scene.objects) if o.cls == c], 0),
                                lh, lh, eng.device))
            att_tokens.append(toks)
    return SceneCond(
        text=custom.text, emb_g=eng.encode([custom.text]), emb_u=eng.encode([cfg.negative_prompt]),
        emb_bg=eng.encode([bg_branch_text]),
        emb_ubg=eng.encode([neg_bg if cfg.prompt_mode != "global_only" else cfg.negative_prompt]),
        eot=eng.eot_index(custom.text), control=control, omega=cfg.omega if cfg.use_scene_cn else 0.0,
        region_masks=region_masks, om=torch.stack(om), att_tokens=att_tokens, bg_tok=custom.groups.get("bg", []),
        en_groups=obj_tok if cfg.energy_tokens == "phrase" else id_groups,
        masks_lr=attn_masks(region_masks, ar, eng.device, cfg.region_min_cell))


def anchor_masks(placed: Sequence[np.ndarray], shrink: float, steps: int, res: int, device, dtype) -> List[torch.Tensor]:
    """Joint mode: the anchor mask for each of the first `steps` steps - the union of the placed objects, each
    shrunk from its border by a growing fraction (0 -> `shrink`) of its own inner radius, so small objects keep their
    core and the model redraws contact and edges. Latent resolution, soft (area-averaged) edges."""
    dists = []
    for m in placed:
        if m.any():
            d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
            dists.append(d / max(float(d.max()), 1e-6))
    shape = placed[0].shape if len(placed) else (res, res)
    out = []
    for i in range(steps):
        p = shrink * i / max(steps - 1, 1)
        u = np.zeros(shape, bool)
        for d in dists:
            u |= d > p
        out.append(_lat_mask(u, res, res, device, dtype)[None, None])
    return out


def _denoise(eng: Engine, ctx: SceneCond, cfg: OCSDConfig, ts, start: int, z: torch.Tensor, lambda0: float,
             anchors, use_id: bool, energy_log: List[float]) -> torch.Tensor:
    """Joint-mode denoising from step `start`: before each U-Net call, every anchor (z0, noise, mask(s), until)
    re-imposes q(z0, t) inside its mask while idx < until; then M5(a)/(b)/(c) and CFG on the whole composite.
    Region attention decays from lambda0 at ts[start]; energy acts on the first tau steps from `start`."""
    t_ref = float(ts[start])
    for idx in range(start, len(ts)):
        t = ts[idx]
        for z0, noise, masks, until in anchors:
            if idx < until:
                m = masks[idx] if isinstance(masks, (list, tuple)) else masks
                z = z * (1 - m) + eng.q_sample(z0, noise, t) * m
        eng.lora(use_id)
        eng.ctrl.bias_enabled = cfg.use_region_attn
        eng.ctrl.lambda_t = lambda0 * (float(t) / t_ref) ** cfg.gamma
        j = idx - start
        if cfg.use_energy and j < cfg.tau and any(ctx.en_groups):
            eta_j = cfg.eta * (1 - 0.5 * j / max(cfg.tau, 1))
            z, E = eng.energy_update(z, t, ctx.emb_g, ctx.en_groups, ctx.masks_lr, ctx.eot, cfg.beta, eta_j,
                                     ctx.control, ctx.omega, reduce=cfg.energy_reduce, grad_norm=cfg.energy_grad_norm)
            energy_log.append(E)
        eps = eng.cfg_eps(z, t, ctx.emb_g, ctx.emb_u, cfg.guidance, ctx.control, ctx.omega)
        z = eng.step(eps, t, z)
    return z


class _SceneRun:
    """Region attention + LoRA strength for one sampling run; always restored, even after an error."""

    def __init__(self, eng: Engine, ctx: SceneCond, cfg: OCSDConfig, use_id: bool):
        self.eng, self.ctx, self.cfg, self.use_id = eng, ctx, cfg, use_id

    def __enter__(self):
        if self.use_id and self.cfg.lora_scale != 1.0:
            self.eng.set_lora_scale(self.cfg.lora_scale)
        self.eng.ctrl.set_regions(self.ctx.om, self.ctx.att_tokens, self.ctx.bg_tok, min_cell=self.cfg.region_min_cell)
        return self

    def __exit__(self, *exc):
        self.eng.ctrl.clear_regions()
        if self.use_id and self.cfg.lora_scale != 1.0:
            self.eng.set_lora_scale(1.0)
        self.eng.lora(False)
        return False


def build_scene(eng: Engine, scene: Scene, objs: Optional[List[ObjectResult]], cfg: OCSDConfig, seed: int,
                id_tokens: Optional[List[str]], lambda0: Optional[float] = None, fg=None,
                log: Optional[dict] = None) -> np.ndarray:
    H = cfg.height
    lh = H // eng.vae_factor
    lambda0 = cfg.lambda0 if lambda0 is None else lambda0
    use_id = id_tokens is not None
    ctx = scene_cond(eng, scene, cfg, id_tokens, fg)
    ts = eng.set_timesteps(cfg.steps)
    T = len(ts)
    k = int(round((1 - cfg.alpha) * T)) if (cfg.use_blend and fg is not None) else 0
    k = min(max(k, 0), T)
    energy_E = []
    with torch.no_grad(), _SceneRun(eng, ctx, cfg, use_id):
        if cfg.blend_mode == "joint":
            anchors = []
            if k > 0:
                anchors.append((eng.to_latent(fg[0]), eng.randn(seed + 7919, 1, H, H),
                                anchor_masks(fg[1], cfg.anchor_shrink, k, lh, eng.device, eng.dtype), k))
            z = _denoise(eng, ctx, cfg, ts, 0, eng.randn(seed, 1, H, H), lambda0, anchors, use_id, energy_E)
        elif cfg.blend_mode == "separate":
            z = _separate(eng, ctx, cfg, ts, k, seed, fg, lambda0, use_id, energy_E)
        else:
            raise ValueError(f"unknown blend_mode {cfg.blend_mode!r}")
    if log is not None:
        log.setdefault("energy", []).append(energy_E[:1] + energy_E[-1:])
        log["global_prompt"] = ctx.text
        log["blend_steps"] = k
    return eng.to_image(z)[0]


def _separate(eng: Engine, ctx: SceneCond, cfg: OCSDConfig, ts, k: int, seed: int, fg, lambda0: float, use_id: bool,
              energy_E: List[float]) -> torch.Tensor:
    """Thesis OCSD (blend_mode "separate"): while idx < k the background is denoised alone with the background prompt
    and the objects are pasted back into the latent; the composite meets the scene prompt at step k - 1, and the
    scene ControlNet / M5 act only on the remaining steps."""
    H = cfg.height
    lh = H // eng.vae_factor
    if k > 0:
        z_fg0 = eng.to_latent(fg[0])
        m_lat = _lat_mask(fg[2], lh, lh, eng.device, eng.dtype)[None, None]
        eps_fix = eng.randn(seed + 7919, 1, H, H)
        z_bg = eng.randn(seed, 1, H, H)
    z = eng.randn(seed, 1, H, H)
    for idx, t in enumerate(ts):
        if idx < k:                                            # ---- suy luận trộn tiềm ẩn
            eng.lora(False)
            eng.ctrl.bias_enabled = False
            z_t = z_bg * (1 - m_lat) + eng.q_sample(z_fg0, eps_fix, t) * m_lat
            if idx == k - 1:
                eng.lora(use_id)
                eps = eng.cfg_eps(z_t, t, ctx.emb_g, ctx.emb_u, cfg.guidance)
                z = eng.step(eps, t, z_t)
            else:
                eps_b = eng.cfg_eps(z_bg, t, ctx.emb_bg, ctx.emb_ubg, cfg.guidance)
                z_bg = eng.step(eps_b, t, z_bg)
            continue
        # ---- suy luận tùy biến + điều kiện hóa nhận biết đối tượng (M5)
        eng.lora(use_id)
        frac = float(t) / float(ts[k])
        eng.ctrl.bias_enabled = cfg.use_region_attn
        eng.ctrl.lambda_t = lambda0 * frac ** cfg.gamma
        j = idx - k
        if cfg.use_energy and j < cfg.tau and any(ctx.en_groups):
            eta_j = cfg.eta * (1 - 0.5 * j / max(cfg.tau, 1))
            z, E = eng.energy_update(z, t, ctx.emb_g, ctx.en_groups, ctx.masks_lr, ctx.eot, cfg.beta, eta_j,
                                     ctx.control, ctx.omega, reduce=cfg.energy_reduce, grad_norm=cfg.energy_grad_norm)
            energy_E.append(E)
        eps = eng.cfg_eps(z, t, ctx.emb_g, ctx.emb_u, cfg.guidance, ctx.control, ctx.omega)
        z = eng.step(eps, t, z)
    return z


def repair_regions(scene: Scene, ctx: SceneCond, c: Dict, dets: Sequence[dict], size: int) -> Tuple[np.ndarray, List[int]]:
    """Regions M5(d) re-denoises: the region of every object the check missed, plus the boxes of unmatched detections
    of the scene's classes outside all object regions (extra instances). Returns (mask, missing object indices)."""
    missing = [i for i, (j, _) in enumerate(c["matched"]) if j < 0]
    used = {j for j, _ in c["matched"] if j >= 0}
    objects = np.any(ctx.region_masks, 0)
    M = np.zeros((size, size), bool)
    for i in missing:
        M |= ctx.region_masks[i]
    for jd, d in enumerate(dets):
        if jd in used:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in d["box"]]
        box = np.zeros((size, size), bool)
        box[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        M |= box & ~objects
    if M.any():
        M = cv2.dilate(M.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return M, missing


def repair_scene(eng: Engine, scene: Scene, prep: "Prepared", cfg: OCSDConfig, img: np.ndarray, c: Dict,
                 dets: Sequence[dict], seed: int, lambda0: float, log: Optional[dict] = None) -> Optional[np.ndarray]:
    """M5(d) repair (joint mode): re-noise `img` to (1 - repair_strength) of the trajectory and denoise again only
    inside repair_regions - the rest is held to the previous image at every step (and pasted back at the end), and a
    missing object's M2 latent is re-imposed for the same alpha fraction of the remaining steps. Region attention
    starts at `lambda0`. Returns None when there is nothing to repair."""
    H = cfg.height
    lh = H // eng.vae_factor
    use_id = prep.id_tokens is not None
    ctx = scene_cond(eng, scene, cfg, prep.id_tokens, prep.fg)
    M, missing = repair_regions(scene, ctx, c, dets, H)
    if not M.any():
        return None
    ts = eng.set_timesteps(cfg.steps)
    T = len(ts)
    start = min(max(int(round((1 - cfg.repair_strength) * T)), 0), T - 1)
    z0 = eng.to_latent(img)
    noise = eng.randn(seed + 31337, 1, H, H)
    anchors = [(z0, noise, 1 - _lat_mask(M, lh, lh, eng.device, eng.dtype)[None, None], T)]
    if missing and prep.fg is not None and cfg.use_blend:
        until = start + int(round((1 - cfg.alpha) * (T - start)))
        miss = np.any([prep.fg[1][i] for i in missing], 0)
        if miss.any() and until > start:
            anchors.append((eng.to_latent(prep.fg[0]), eng.randn(seed + 7919, 1, H, H),
                            _lat_mask(miss, lh, lh, eng.device, eng.dtype)[None, None], until))
    energy_E = []
    with torch.no_grad(), _SceneRun(eng, ctx, cfg, use_id):
        z = _denoise(eng, ctx, cfg, ts, start, eng.q_sample(z0, noise, ts[start]), lambda0, anchors, use_id, energy_E)
    out = eng.to_image(z)[0].astype(np.float32)
    soft = np.clip(cv2.GaussianBlur(M.astype(np.float32), (0, 0), 4) * 1.5, 0, 1)[..., None]
    if log is not None:
        log.setdefault("repairs", []).append(dict(missing=len(missing), area=float(M.mean()), start=start))
    return (img.astype(np.float32) * (1 - soft) + out * soft).round().clip(0, 255).astype(np.uint8)


# ============================================================================ toàn bộ OCSD
@dataclass
class Prepared:
    objs: Optional[List[ObjectResult]]
    fg: Optional[tuple]
    id_tokens: Optional[List[str]]
    times: Dict[str, float] = field(default_factory=dict)


def prepare(eng: Engine, vis, scene: Scene, cfg: OCSDConfig, objs: Optional[List[ObjectResult]] = None,
            seed: int = 0, log: Optional[dict] = None) -> Prepared:
    """Các bước 1-3 (M1 đã có trong Scene, M2, M3) - làm một lần cho mỗi cảnh, dùng lại cho mọi seed/biến thể."""
    times = {}
    if objs is None:
        t0 = time.time()
        objs = object_branch(eng, vis, scene, cfg, seed, log)
        times["m2"] = time.time() - t0
    fg = compose_foreground(scene, objs, cfg)
    id_tokens = None
    if cfg.use_identity:
        t0 = time.time()
        id_tokens = learn_identity(eng, scene, objs, cfg, seed, log)
        times["m3"] = time.time() - t0
    return Prepared(objs, fg, id_tokens, times)


def generate(eng: Engine, vis, scene: Scene, prep: Prepared, cfg: OCSDConfig, seed: int,
             eval_iou: float = 0.1, log: Optional[dict] = None) -> np.ndarray:
    """M4 + M5 với vòng kiểm tra hậu sinh M5(d): sinh lại (seed mới, lambda0 tăng) tối đa R lần.
    With blend_mode "joint" and repair, a failed check re-denoises only the failing regions (repair_scene) of the
    best image so far, up to R times, instead of generating the whole image again."""
    log = log if log is not None else {}
    t0 = time.time()
    tries = []
    if cfg.blend_mode == "joint" and cfg.repair:
        best_img = build_scene(eng, scene, prep.objs, cfg, seed, prep.id_tokens, cfg.lambda0, prep.fg, log)
        if cfg.use_verify and vis is not None:
            classes = [o.cls for o in scene.objects]
            dets = vis.gdino(best_img, classes, thr=cfg.verify_thr)
            best_c = consistency(scene, dets, eval_iou)
            best_s = verify_score(best_c)
            tries.append(dict(score=best_s, opr=best_c["opr"], oce_c=best_c["oce_c"], kind="full"))
            for r in range(cfg.R):
                if verify_pass(best_c):
                    break
                img = repair_scene(eng, scene, prep, cfg, best_img, best_c, dets, seed + 100003 * (r + 1),
                                   cfg.lambda0 * cfg.lambda0_boost ** (r + 1), log)
                if img is None:
                    break
                d2 = vis.gdino(img, classes, thr=cfg.verify_thr)
                c = consistency(scene, d2, eval_iou)
                s = verify_score(c)
                tries.append(dict(score=s, opr=c["opr"], oce_c=c["oce_c"], kind="repair"))
                if s > best_s:
                    best_img, best_s, best_c, dets = img, s, c, d2
    else:
        lam = cfg.lambda0
        best_img, best_s = None, -1e9
        n_try = (cfg.R + 1) if cfg.use_verify else 1
        for r in range(n_try):
            img = build_scene(eng, scene, prep.objs, cfg, seed + 100003 * r, prep.id_tokens, lam, prep.fg, log)
            if cfg.use_verify and vis is not None:
                dets = vis.gdino(img, [o.cls for o in scene.objects], thr=cfg.verify_thr)
                c = consistency(scene, dets, eval_iou)
                s = verify_score(c)
                tries.append(dict(score=s, opr=c["opr"], oce_c=c["oce_c"]))
                if s > best_s:
                    best_img, best_s = img, s
                if verify_pass(c):
                    break
                lam *= cfg.lambda0_boost
            else:
                best_img = img
    log["tries"] = len(tries) if tries else 1
    log["verify"] = tries
    log["scene_time"] = time.time() - t0
    log["peak_gb"] = _cuda_peak()
    return best_img
