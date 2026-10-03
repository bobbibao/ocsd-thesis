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

from .attention import token_maps
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

    region = dict(masks, tokens, bg_tokens, lambda0, gamma) ; energy = dict(groups, masks_lr, eot, beta, eta, tau)
    """
    H, W = size or (cfg.height, cfg.width)
    B = len(prompts)
    z = torch.cat([eng.randn(s, 1, H, W) for s in seeds])
    emb_c = eng.encode(prompts)
    emb_u = eng.encode([neg if neg is not None else cfg.negative_prompt] * B)
    ts = eng.set_timesteps(steps or cfg.steps)
    if region is not None:
        eng.ctrl.set_regions(region["masks"], region["tokens"], region["bg_tokens"])
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
                                              control[b:b + 1] if control is not None else None, cn_scale)
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
def build_scene(eng: Engine, scene: Scene, objs: Optional[List[ObjectResult]], cfg: OCSDConfig, seed: int,
                id_tokens: Optional[List[str]], lambda0: Optional[float] = None, fg=None,
                log: Optional[dict] = None) -> np.ndarray:
    H = cfg.height
    lh = H // eng.vae_factor
    ar = lh // eng.attn_div
    lambda0 = cfg.lambda0 if lambda0 is None else lambda0
    use_id = id_tokens is not None
    gp = scene_global_prompt(eng.tokenizer, scene, id_tokens, use_phrase=cfg.use_user_phrase, caption=cfg.use_caption)
    bg_text = cfg.bg_prompt_tmpl.format(bg=scene.bg)
    neg_bg = cfg.negative_prompt + ", " + ", ".join(sorted({o.cls for o in scene.objects}))
    if cfg.prompt_mode == "bg_only":
        custom = build_grouped_prompt(eng.tokenizer, [(bg_text, "bg")])
    else:
        custom = gp
    bg_branch_text = gp.text if cfg.prompt_mode == "global_only" else bg_text
    emb_g = eng.encode([custom.text])
    emb_u = eng.encode([cfg.negative_prompt])
    emb_bg = eng.encode([bg_branch_text])
    emb_ubg = eng.encode([neg_bg if cfg.prompt_mode != "global_only" else cfg.negative_prompt])
    eot = eng.eot_index(custom.text)

    control = eng.control_tensor(sketch_to_control(scene.sketch)) if cfg.use_scene_cn else None
    # vùng cho M5: mặt nạ phác thảo m_i hợp với vùng đã đặt đối tượng
    if fg is not None:
        _, placed, m_init = fg
    else:
        placed, m_init = [o.mask for o in scene.objects], np.any([o.mask for o in scene.objects], 0)
    region_masks = [np.logical_or(o.mask, p) for o, p in zip(scene.objects, placed)]
    obj_tok = [custom.groups.get(f"obj{i}", []) for i in range(scene.n)]
    id_groups = [custom.groups.get(f"id{i}", []) for i in range(scene.n)]
    bg_tok = custom.groups.get("bg", [])
    en_groups = obj_tok if cfg.energy_tokens == "phrase" else id_groups
    om = torch.stack([_lat_mask(m, lh, lh, eng.device) for m in region_masks])
    masks_lr = torch.stack([_lat_mask(m, ar, ar, eng.device, thr=0.3) for m in region_masks])

    ts = eng.set_timesteps(cfg.steps)
    T = len(ts)
    k = int(round((1 - cfg.alpha) * T)) if (cfg.use_blend and fg is not None) else 0
    k = min(max(k, 0), T)
    if k > 0:
        z_fg0 = eng.to_latent(fg[0])
        m_lat = _lat_mask(m_init, lh, lh, eng.device, eng.dtype)[None, None]
        eps_fix = eng.randn(seed + 7919, 1, H, H)
        z_bg = eng.randn(seed, 1, H, H)
    z = eng.randn(seed, 1, H, H)
    energy_E = []
    if use_id and cfg.lora_scale != 1.0:
        eng.set_lora_scale(cfg.lora_scale)
    with torch.no_grad():
        eng.ctrl.set_regions(om, obj_tok, bg_tok)
        try:
            for idx, t in enumerate(ts):
                if idx < k:                                            # ---- suy luận trộn tiềm ẩn
                    eng.lora(False)
                    eng.ctrl.bias_enabled = False
                    z_t = z_bg * (1 - m_lat) + eng.q_sample(z_fg0, eps_fix, t) * m_lat
                    if idx == k - 1:
                        eng.lora(use_id)
                        eps = eng.cfg_eps(z_t, t, emb_g, emb_u, cfg.guidance)
                        z = eng.step(eps, t, z_t)
                    else:
                        eps_b = eng.cfg_eps(z_bg, t, emb_bg, emb_ubg, cfg.guidance)
                        z_bg = eng.step(eps_b, t, z_bg)
                    continue
                # ---- suy luận tùy biến + điều kiện hóa nhận biết đối tượng (M5)
                eng.lora(use_id)
                frac = float(t) / float(ts[k])
                eng.ctrl.bias_enabled = cfg.use_region_attn
                eng.ctrl.lambda_t = lambda0 * frac ** cfg.gamma
                j = idx - k
                if cfg.use_energy and j < cfg.tau and any(en_groups):
                    eta_j = cfg.eta * (1 - 0.5 * j / max(cfg.tau, 1))
                    z, E = eng.energy_update(z, t, emb_g, en_groups, masks_lr, eot, cfg.beta, eta_j,
                                             control, cfg.omega if cfg.use_scene_cn else 0.0)
                    energy_E.append(E)
                eps = eng.cfg_eps(z, t, emb_g, emb_u, cfg.guidance, control, cfg.omega if cfg.use_scene_cn else 0.0)
                z = eng.step(eps, t, z)
        finally:
            eng.ctrl.clear_regions()
            if use_id and cfg.lora_scale != 1.0:
                eng.set_lora_scale(1.0)
            eng.lora(False)
    if log is not None:
        log.setdefault("energy", []).append(energy_E[:1] + energy_E[-1:])
        log["global_prompt"] = custom.text
        log["blend_steps"] = k
    return eng.to_image(z)[0]


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
    """M4 + M5 với vòng kiểm tra hậu sinh M5(d): sinh lại (seed mới, lambda0 tăng) tối đa R lần."""
    log = log if log is not None else {}
    t0 = time.time()
    tries = []
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
