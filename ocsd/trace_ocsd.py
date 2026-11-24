"""Trace one OCSD generation (SD 1.5, all five modules M1-M5, M3 on) and save every intermediate.

The image is made with the code and settings of the OCSD runs reported in the thesis (tier "heldout_final" = the
frozen paper_v2 tuning: alpha 0.1, LoRA strength 0.5, energy on phrase tokens eta 20, region attention), in the same
order as ocsd.runner: M2 object_branch -> compose_foreground -> M3 learn_identity -> generate (M4 + M5).
Nothing in the computation is changed; light wrappers only record what passes through:
  M1  the parsed scene: per-object sketches, masks, boxes, relations, grouped prompt
  M2  the K candidates of every object, the Grounding DINO box and SAM mask of each, the selected candidate
  M3  identity tokens <o_i> + LoRA: loss at the start and end of training, and after generation an identity check
      (same seed: "a photo of a <class>" vs "a photo of <o_i> <class>" with the learned token and LoRA)
  M4  the composed foreground, the latent blending phase (background + re-noised objects at every step)
  M5  (a) cross-attention map of every object at a few steps, (b) energy per step, (c) scene ControlNet input,
      the latent at every step (decoded) and the predicted clean image x0, (d) the post-generation check and retries
Output: <drive>/trace_ocsd/<sid>_s<seed>/ (PNG/JPG per module, trace.json, overview sheets, one ZIP).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import torch

METHOD = "ocsd"
TIER = "heldout_final"         # frozen paper_v2 tuning (assets/heldout_v3/tuned_paper_v2.json)
PROTOCOL = "xl_paper_v1"      # only used to import the 72 paper scenes from Drive
REF_RUN = "paper_v3/outputs/runs/20261105T085721Z_c08740c2/quickdraw/ocsd"
DEFAULT_SID = "qd_3_medium_003"     # "an orange bus, a wooden bench and an orange bus near a train station" (slides)
ATTN_STEPS = (0, 10, 20, 26, 27, 28, 29)
PALETTE = [(230, 97, 1), (46, 90, 156), (60, 150, 80), (160, 60, 160), (200, 160, 0), (0, 150, 160), (200, 40, 60),
           (120, 90, 40), (90, 90, 90), (240, 120, 160)]


# ---------------------------------------------------------------------------------------------- small image helpers
def _w(path: Path, img: np.ndarray, size: int | None = 512):
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, 2)
    if size and img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_NEAREST)
    ext = path.suffix.lower()
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92] if ext == ".jpg" else [])
    return str(path)


def _boxes(img: np.ndarray, boxes, labels, colors, width: int = 3) -> np.ndarray:
    out = img.copy()
    f = out.shape[0] / 512
    for b, lab, c in zip(boxes, labels, colors):
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x0, y0), (x1, y1), c, max(1, int(width * f)))
        if lab:
            cv2.putText(out, lab, (x0 + 3, max(y0 + int(16 * f), int(14 * f))), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * f, c,
                        max(1, int(1.5 * f)), cv2.LINE_AA)
    return out


def _tint(img: np.ndarray, mask: np.ndarray, color, a: float = 0.45) -> np.ndarray:
    out = img.astype(np.float32)
    m = mask.astype(bool)
    out[m] = out[m] * (1 - a) + np.array(color, np.float32) * a
    return out.clip(0, 255).astype(np.uint8)


def _heat(base: np.ndarray, amap: np.ndarray) -> np.ndarray:
    a = amap.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    a = cv2.resize(a, base.shape[:2][::-1], interpolation=cv2.INTER_CUBIC).clip(0, 1)
    col = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (base * 0.45 + col * 0.55).astype(np.uint8)


def _label(img: np.ndarray, text: str, h: int = 30) -> np.ndarray:
    bar = np.full((h, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (31, 36, 48), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def _sheet(rows, tile: int = 256, gap: int = 6) -> np.ndarray:
    """rows: list of (row title, [(caption, image), ...])."""
    ncol = max(len(r[1]) for r in rows)
    W = 180 + ncol * (tile + gap)
    out = []
    for title, items in rows:
        cells = [np.full((tile + 30, 180, 3), 255, np.uint8)]
        cv2.putText(cells[0], title, (6, tile // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (217, 105, 43), 2, cv2.LINE_AA)
        for cap, im in items:
            im = np.repeat(im[..., None], 3, 2) if im.ndim == 2 else im
            im = cv2.resize(im.astype(np.uint8), (tile, tile), interpolation=cv2.INTER_AREA)
            cells += [_label(im, cap), np.full((tile + 30, gap, 3), 255, np.uint8)]
        row = np.concatenate(cells, 1)
        row = np.concatenate([row, np.full((row.shape[0], W - row.shape[1], 3), 255, np.uint8)], 1)
        out += [row, np.full((gap, W, 3), 255, np.uint8)]
    return np.concatenate(out, 0)


# ---------------------------------------------------------------------------------------------- recorder
class _Vis:
    """Delegates to the real Vision; records every Grounding DINO / SAM call with the current phase."""

    def __init__(self, vis, rec):
        self._vis, self._rec = vis, rec

    def __getattr__(self, name):
        obj = getattr(self._vis, name)
        if name == "gdino":
            def gdino(img, classes, *a, **k):
                dets = obj(img, classes, *a, **k)
                self._rec.calls.append(dict(kind="gdino", phase=self._rec.phase, img=img, classes=list(classes),
                                            dets=[dict(d) for d in dets], thr=k.get("thr")))
                return dets
            return gdino
        if name == "sam":
            def sam(img, box):
                m = obj(img, box)
                self._rec.calls.append(dict(kind="sam", phase=self._rec.phase, box=list(map(float, box)), mask=m))
                return m
            return sam
        return obj


def _gen(ph) -> bool:
    return bool(ph) and (ph == "scene" or ph.startswith("retry") or ph.startswith("repair"))


class Recorder:
    def __init__(self, eng):
        self.eng = eng
        self.phase = None
        self.calls = []             # gdino / sam calls
        self.samples = []           # M2: (prompts, seeds, images)
        self.steps = []             # per U-Net step: phase, idx, t, z_in (cpu), x0 (cpu), attn maps
        self.energy = []            # (phase, step, E)
        self.ctx = {}               # phase -> SceneCond
        self.anchors = []
        self.fg = None
        self.repair_out = []
        self.outputs = {}           # generation phase -> image it produced
        self.k_blend, self.m_lat, self.last_q = 0, None, None
        self._n = {}

    def install(self):
        from . import method
        eng, rec = self.eng, self
        self._orig = dict(sample=method.sample, build_scene=method.build_scene, repair_scene=method.repair_scene,
                          scene_cond=method.scene_cond, anchor_masks=method.anchor_masks,
                          compose_foreground=method.compose_foreground)
        o = self._orig

        def sample(eng_, prompts, seeds, *a, **k):
            imgs = o["sample"](eng_, prompts, seeds, *a, **k)
            if rec.phase == "m2":
                rec.samples.append(dict(prompts=list(prompts), seeds=list(seeds), imgs=[np.asarray(i) for i in imgs]))
            return imgs

        def build_scene(*a, **k):
            r = sum(1 for p in rec._n if p == "scene" or p.startswith("retry"))
            rec.phase = "scene" if r == 0 else f"retry{r}"
            rec._n[rec.phase], rec.last_q = 0, None
            try:
                out = o["build_scene"](*a, **k)
                rec.outputs[rec.phase] = out
                return out
            finally:
                rec.phase = "verify"

        def repair_scene(*a, **k):
            r = sum(1 for p in rec._n if p.startswith("repair")) + 1
            rec.phase = f"repair{r}"
            rec._n[rec.phase] = 0
            try:
                out = o["repair_scene"](*a, **k)
                rec.repair_out.append(out)
                return out
            finally:
                rec.phase = "verify"

        def scene_cond(*a, **k):
            ctx = o["scene_cond"](*a, **k)
            if rec.phase and rec.phase not in rec.ctx:
                rec.ctx[rec.phase] = ctx
            return ctx

        def anchor_masks(*a, **k):
            out = o["anchor_masks"](*a, **k)
            if rec.phase == "scene":
                rec.anchors = [m.detach().float().cpu()[0, 0].numpy() for m in out]
            return out

        def compose_foreground(*a, **k):
            out = o["compose_foreground"](*a, **k)
            rec.fg = out
            return out

        for name, fn in dict(sample=sample, build_scene=build_scene, repair_scene=repair_scene, scene_cond=scene_cond,
                             anchor_masks=anchor_masks, compose_foreground=compose_foreground).items():
            setattr(method, name, fn)

        self._cfg_eps, self._energy, self._q = eng.cfg_eps, eng.energy_update, eng.q_sample

        def q_sample(z0, noise, t):
            out = rec._q(z0, noise, t)
            if _gen(rec.phase):
                rec.last_q = out.detach().float().cpu()
            return out

        def cfg_eps(z, t, emb_c, emb_u, guidance, control=None, cn_scale=0.0, *a, **k):
            ph = rec.phase
            if not _gen(ph):
                return rec._cfg_eps(z, t, emb_c, emb_u, guidance, control, cn_scale, *a, **k)
            idx = rec._n[ph]
            want_attn = (ph == "scene" or ph.startswith("retry")) and idx in ATTN_STEPS
            if want_attn:
                ctx = rec.ctx.get(ph)
                n_pix = int(ctx.masks_lr.shape[-1] * ctx.masks_lr.shape[-2]) if ctx is not None else None
                eng.ctrl.store, eng.ctrl.store_res = True, n_pix
                eng.ctrl.reset_maps()
            eps = rec._cfg_eps(z, t, emb_c, emb_u, guidance, control, cn_scale, *a, **k)
            maps = None
            if want_attn:
                try:
                    from .attention import token_maps
                    A = eng.ctrl.aggregated_map(-1)
                    ctx = rec.ctx.get(ph)
                    if A is not None and ctx is not None:
                        n = len(ctx.region_masks)
                        maps = token_maps(A.float(), ctx.att_tokens[:n], ctx.eot).cpu().numpy()
                finally:
                    eng.ctrl.store = False
                    eng.ctrl.reset_maps()
            z_in = z.detach().float().cpu()
            comp = None
            if rec.m_lat is not None and rec.last_q is not None and idx < rec.k_blend and not ph.startswith("repair"):
                comp = z_in * (1 - rec.m_lat) + rec.last_q * rec.m_lat      # background + re-noised objects (M4)
            rec.steps.append(dict(phase=ph, idx=idx, t=int(t), z_in=z_in, comp=comp,
                                  eps=eps.detach().float().cpu(), attn=maps, cn_scale=float(cn_scale)))
            rec._n[ph] = idx + 1
            return eps

        def energy_update(z, t, *a, **k):
            z2, E = rec._energy(z, t, *a, **k)
            rec.energy.append(dict(phase=rec.phase, step=rec._n.get(rec.phase, -1), t=int(t), E=float(E)))
            return z2, E

        eng.cfg_eps, eng.energy_update, eng.q_sample = cfg_eps, energy_update, q_sample
        return self

    def uninstall(self):
        from . import method
        for name, fn in self._orig.items():
            setattr(method, name, fn)
        for name in ("cfg_eps", "energy_update", "q_sample"):
            self.eng.__dict__.pop(name, None)


# ---------------------------------------------------------------------------------------------- main
def _loss_plot(path: Path, losses, s1: int) -> str:
    """M3 training loss per step (light) and its 10-step moving average; the dashed line is where stage 2 (LoRA) starts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(losses, np.float64)
    k = min(10, len(y))
    ma = np.convolve(y, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    ax.plot(y, color="#C9CDD4", lw=1)
    ax.plot(np.arange(k - 1, len(y)), ma, color="#D9692B", lw=2.5)
    if 0 < s1 < len(y):
        ax.axvline(s1, color="#2E5A9C", ls="--", lw=1.2)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _x0(eng, z, eps, t):
    ab = eng.scheduler.alphas_cumprod[int(t)].float().cpu()   # add_noise moves the table to the GPU; z and eps are on the CPU
    return (z - (1 - ab).sqrt() * eps) / ab.sqrt()


def _decode(eng, z: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return eng.to_image(z.to(eng.device, eng.dtype))[0]


def ocsd_config(overrides: dict | None = None):
    """The OCSD (SD 1.5, M1-M5) configuration of the thesis runs."""
    from .config import ExperimentConfig
    from .stages import tier_variants
    cfg = tier_variants(ExperimentConfig(tier=TIER))[0][METHOD]
    return cfg.replace(**overrides) if overrides else cfg


def trace(drive_root: str, sid: str = DEFAULT_SID, seed: int = 0, eng=None, vis=None, out_root: str | None = None,
          log=print, scene_dir: str | None = None, overrides: dict | None = None, size: int | None = None) -> dict:
    """scene_dir / overrides / size: a local scene folder, settings and frame size for CPU tests with tiny models."""
    from . import method, xl_study
    from .data import load_scene
    from .engine import ID_TOKENS
    from .matching import consistency, verify_pass
    from .sketch import sketch_to_control

    if scene_dir is None:
        p = xl_study.load_protocol(PROTOCOL, drive_root)      # copies the 72 paper scenes from the Drive benchmark
        if sid not in p["scenes"]:
            raise ValueError(f"{sid} is not one of the 72 paper scenes")
        scene_dir = [d for d in xl_study.scene_dirs(p) if Path(d).name == sid][0]
    scene = load_scene(scene_dir)
    if size:
        from .ocsd_xl import upscale_scene
        scene = upscale_scene(scene, size)
    cfg = ocsd_config(overrides)
    out = Path(out_root or Path(drive_root) / "trace_ocsd") / f"{sid}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if eng is None:
        from .engine import Engine
        eng = Engine.from_pretrained("sd15", "cuda", True, cache_dir=os.environ.get("HF_HOME"), load_adapter=False)
    if vis is None:
        from .vision import Vision
        vis = Vision(str(eng.device), cache_dir=os.environ.get("HF_HOME"))
    if hasattr(vis, "set_scene"):
        vis.set_scene(scene)
    H = cfg.height
    log(f"[trace] {sid} seed {seed}, OCSD (SD 1.5, M1-M5): alpha={cfg.alpha}, blend={cfg.blend_mode}, steps={cfg.steps}, "
        f"M3={cfg.use_identity} (S1 {cfg.S1}, S2 {cfg.S2}, LoRA {cfg.lora_scale}), region attn={cfg.use_region_attn}, "
        f"energy={cfg.use_energy} (eta {cfg.eta}), scene CN omega={cfg.omega}, verify={cfg.use_verify} (R {cfg.R})")

    rec = Recorder(eng).install()
    pv = _Vis(vis, rec)
    t0 = time.time()
    run_log, m3_log, check = {}, {}, []
    try:
        # same order as ocsd.runner for the OCSD family
        rec.phase = "m2"
        objs = method.object_branch(eng, pv, scene, cfg, seed=seed if cfg.m2_per_seed else 0)
        rec.phase = None
        fg = method.compose_foreground(scene, objs, cfg)
        id_tokens = None
        if cfg.use_identity:
            rec.phase = "m3"
            id_tokens = method.learn_identity(eng, scene, objs, cfg, seed=0, log=m3_log)
            rec.phase = None
        T = len(eng.set_timesteps(cfg.steps))
        rec.k_blend = min(max(int(round((1 - cfg.alpha) * T)), 0), T) if cfg.use_blend else 0
        if cfg.blend_mode == "separate" and rec.k_blend:
            lh = H // eng.vae_factor
            rec.m_lat = method._lat_mask(fg[2], lh, lh, "cpu")[None, None].float()
        prep = method.Prepared(objs, fg, id_tokens, {})
        run_log = dict(m3_log)
        img = method.generate(eng, pv, scene, prep, cfg, seed, log=run_log)
        # M3 identity check (after the image, so it cannot change it)
        rec.phase = "m3check"
        if id_tokens is not None:
            for i, o in enumerate(scene.objects):
                plain = method.sample(eng, [f"a photo of a {o.cls}"], [seed + 9000 + i], cfg, steps=cfg.obj_steps)[0]
                eng.set_lora_scale(cfg.lora_scale)
                eng.lora(True)
                try:
                    learned = method.sample(eng, [f"a photo of {id_tokens[i]} {o.cls}"], [seed + 9000 + i], cfg,
                                            steps=cfg.obj_steps)[0]
                finally:
                    eng.lora(False)
                    eng.set_lora_scale(1.0)
                check.append((plain, learned))
    finally:
        rec.phase = None
        rec.uninstall()
        eng.reset_identity()
        eng.ctrl.clear_regions()
        eng.ctrl.reset_maps()
        eng.ctrl.store = False
    gen_s = time.time() - t0
    log(f"[trace] done in {gen_s:.0f} s; writing intermediates")

    meta = dict(sid=sid, seed=seed, method="OCSD (SD 1.5, M1-M5)", tier=TIER, cfg=cfg.to_dict(), n_obj=scene.n,
                caption=scene.caption, bg=scene.bg, seconds=gen_s, files={})
    F = meta["files"]
    n = scene.n
    cols = [PALETTE[i % len(PALETTE)] for i in range(n)]
    labels = [f"{i + 1}:{o.cls}" for i, o in enumerate(scene.objects)]
    rd = lambda f: cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)

    # ---------------- input + M1
    sk = np.repeat(scene.sketch[..., None], 3, 2)
    F["input_sketch"] = _w(out / "00_input" / "sketch.png", sk)
    parsed = sk.copy()
    for o, c in zip(scene.objects, cols):
        parsed = _tint(parsed, o.mask, c, 0.35)
    parsed = _boxes(parsed, [o.box for o in scene.objects], labels, cols)
    F["m1_parsed"] = _w(out / "M1" / "parsed_masks_boxes.png", parsed)
    F["m1_objects"] = [_w(out / "M1" / f"obj{i + 1}_{o.cls}_sketch.png", o.sketch) for i, o in enumerate(scene.objects)]
    meta["m1"] = dict(objects=[dict(cls=o.cls, phrase=o.phrase, box=[round(float(v), 1) for v in o.box],
                                    mask_area=float(o.mask.mean())) for o in scene.objects],
                      relations=[list(r) for r in scene.relations], global_prompt=run_log.get("global_prompt"))
    (out / "M1" / "prompt_and_relations.txt").write_text(
        "Caption: " + scene.caption + "\nGrouped global prompt: " + str(run_log.get("global_prompt")) + "\nRelations:\n"
        + "\n".join(f"  {labels[i]} {r} {labels[j]}" for i, j, r in scene.relations))

    # ---------------- M2
    m2 = []
    m2_calls = [c for c in rec.calls if c["phase"] == "m2"]
    ci = 0
    for i, (o, smp) in enumerate(zip(scene.objects, rec.samples)):
        cands = []
        for k, cimg in enumerate(smp["imgs"]):
            g = m2_calls[ci] if ci < len(m2_calls) and m2_calls[ci]["kind"] == "gdino" else None
            ci += 1 if g is not None else 0
            sm = m2_calls[ci] if ci < len(m2_calls) and m2_calls[ci]["kind"] == "sam" else None
            ci += 1 if sm is not None else 0
            vis_img = cimg.copy()
            if sm is not None:
                vis_img = _boxes(_tint(vis_img, sm["mask"], cols[i], 0.35), [sm["box"]], [o.cls], [cols[i]])
            chosen = bool(np.array_equal(cimg, objs[i].img))
            f = _w(out / "M2" / f"obj{i + 1}_{o.cls}_cand{k + 1}{'_SELECTED' if chosen else ''}.jpg", vis_img)
            cands.append(dict(file=f, seed=smp["seeds"][k], selected=chosen, gdino=[
                dict(box=[round(v, 1) for v in d["box"]], score=round(d["score"], 3)) for d in (g["dets"] if g else [])]))
        m2.append(dict(cls=o.cls, prompt=smp["prompts"][0], score=float(objs[i].score), candidates=cands,
                       sketch_crop=_w(out / "M2" / f"obj{i + 1}_{o.cls}_sketch_crop.png", objs[i].sketch_crop),
                       selected_cutout=_w(out / "M2" / f"obj{i + 1}_{o.cls}_selected_cutout.png",
                                          method._on_white(objs[i].img, objs[i].mask))))
    meta["m2"] = m2

    # ---------------- M3
    m3_rows = []
    for i, (plain, learned) in enumerate(check):
        o = scene.objects[i]
        m3_rows.append(dict(cls=o.cls, token=ID_TOKENS[i],
                            reference=m2[i]["selected_cutout"],
                            class_word=_w(out / "M3" / f"obj{i + 1}_{o.cls}_prompt_class_word.jpg", plain),
                            learned_token=_w(out / "M3" / f"obj{i + 1}_{o.cls}_prompt_learned_token.jpg", learned)))
    meta["m3"] = dict(used=cfg.use_identity, S1=cfg.S1, S2=cfg.S2, lora_rank=cfg.lora_rank, lora_scale=cfg.lora_scale,
                      loss_first20=m3_log.get("m3_loss_first"), loss_last20=m3_log.get("m3_loss_last"),
                      seconds=m3_log.get("m3_time"), skipped_steps=m3_log.get("m3_skipped_steps"), check=m3_rows,
                      losses=m3_log.get("m3_losses"))
    if m3_log.get("m3_losses"):
        F["m3_loss"] = _loss_plot(out / "M3" / "training_loss.png", m3_log["m3_losses"], cfg.S1)

    # ---------------- M4
    canvas, placed, m_init = rec.fg
    F["m4_foreground"] = _w(out / "M4" / "composed_foreground.jpg", canvas)
    pm = np.full((H, H, 3), 255, np.uint8)
    for m, c in zip(placed, cols):
        pm = _tint(pm, m, c, 0.8)
    F["m4_placed"] = _w(out / "M4" / "placed_object_masks.png", pm)
    F["m4_blend_mask"] = _w(out / "M4" / "blend_mask.png", (m_init * 255).astype(np.uint8))
    meta["m4"] = dict(blend_steps=rec.k_blend, total_steps=cfg.steps, alpha=cfg.alpha, blend_mode=cfg.blend_mode)

    # ---------------- M5 trajectory (every generation try), attention, energy
    F["m5c_control"] = _w(out / "M5" / "scene_controlnet_input.png", sketch_to_control(scene.sketch))
    traj = []
    for st in rec.steps:
        ph = st["phase"]
        sub = out / ("M5_denoise" if ph == "scene" else f"M5_{ph}")
        x0_img = _decode(eng, _x0(eng, st["z_in"], st["eps"], st["t"]))
        row = dict(phase=ph, step=st["idx"], t=st["t"], blend=st["comp"] is not None, scene_cn_scale=st["cn_scale"],
                   unet_input=_w(sub / f"step{st['idx']:02d}_t{st['t']:03d}_unet_input.jpg", _decode(eng, st["z_in"])),
                   x0=_w(sub / f"step{st['idx']:02d}_t{st['t']:03d}_predicted_x0.jpg", x0_img))
        if st["comp"] is not None:
            row["composite"] = _w(sub / f"step{st['idx']:02d}_t{st['t']:03d}_blend_composite.jpg", _decode(eng, st["comp"]))
        if st["attn"] is not None:
            row["attn"] = [_w(out / f"M5_attention_{ph}" / f"step{st['idx']:02d}_obj{i + 1}_{scene.objects[i].cls}.jpg",
                              _heat(x0_img, a)) for i, a in enumerate(st["attn"])]
        traj.append(row)
    ctx = rec.ctx.get("scene")
    meta["m5"] = dict(trajectory=traj, energy=rec.energy, omega=cfg.omega, lambda0=cfg.lambda0, gamma=cfg.gamma,
                      eta=cfg.eta, tau=cfg.tau,
                      region_masks=[_w(out / "M5" / f"region_obj{i + 1}.png", (m * 255).astype(np.uint8))
                                    for i, m in enumerate(ctx.region_masks)] if ctx is not None else [])

    # ---------------- M5(d) checks
    checks = []
    for c in [c for c in rec.calls if c["kind"] == "gdino" and c["phase"] == "verify"]:
        cons = consistency(scene, c["dets"], 0.1)          # exactly as method.generate
        im = _boxes(c["img"], [o.box for o in scene.objects], [""] * n, [(170, 170, 170)] * n, 2)
        dc = [(0, 160, 0) if any(j == jd for j, _ in cons["matched"]) else (220, 30, 30) for jd in range(len(c["dets"]))]
        im = _boxes(im, [d["box"] for d in c["dets"]], [f"{d['cls']} {d['score']:.2f}" for d in c["dets"]], dc)
        checks.append(dict(file=_w(out / "M5d" / f"check{len(checks) + 1}.jpg", im), opr=cons["opr"],
                           oce_c=cons["oce_c"], ra=cons["ra"], miou=cons["miou"], passed=bool(verify_pass(cons)),
                           missing=cons["missing"], detections=[dict(cls=d["cls"], box=[round(v, 1) for v in d["box"]],
                                                                     score=round(d["score"], 3)) for d in c["dets"]]))
    chosen = next((ph for ph, o in rec.outputs.items() if np.array_equal(o, img)), "scene")
    meta["m5d"] = dict(checks=checks, verify_log=run_log.get("verify"), tries=run_log.get("tries"), selected_try=chosen)
    F["final"] = _w(out / "final.png", img, None)

    # ---------------- same image as the thesis run?
    ref = Path(drive_root) / REF_RUN / f"{sid}_s{seed}.png"
    if ref.exists():
        r = rd(str(ref)).astype(np.float32)
        diff = float(np.abs(cv2.resize(r, img.shape[:2][::-1]) - img.astype(np.float32)).mean())
        meta["thesis_run_check"] = dict(reference=str(ref), mean_abs_diff=diff, same_image=diff < 3.0)
        log(f"[trace] compared with the thesis-run image: mean |diff| = {diff:.2f}")
    else:
        meta["thesis_run_check"] = dict(reference=str(ref), missing=True)

    # ---------------- overview sheets
    rows_sel = [r for r in traj if r["phase"] == chosen]
    pick = sorted({0, 5, 10, 15, 20, max(rec.k_blend - 1, 0), rec.k_blend, len(rows_sel) - 1})
    pick = [j for j in pick if 0 <= j < len(rows_sel)]
    xt = lambda r: rd(r.get("composite") or r["unet_input"])
    sheet = [("Input / M1", [("sketch", sk), ("M1 parsed", parsed)] + [(labels[i], rd(f)) for i, f in enumerate(F["m1_objects"])]),
             ("M2", [(f"{labels[i]} c{k + 1}{' *' if c['selected'] else ''}", rd(c["file"]))
                     for i, o in enumerate(m2) for k, c in enumerate(o["candidates"])][:10])]
    if m3_rows:
        sheet.append(("M3", sum([[(f"{r['cls']} M2", rd(r["reference"])), (f"'a {r['cls']}'", rd(r["class_word"])),
                                  (f"'{r['token']} {r['cls']}'", rd(r["learned_token"]))] for r in m3_rows], [])[:9]))
    sheet.append(("M4", [("foreground", canvas), ("placed masks", pm), ("blend mask", rd(F["m4_blend_mask"])),
                         ("scene CN input", rd(F["m5c_control"]))]))
    sheet.append(("M4/M5 x_t", [(f"step {rows_sel[j]['step']}{' blend' if rows_sel[j]['blend'] else ''}", xt(rows_sel[j]))
                               for j in pick]))
    sheet.append(("M5 x0-pred", [(f"step {rows_sel[j]['step']}", rd(rows_sel[j]["x0"])) for j in pick]))
    for i in range(n):
        sheet.append((f"M5a {labels[i]}"[:16], [(f"step {r['step']}", rd(r["attn"][i])) for r in rows_sel if r.get("attn")]))
    sheet.append(("M5d / final", [(f"check {k + 1} {'PASS' if c['passed'] else 'FAIL'}", rd(c["file"]))
                                  for k, c in enumerate(checks)] + [("final", img)]))
    F["overview"] = _w(out / "OVERVIEW.jpg", _sheet(sheet), None)
    strip = [("noise", xt(rows_sel[0]))] + [(f"step {rows_sel[j]['step']}", rd(rows_sel[j]["x0"])) for j in pick[1:]]
    F["strip"] = _w(out / "NOISE_TO_IMAGE.jpg", _sheet([("noise -> image", strip + [("final", img)])]), None)

    (out / "trace.json").write_text(json.dumps(meta, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    zpath = out.parent / f"TRACE_{sid}_s{seed}.zip"
    with ZipFile(zpath, "w", ZIP_DEFLATED) as z:
        for f in out.rglob("*"):
            if f.is_file():
                z.write(f, str(f.relative_to(out.parent)))
    meta["zip"] = str(zpath)
    log(f"[trace] wrote {out} and {zpath}")
    return meta
