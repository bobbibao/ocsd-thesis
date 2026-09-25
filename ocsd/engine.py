"""Các thao tác khuếch tán cơ bản dùng chung cho OCSD và mọi baseline (cùng scheduler, cùng CFG, cùng seed)."""
from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .attention import OCSDController, attention_energy, install_processors, token_maps
from .config import BACKBONES

N_ID_TOKENS = 16
ID_TOKENS = [f"<o{i}>" for i in range(N_ID_TOKENS)]


class Engine:
    def __init__(self, tokenizer, text_encoder, unet, vae, scheduler, controlnet=None, adapter=None,
                 device="cuda", dtype=torch.float16):
        self.tokenizer, self.text_encoder, self.unet, self.vae = tokenizer, text_encoder, unet, vae
        self.scheduler, self.controlnet, self.adapter = scheduler, controlnet, adapter
        self.device, self.dtype = torch.device(device), dtype
        self.ctrl = OCSDController(tokenizer.model_max_length)
        install_processors(self.unet, self.ctrl)
        self.vae_scale = getattr(vae.config, "scaling_factor", 0.18215)
        self.vae_factor = 2 ** (len(vae.config.block_out_channels) - 1)   # 8 với SD
        self.attn_div = 4        # bản đồ chú ý dùng cho M5(b)/L_att: latent/4 (16x16 với ảnh 512)
        self.has_lora = False
        self._add_id_tokens()

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, backbone: str = "sd15", device: str = "cuda", fp16: bool = True,
                        load_controlnet: bool = True, load_adapter: bool = True, cache_dir: Optional[str] = None):
        from .hfcache import retry_broken
        return retry_broken(lambda: cls._load(backbone, device, fp16, load_controlnet, load_adapter, cache_dir), cache_dir)

    @classmethod
    def _load(cls, backbone, device, fp16, load_controlnet, load_adapter, cache_dir):
        from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler, T2IAdapter, UNet2DConditionModel)
        from transformers import CLIPTextModel, CLIPTokenizer
        b = BACKBONES[backbone]
        dtype = torch.float16 if fp16 else torch.float32
        kw = dict(cache_dir=cache_dir)
        tok = CLIPTokenizer.from_pretrained(b["sd"], subfolder="tokenizer", **kw)
        te = CLIPTextModel.from_pretrained(b["sd"], subfolder="text_encoder", **kw).to(device)  # fp32 (học nhúng)
        unet = UNet2DConditionModel.from_pretrained(b["sd"], subfolder="unet", torch_dtype=dtype, **kw).to(device)
        vae = AutoencoderKL.from_pretrained(b["sd"], subfolder="vae", torch_dtype=dtype, **kw).to(device)
        sch = DDIMScheduler.from_pretrained(b["sd"], subfolder="scheduler", **kw)
        cn = ControlNetModel.from_pretrained(b["controlnet"], torch_dtype=dtype, **kw).to(device) \
            if load_controlnet and b["controlnet"] else None
        ad = T2IAdapter.from_pretrained(b["t2i_adapter"], torch_dtype=dtype, **kw).to(device) \
            if load_adapter and b["t2i_adapter"] else None
        for m in [te, unet, vae, cn, ad]:
            if m is not None:
                m.requires_grad_(False)
                m.eval()
        return cls(tok, te, unet, vae, sch, cn, ad, device, dtype)

    def _add_id_tokens(self):
        added = self.tokenizer.add_tokens(ID_TOKENS)
        if added:
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.id_token_ids = self.tokenizer.convert_tokens_to_ids(ID_TOKENS)
        emb = self.text_encoder.get_input_embeddings().weight
        self._orig_id_emb = emb.data[self.id_token_ids].clone()

    # ------------------------------------------------------------------ text
    @torch.no_grad()
    def encode(self, prompts: Sequence[str]) -> torch.Tensor:
        return self.encode_grad(prompts)

    def encode_grad(self, prompts: Sequence[str]) -> torch.Tensor:
        ids = self.tokenizer(list(prompts), padding="max_length", max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(self.device)
        return self.text_encoder(ids)[0].to(self.dtype)

    def eot_index(self, prompt: str) -> int:
        n = len(self.tokenizer(prompt, truncation=True, max_length=self.tokenizer.model_max_length).input_ids)
        return n - 1

    # ------------------------------------------------------------------ images
    @torch.no_grad()
    def to_latent(self, img: np.ndarray, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        x = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 127.5 - 1
        x = x.to(self.device, self.dtype)
        return self.vae.encode(x).latent_dist.mean * self.vae_scale

    @torch.no_grad()
    def to_image(self, z: torch.Tensor) -> np.ndarray:
        x = self.vae.decode(z.to(self.dtype) / self.vae_scale).sample
        x = ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte()
        return x.permute(0, 2, 3, 1).cpu().numpy()

    def control_tensor(self, control_rgb: np.ndarray, batch: int = 1) -> torch.Tensor:
        """Ảnh điều khiển (nét trắng nền đen, HxWx3 uint8) -> (B,3,H,W) trong [0,1]."""
        t = torch.from_numpy(control_rgb).float().permute(2, 0, 1)[None] / 255.0
        return t.to(self.device, self.dtype).repeat(batch, 1, 1, 1)

    def adapter_states(self, sketch_gray: np.ndarray, scale: float = 1.0) -> List[torch.Tensor]:
        """T2I-Adapter sketch nhận ảnh 1 kênh nét trắng nền đen trong [0,1]."""
        x = torch.from_numpy((sketch_gray < 128).astype(np.float32))[None, None].to(self.device, self.dtype)
        with torch.no_grad():
            st = self.adapter(x)
        return [s * scale for s in st]

    def randn(self, seed: int, batch: int = 1, h: int = 512, w: int = 512) -> torch.Tensor:
        g = torch.Generator("cpu").manual_seed(int(seed))
        c = self.unet.config.in_channels
        f = self.vae_factor
        return torch.randn((batch, c, h // f, w // f), generator=g).to(self.device, self.dtype)

    # ------------------------------------------------------------------ denoising primitives
    def set_timesteps(self, steps: int):
        self.scheduler.set_timesteps(steps, device=self.device)
        return self.scheduler.timesteps

    def unet_eps(self, z, t, emb, control=None, cn_scale: float = 0.0, adapter_states=None,
                 cond_flags: Optional[Sequence[bool]] = None):
        kw = {}
        if control is not None and cn_scale > 0 and self.controlnet is not None:
            down, mid = self.controlnet(z, t, encoder_hidden_states=emb, controlnet_cond=control,
                                        conditioning_scale=cn_scale, return_dict=False)
            kw.update(down_block_additional_residuals=down, mid_block_additional_residual=mid)
        if adapter_states is not None:
            kw["down_intrablock_additional_residuals"] = [
                s.repeat(z.shape[0] // s.shape[0], 1, 1, 1).clone() for s in adapter_states]
        self.ctrl.cond_flags = cond_flags
        return self.unet(z, t, encoder_hidden_states=emb, return_dict=False, **kw)[0]

    def cfg_eps(self, z, t, emb_c, emb_u, guidance: float, control=None, cn_scale: float = 0.0,
                adapter_states=None):
        zz = torch.cat([z, z])
        emb = torch.cat([emb_u.expand_as(emb_c), emb_c])
        ctl = torch.cat([control, control]) if control is not None else None
        eps = self.unet_eps(zz, t, emb, ctl, cn_scale, adapter_states, cond_flags=[False] * len(z) + [True] * len(z))
        eu, ec = eps.chunk(2)
        return eu + guidance * (ec - eu)

    def step(self, eps, t, z):
        return self.scheduler.step(eps, t, z, eta=0.0, return_dict=False)[0]

    def q_sample(self, z0, noise, t):
        return self.scheduler.add_noise(z0, noise, t.reshape(1) if torch.is_tensor(t) else torch.tensor([t]))

    # ------------------------------------------------------------------ attention-energy guidance (M5b)
    def energy_update(self, z, t, emb_c, groups: List[List[int]], masks_lr: torch.Tensor, eot: int,
                      beta: float, eta: float, control=None, cn_scale: float = 0.0):
        """Một bước cập nhật z <- z - eta * dE/dz với E theo công thức năng lượng chú ý.
        masks_lr: (G, h, w) mặt nạ ở độ phân giải bản đồ chú ý (16x16 với ảnh 512)."""
        n_pix = masks_lr.shape[-1] * masks_lr.shape[-2]
        kw = {}
        if control is not None and cn_scale > 0 and self.controlnet is not None:
            with torch.no_grad():
                down, mid = self.controlnet(z, t, encoder_hidden_states=emb_c, controlnet_cond=control,
                                            conditioning_scale=cn_scale, return_dict=False)
            kw.update(down_block_additional_residuals=down, mid_block_additional_residual=mid)
        z = z.detach().requires_grad_(True)
        self.ctrl.store, self.ctrl.store_res = True, n_pix
        self.ctrl.reset_maps()
        self.ctrl.cond_flags = [True]
        with torch.enable_grad():
            self.unet(z, t, encoder_hidden_states=emb_c, return_dict=False, **kw)
            A = self.ctrl.aggregated_map(0)
            if A is None:
                self.ctrl.store = False
                return z.detach(), 0.0
            maps = token_maps(A, groups, eot)
            E = attention_energy(maps, masks_lr.to(maps.dtype), beta)
            grad = torch.autograd.grad(E, z, allow_unused=True)[0] if E.requires_grad else None
        self.ctrl.store = False
        self.ctrl.reset_maps()
        if grad is None:
            return z.detach(), float(E)
        return (z - eta * grad).detach(), float(E)

    # ------------------------------------------------------------------ LoRA / identity
    def add_lora(self, rank: int = 16):
        if self.has_lora:
            return
        from peft import LoraConfig
        cfg = LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                         target_modules=["to_k", "to_q", "to_v", "to_out.0"])
        try:
            self.unet.add_adapter(cfg, adapter_name="ocsd")
        except Exception:
            # undo the half-done injection so the next call reports the real error again
            self.unet._hf_peft_config_loaded = False
            self.unet.__dict__.pop("peft_config", None)
            raise
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data = p.data.float()
        self._lora_init = {n: p.detach().clone() for n, p in self.unet.named_parameters() if "lora" in n}
        install_processors(self.unet, self.ctrl)  # bảo toàn bộ xử lý của OCSD sau khi gắn LoRA
        self.has_lora = True
        self.lora(False)

    def set_lora_scale(self, scale: float):
        if self.has_lora:
            self.unet.set_adapters(["ocsd"], weights=[float(scale)])
            self.lora(False)

    def lora_params(self):
        return [p for n, p in self.unet.named_parameters() if "lora" in n]

    def lora(self, enabled: bool):
        if not self.has_lora:
            return
        if enabled:
            self.unet.enable_adapters()
        else:
            self.unet.disable_adapters()

    def reset_identity(self):
        """Đưa token định danh và LoRA về trạng thái ban đầu trước mỗi cảnh (không rò rỉ giữa các cảnh)."""
        emb = self.text_encoder.get_input_embeddings().weight
        emb.data[self.id_token_ids] = self._orig_id_emb.to(emb.device, emb.dtype)
        if self.has_lora:
            with torch.no_grad():
                for n, p in self.unet.named_parameters():
                    if n in self._lora_init:
                        p.copy_(self._lora_init[n])
            self.lora(False)

    def autocast(self):
        if self.device.type == "cuda" and self.dtype == torch.float16:
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()
