"""SDXL engine for OCSD-XL: the same diffusion primitives as `Engine` (DDIM, CFG, ControlNet, region attention,
attention energy, identity tokens + LoRA) on Stable Diffusion XL, so `ocsd.method` (M2-M5) runs unchanged.

SDXL needs two text encoders plus a pooled embedding and size conditioning. To keep every caller of `encode` /
`cfg_eps` unchanged, the pooled vector travels as one extra row behind the 77 token rows:
    encode(prompts) -> (B, 77 + 1, 2048); row 77 holds the pooled embedding (first `pool_dim` values).
`_unet` / `_control` split it again and add `text_embeds` / `time_ids`.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .engine import Engine

XL_MODELS = dict(
    sd="stabilityai/stable-diffusion-xl-base-1.0",
    vae="madebyollin/sdxl-vae-fp16-fix",                 # the stock SDXL VAE overflows in half precision
    controlnet="xinsir/controlnet-scribble-sdxl-1.0",    # scribble ControlNet trained for hand-drawn lines
)


class EngineXL(Engine):
    def __init__(self, tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet, vae, scheduler, controlnet=None,
                 device="cuda", dtype=torch.float16):
        self.tokenizer_2, self.text_encoder_2 = tokenizer_2, text_encoder_2
        self.pool_dim = int(text_encoder_2.config.projection_dim)
        super().__init__(tokenizer, text_encoder, unet, vae, scheduler, controlnet, None, device, dtype)
        self.revisions = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, device: str = "cuda", fp16: bool = True, cache_dir: Optional[str] = None,
                        models: Optional[dict] = None, revisions: Optional[dict] = None):
        from .hfcache import retry_broken
        return retry_broken(lambda: cls._load_xl(device, fp16, cache_dir, dict(XL_MODELS, **(models or {})),
                                                 revisions or {}), cache_dir)

    @classmethod
    def _load_xl(cls, device, fp16, cache_dir, b, revisions):
        from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        dtype = torch.float16 if fp16 else torch.float32

        def kw(repo, **extra):
            return dict(cache_dir=cache_dir, **({"revision": revisions[repo]} if repo in revisions else {}), **extra)

        sd = b["sd"]
        tok = CLIPTokenizer.from_pretrained(sd, subfolder="tokenizer", **kw(sd))
        tok2 = CLIPTokenizer.from_pretrained(sd, subfolder="tokenizer_2", **kw(sd))
        # text encoders stay in float32 (identity-token learning), like Engine
        te = CLIPTextModel.from_pretrained(sd, subfolder="text_encoder", **kw(sd)).to(device)
        te2 = CLIPTextModelWithProjection.from_pretrained(sd, subfolder="text_encoder_2", **kw(sd)).to(device)
        unet = UNet2DConditionModel.from_pretrained(sd, subfolder="unet", torch_dtype=dtype, use_safetensors=True,
                                                    **kw(sd, **({"variant": "fp16"} if fp16 else {}))).to(device)
        vae = AutoencoderKL.from_pretrained(b["vae"], torch_dtype=dtype, **kw(b["vae"])).to(device)
        cn = ControlNetModel.from_pretrained(b["controlnet"], torch_dtype=dtype, **kw(b["controlnet"])).to(device) \
            if b.get("controlnet") else None
        for m in (te, te2, unet, vae, cn):
            if m is not None:
                m.requires_grad_(False)
                m.eval()
        eng = cls(tok, tok2, te, te2, unet, vae, xl_scheduler(), cn, device, dtype)
        eng.revisions = dict(revisions)
        return eng

    # ------------------------------------------------------------------ text
    def text_stacks(self):
        return [(self.tokenizer, self.text_encoder), (self.tokenizer_2, self.text_encoder_2)]

    def encode_grad(self, prompts: Sequence[str]) -> torch.Tensor:
        hidden, pooled = [], None
        # The text encoders are float32 and must stay so: M3 calls this inside eng.autocast(), and the second SDXL
        # encoder overflows in half precision (rounds 1-2: every M3 training step had a non-finite loss).
        with torch.autocast(self.device.type, enabled=False):
            for tok, te in self.text_stacks():
                ids = tok(list(prompts), padding="max_length", max_length=self.tokenizer.model_max_length,
                          truncation=True, return_tensors="pt").input_ids.to(self.device)
                out = te(ids, output_hidden_states=True)
                hidden.append(out.hidden_states[-2].float())
                if te is self.text_encoder_2:
                    pooled = out.text_embeds.float()
        h = torch.cat(hidden, dim=-1)
        row = F.pad(pooled, (0, h.shape[-1] - pooled.shape[-1]))[:, None]
        return torch.cat([h, row], dim=1).to(self.dtype)

    # ------------------------------------------------------------------ denoising primitives
    def _split(self, z, emb):
        tokens, pooled = emb[:, :-1], emb[:, -1, :self.pool_dim]
        h, w = z.shape[-2] * self.vae_factor, z.shape[-1] * self.vae_factor
        ids = torch.tensor([[h, w, 0, 0, h, w]], device=z.device, dtype=pooled.dtype).repeat(z.shape[0], 1)
        return tokens, dict(text_embeds=pooled, time_ids=ids)

    def _unet(self, z, t, emb, **kw):
        tokens, added = self._split(z, emb)
        return self.unet(z, t, encoder_hidden_states=tokens, added_cond_kwargs=added, return_dict=False, **kw)[0]

    def _control(self, z, t, emb, control, cn_scale: float):
        tokens, added = self._split(z, emb)
        down, mid = self.controlnet(z, t, encoder_hidden_states=tokens, controlnet_cond=control,
                                    conditioning_scale=cn_scale, added_cond_kwargs=added, return_dict=False)
        return dict(down_block_additional_residuals=down, mid_block_additional_residual=mid)

    def adapter_states(self, *a, **k):
        raise NotImplementedError("No T2I-Adapter arm on SDXL")


def xl_scheduler():
    """DDIM with the SDXL noise schedule (same sampler family as the SD 1.5 experiments)."""
    from diffusers import DDIMScheduler
    return DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
                         clip_sample=False, set_alpha_to_one=False, steps_offset=1, timestep_spacing="leading",
                         prediction_type="epsilon")
