"""Bộ xử lý chú ý chéo của OCSD.

- M5(a) Chú ý chéo giới hạn theo vùng: cộng ma trận độ lệch B vào QK^T/sqrt(d) trước softmax (công thức 3.x).
- Ghi lại bản đồ chú ý chéo ở độ phân giải 16x16 để tính năng lượng chú ý M5(b) và hàm mất mát tách biệt
  chú ý L_att ở M3. Bản đồ được giữ đồ thị tính toán (có gradient) khi cần.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


class OCSDController:
    """Trạng thái dùng chung cho mọi lớp chú ý chéo trong một lần gọi U-Net."""

    def __init__(self, n_tokens: int = 77):
        self.L = n_tokens
        # ---- độ lệch theo vùng
        self.bias_enabled = False
        self.lambda_t = 0.0
        self.obj_masks: Optional[torch.Tensor] = None    # (N_obj, H, W) float 0/1, độ phân giải ảnh ẩn
        self.fg_mask: Optional[torch.Tensor] = None      # (H, W)
        self.obj_tokens: List[List[int]] = []            # T_i
        self.bg_tokens: List[int] = []                   # T_bg
        self.cond_flags: Optional[Sequence[bool]] = None # phần tử nào trong batch là điều kiện (nhận độ lệch)
        self.min_cell = False                            # keep each mask's strongest cell at coarse resolutions
        self._cache: Dict[int, torch.Tensor] = {}
        # ---- ghi bản đồ chú ý
        self.store = False
        self.store_res: Optional[int] = None             # số vị trí không gian N cần ghi (vd. 16*16)
        self.maps: List[torch.Tensor] = []               # mỗi phần tử (B, N, L) - trung bình trên các đầu

    # ------------------------------------------------------------------ regions
    def set_regions(self, obj_masks: torch.Tensor, obj_tokens: List[List[int]], bg_tokens: List[int],
                    fg_mask: Optional[torch.Tensor] = None, min_cell: bool = False):
        self.obj_masks = obj_masks.float()
        self.fg_mask = (obj_masks.amax(0) if fg_mask is None else fg_mask).float()
        self.obj_tokens = obj_tokens
        self.bg_tokens = bg_tokens
        self.min_cell = min_cell
        self._cache = {}

    def clear_regions(self):
        self.obj_masks = None
        self.fg_mask = None
        self.obj_tokens, self.bg_tokens = [], []
        self.min_cell = False
        self._cache = {}
        self.bias_enabled = False

    def _base_bias(self, n_pix: int, device) -> Optional[torch.Tensor]:
        """Mẫu (N, L) với giá trị 1 tại vị trí cần phạt (nhân với -lambda_t khi dùng)."""
        if self.obj_masks is None:
            return None
        if n_pix in self._cache:
            return self._cache[n_pix]
        H, W = self.obj_masks.shape[-2:]
        r = int(round((H * W / n_pix) ** 0.5))
        h, w = H // r, W // r
        om = binarize_masks(F.interpolate(self.obj_masks[None], size=(h, w), mode="area")[0], self.min_cell)
        fg = F.interpolate(self.fg_mask[None, None], size=(h, w), mode="area")[0, 0] > 0.3
        if self.min_cell:
            fg = fg | om.any(0)
        pen = torch.zeros(h * w, self.L, device=device)
        for i, toks in enumerate(self.obj_tokens):
            toks = [t for t in toks if t < self.L]
            if toks:
                outside = (~om[i]).flatten().float().to(device)
                pen[:, toks] = torch.maximum(pen[:, toks], outside[:, None].expand(-1, len(toks)))
        bgt = [t for t in self.bg_tokens if t < self.L]
        if bgt:
            inside = fg.flatten().float().to(device)
            pen[:, bgt] = torch.maximum(pen[:, bgt], inside[:, None].expand(-1, len(bgt)))
        self._cache[n_pix] = pen
        return pen

    def bias(self, batch: int, heads: int, n_pix: int, n_tok: int, device, dtype) -> Optional[torch.Tensor]:
        if not self.bias_enabled or self.lambda_t <= 0:
            return None
        pen = self._base_bias(n_pix, device)
        if pen is None or pen.shape[1] != n_tok:
            return None
        flags = self.cond_flags if self.cond_flags is not None else [True] * batch
        if len(flags) != batch:
            flags = [True] * batch
        per = torch.stack([pen if f else torch.zeros_like(pen) for f in flags])  # (B, N, L)
        return (-self.lambda_t * per).to(dtype).repeat_interleave(heads, dim=0)

    # ------------------------------------------------------------------ maps
    def reset_maps(self):
        self.maps = []

    def aggregated_map(self, batch_index: int = -1) -> Optional[torch.Tensor]:
        """Trung bình các bản đồ đã ghi -> (N, L) cho phần tử batch_index."""
        if not self.maps:
            return None
        return torch.stack([m[batch_index] for m in self.maps]).mean(0)


class OCSDAttnProcessor:
    """Thay thế bộ xử lý chú ý chéo (attn2) của U-Net. Tự chiếu Q/K/V (qua LoRA nếu có)."""

    def __init__(self, controller: OCSDController):
        self.c = controller

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, ch, hh, ww = hidden_states.shape
            hidden_states = hidden_states.view(b, ch, hh * ww).transpose(1, 2)
        is_cross = encoder_hidden_states is not None
        batch_size, n_pix, _ = hidden_states.shape
        seq_len = (encoder_hidden_states if is_cross else hidden_states).shape[1]
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if not is_cross:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if is_cross:
            bias = self.c.bias(batch_size, attn.heads, n_pix, seq_len, query.device, query.dtype)
            if bias is not None:
                attention_mask = bias if attention_mask is None else attention_mask + bias
        probs = attn.get_attention_scores(query, key, attention_mask)
        if is_cross and self.c.store and (self.c.store_res is None or n_pix == self.c.store_res):
            self.c.maps.append(probs.view(batch_size, attn.heads, n_pix, seq_len).mean(1))

        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, ch, hh, ww)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install_processors(unet, controller: OCSDController):
    """Chỉ thay attn2 (chú ý chéo); chú ý tự thân giữ bộ xử lý mặc định (SDPA, tiết kiệm bộ nhớ)."""
    procs = {}
    for name, proc in unet.attn_processors.items():
        procs[name] = OCSDAttnProcessor(controller) if name.endswith("attn2.processor") else proc
    unet.set_attn_processor(procs)


def binarize_masks(soft: torch.Tensor, min_cell: bool = False, thr: float = 0.3) -> torch.Tensor:
    """(N, h, w) area-downsampled masks -> bool. With min_cell, a non-empty mask that covers no cell by `thr` keeps
    its strongest cell(s), so a small object is not dropped at coarse resolutions."""
    out = soft > thr
    if min_cell:
        flat = soft.flatten(1)
        top = flat.amax(1)
        lost = (~out.flatten(1).any(1)) & (top > 0)
        if lost.any():
            keep = (flat >= top[:, None] * 0.999) & (flat > 0)
            out = torch.where(lost[:, None, None], keep.view_as(out), out)
    return out


# ----------------------------------------------------------------------------- attention maps -> energy
def _gauss_kernel(device, dtype, k: int = 3, sigma: float = 0.5):
    x = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).to(dtype)[None, None]


def token_maps(A: torch.Tensor, groups: List[List[int]], eot: int, smooth: bool = True) -> torch.Tensor:
    """A: (N, L) bản đồ chú ý chéo -> (G, h, w) bản đồ cho từng nhóm token.

    Theo Attend-and-Excite: bỏ token <sot>, nhân 100 rồi softmax trên các token văn bản, làm mịn Gauss."""
    N = A.shape[0]
    h = w = int(round(N ** 0.5))
    txt = A[:, 1:eot].float() * 100
    txt = txt.softmax(-1)
    out = []
    for g in groups:
        idx = [t - 1 for t in g if 1 <= t < eot]
        if not idx:
            out.append(torch.zeros(h, w, device=A.device))
            continue
        m = txt[:, idx].mean(-1).view(1, 1, h, w)
        if smooth:
            m = F.conv2d(F.pad(m, (1, 1, 1, 1), mode="reflect"), _gauss_kernel(m.device, m.dtype))
        out.append(m[0, 0])
    return torch.stack(out)


def attention_energy(maps: torch.Tensor, masks: torch.Tensor, beta: float = 1.0, reduce: str = "sum") -> torch.Tensor:
    """Năng lượng M5(b): sum_i (1 - max_{n in m_i} A_i[n]) + beta * (chú ý ngoài vùng / tổng chú ý).
    reduce="mean" divides by the number of objects, so the gradient scale does not grow with the object count.

    maps: (G, h, w) ; masks: (G, h, w) 0/1 cùng độ phân giải."""
    e = maps.new_zeros(())
    n = 0
    for Ai, mi in zip(maps, masks):
        if mi.sum() == 0:
            continue
        inside_max = (Ai * mi).max()
        out_frac = (Ai * (1 - mi)).sum() / (Ai.sum() + 1e-8)
        e = e + (1 - inside_max) + beta * out_frac
        n += 1
    return e / max(n, 1) if reduce == "mean" else e
