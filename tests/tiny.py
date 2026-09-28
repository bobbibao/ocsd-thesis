"""Mô hình thu nhỏ khởi tạo ngẫu nhiên + bộ phát hiện giả, để kiểm thử toàn bộ pipeline trên CPU
(không cần tải trọng số). Kết quả ảnh vô nghĩa; chỉ kiểm tra tính đúng đắn của luồng xử lý và kích thước."""
from __future__ import annotations

import gzip
import json
import os
import random

import numpy as np
import torch


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def make_clip_tokenizer(bpe_gz: str, out_dir: str):
    """Dựng lại tokenizer CLIP gốc (giống openai/clip-vit-large-patch14) từ tệp BPE của open_clip."""
    from transformers import CLIPTokenizer
    if not os.path.exists(os.path.join(out_dir, "vocab.json")):
        os.makedirs(out_dir, exist_ok=True)
        merges = gzip.open(bpe_gz).read().decode("utf-8").split("\n")[1:49152 - 256 - 2 + 1]
        vocab = list(_bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for m in merges:
            vocab.append("".join(m.split()))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        json.dump({v: i for i, v in enumerate(vocab)}, open(os.path.join(out_dir, "vocab.json"), "w"))
        open(os.path.join(out_dir, "merges.txt"), "w").write("#version: 0.2\n" + "\n".join(merges) + "\n")
    return CLIPTokenizer(os.path.join(out_dir, "vocab.json"), os.path.join(out_dir, "merges.txt"),
                         pad_token="<|endoftext|>", model_max_length=77)


def make_engine(tok_dir: str, bpe_gz: str, image_size: int = 64):
    from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler, T2IAdapter, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel

    from ocsd.engine import Engine
    torch.manual_seed(0)
    tok = make_clip_tokenizer(bpe_gz, tok_dir)
    te = CLIPTextModel(CLIPTextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=37, num_hidden_layers=2,
                                      num_attention_heads=4, max_position_embeddings=77, projection_dim=32))
    unet = UNet2DConditionModel(sample_size=image_size // 2, in_channels=4, out_channels=4, layers_per_block=1,
                                block_out_channels=(32, 64), norm_num_groups=8, cross_attention_dim=32,
                                attention_head_dim=8,
                                down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
                                up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D"))
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(32, 64),
                        down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
                        norm_num_groups=8, sample_size=image_size)
    cn = ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(16, 32))
    ad = T2IAdapter(in_channels=1, channels=[32, 64], num_res_blocks=1, downscale_factor=2,
                    adapter_type="full_adapter")
    sch = DDIMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012, clip_sample=False,
                        set_alpha_to_one=False, steps_offset=1)
    for m in (te, unet, vae, cn, ad):
        m.requires_grad_(False)
        m.eval()
    eng = Engine(tok, te, unet, vae, sch, cn, ad, device="cpu", dtype=torch.float32)
    eng.attn_div = 2   # latent 32 -> bản đồ chú ý 16x16 (như SD ở 512)
    return eng


class _FakeDet:
    """Trả về hộp của đối tượng thật (có nhiễu, đôi khi bỏ sót / nhân bản) - chỉ để kiểm thử luồng."""

    def __init__(self, vis):
        self.vis = vis

    def __call__(self, img, classes, thr=0.3, **kw):
        rng = random.Random(int(img.astype(np.int64).sum()) % 100003)
        H = img.shape[0]
        dets = []
        for c, box in self.vis.current_gt:
            if c not in classes or rng.random() < 0.2:
                continue
            j = [v + rng.gauss(0, 0.03 * H) for v in box]
            dets.append(dict(cls=c, box=tuple(j), score=rng.uniform(thr, 1.0)))
            if rng.random() < 0.1:
                dets.append(dict(cls=c, box=tuple(v + 0.2 * H for v in box), score=rng.uniform(thr, 1.0)))
        return dets


class _FakeCLIP:
    def score(self, imgs, texts):
        return np.array([20 + (hash(t) % 1000) / 100 + float(np.asarray(i).mean()) / 50 for i, t in zip(imgs, texts)])


class _FakeDINO:
    def emb(self, imgs):
        x = torch.stack([torch.from_numpy(np.asarray(i, np.float32)).mean((0, 1)) for i in imgs]) + 1
        return torch.nn.functional.normalize(x, dim=-1)


class FakeVision:
    """Giả lập Vision: gdino/owlv2 dùng hộp thật của cảnh hiện tại (đặt qua set_scene)."""

    def __init__(self):
        self.current_gt = []
        self.gdino = _FakeDet(self)
        self.owlv2 = _FakeDet(self)
        self.clip = _FakeCLIP()
        self.dino = _FakeDINO()

    def set_scene(self, scene, crop=False):
        self.current_gt = [(o.cls, o.box) for o in scene.objects]

    def sam(self, img, box):
        m = np.zeros(img.shape[:2], bool)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        return m
