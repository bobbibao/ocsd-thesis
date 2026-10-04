"""Kiểm tra lớp bao (OWLv2, CLIP, SAM, DINOv2) với mô hình thu nhỏ ngẫu nhiên: đúng API transformers đã ghim."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import numpy as np, torch
from tiny import make_clip_tokenizer
from ocsd import vision as V
tok = make_clip_tokenizer(os.environ["CLIP_BPE"], "/tmp/ocsd_tok"); tok.model_max_length = 16
txt = dict(vocab_size=len(tok), hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=16)
img = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)

from transformers import Owlv2Config, Owlv2ForObjectDetection, Owlv2Processor, Owlv2ImageProcessor
o = V.OWLv2.__new__(V.OWLv2); o.device = "cpu"
o.model = Owlv2ForObjectDetection(Owlv2Config(text_config=txt, vision_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=4, image_size=64, patch_size=16), projection_dim=32)).eval()
o.proc = Owlv2Processor(image_processor=Owlv2ImageProcessor(size={"height": 64, "width": 64}), tokenizer=tok)
d = o(img, ["cat", "dog"], thr=0.0); print("owlv2", len(d), d[:1])
d = o(img, ["cat"], thr=0.0, distractors=["dog", "car", "cat"]); print("owlv2 + distractors", len(d))
assert all(x["cls"] == "cat" for x in d)            # boxes won by a distractor query are dropped

from transformers import CLIPConfig, CLIPModel, CLIPProcessor, CLIPImageProcessor
c = V.CLIPScorer.__new__(V.CLIPScorer); c.device = "cpu"
c.model = CLIPModel(CLIPConfig(text_config=txt, vision_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=4, image_size=64, patch_size=16), projection_dim=16)).eval()
c.proc = CLIPProcessor(image_processor=CLIPImageProcessor(size={"shortest_edge": 64}, crop_size={"height": 64, "width": 64}), tokenizer=tok)
print("clip", c.score([img, img], ["a cat", "a dog on the road"]))

from transformers import Dinov2Config, Dinov2Model, BitImageProcessor
g = V.DINOv2.__new__(V.DINOv2); g.device = "cpu"
g.model = Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=37, image_size=64, patch_size=16)).eval()
g.proc = BitImageProcessor(size={"shortest_edge": 64}, crop_size={"height": 64, "width": 64})
print("dino", g.emb([img, img]).shape)

from transformers import SamConfig, SamModel, SamProcessor, SamImageProcessor
s = V.SAM.__new__(V.SAM); s.device = "cpu"
cfg = SamConfig(vision_config=dict(hidden_size=32, output_channels=16, num_hidden_layers=2, num_attention_heads=2, image_size=64, patch_size=16, mlp_dim=64, global_attn_indexes=[1], num_pos_feats=8),
                prompt_encoder_config=dict(hidden_size=16, image_size=64, patch_size=16, mask_input_channels=4),
                mask_decoder_config=dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, mlp_dim=32, iou_head_hidden_dim=16))
s.model = SamModel(cfg).eval()
s.proc = SamProcessor(SamImageProcessor(size={"longest_edge": 64}, pad_size={"height": 64, "width": 64}, mask_size={"longest_edge": 16}, mask_pad_size={"height": 16, "width": 16}))
m = s(img, (10, 10, 60, 70)); print("sam", m.shape, m.dtype)
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
r = V.COCODetector.__new__(V.COCODetector); r.device = "cpu"
r.model = DetrForObjectDetection(DetrConfig(use_timm_backbone=True, backbone="resnet18", use_pretrained_backbone=False,
                                            d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                                            decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                                            num_queries=10, num_labels=91,
                                            id2label={i: (V.COCO80[i] if i < 80 else "N/A") for i in range(91)})).eval()
r.proc = DetrImageProcessor(size={"shortest_edge": 64, "longest_edge": 64})
d = r(img, V.COCO80, thr=0.0); print("detr", len(d), d[:1])
assert all(x["cls"] in V.COCO80 for x in d)
print("VISION OK")
