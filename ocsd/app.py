"""Ứng dụng minh họa Gradio: phác thảo + mô tả -> ảnh cảnh nhất quán đối tượng."""
from __future__ import annotations

import numpy as np

from .config import OCSDConfig
from .sketch import Scene, SceneObject, auto_decompose, bbox_of, infer_relations, make_caption, mask_from_strokes, \
    normalize_sketch, to_binary

_STATE = {}


def _models(backbone):
    if "eng" not in _STATE:
        from .engine import Engine
        from .vision import Vision
        _STATE["eng"] = Engine.from_pretrained(backbone, "cuda", True)
        _STATE["vis"] = Vision("cuda")
    return _STATE["eng"], _STATE["vis"]


def build_scene_from_input(sketch_rgb: np.ndarray, phrases_text: str, bg: str) -> Scene:
    sk = normalize_sketch(sketch_rgb, 512)
    parts = auto_decompose(sk)
    parts = sorted(parts, key=lambda p: bbox_of(to_binary(p))[0])        # trái -> phải
    phrases = [p.strip() for p in phrases_text.split(",") if p.strip()]
    if len(phrases) != len(parts):
        raise ValueError(f"Phát hiện {len(parts)} đối tượng trong phác thảo nhưng có {len(phrases)} cụm từ. "
                         f"Hãy nhập đúng {len(parts)} cụm từ, cách nhau bởi dấu phẩy, theo thứ tự trái -> phải.")
    objs = []
    for p, ph in zip(parts, phrases):
        objs.append(SceneObject(cls=ph.split()[-1], phrase=ph, box=bbox_of(to_binary(p)), mask=mask_from_strokes(p),
                                sketch=p))
    return Scene("app", sk, objs, bg, make_caption(phrases, bg), infer_relations([o.box for o in objs]),
                 dict(count_bin=str(len(objs)), complexity="user"))


def run(sketch, phrases_text, bg, full, alpha, seed, backbone="sd15"):
    from .matching import consistency
    from .method import generate, prepare
    if isinstance(sketch, dict):   # gradio ImageEditor
        sketch = sketch.get("composite")
    sketch = np.asarray(sketch)[..., :3]
    scene = build_scene_from_input(sketch, phrases_text, bg)
    eng, vis = _models(backbone)
    cfg = OCSDConfig().replace(alpha=float(alpha), use_identity=bool(full))
    prep = prepare(eng, vis, scene, cfg, seed=0)
    img = generate(eng, vis, scene, prep, cfg, int(seed))
    eng.reset_identity()
    c = consistency(scene, vis.gdino(img, [o.cls for o in scene.objects]))
    report = (f"Đối tượng yêu cầu: {c['n_in']} | phát hiện đúng: {c['n_preserved']} | OPR = {100 * c['opr']:.0f}% | "
              f"sai số đếm = {c['oce_c']}" + (f" | thiếu: {', '.join(c['missing'])}" if c["missing"] else ""))
    return img, prep.fg[0], report


def launch(backbone="sd15", share=True):
    import gradio as gr
    with gr.Blocks(title="OCSD demo") as demo:
        gr.Markdown("## OCSD: sinh ảnh cảnh nhất quán đối tượng từ phác thảo và văn bản")
        with gr.Row():
            with gr.Column():
                sk = gr.Sketchpad(label="Phác thảo (nét đen, nền trắng)", type="numpy", canvas_size=(512, 512))
                ph = gr.Textbox(label="Mô tả từng đối tượng, trái -> phải, cách nhau dấu phẩy",
                                value="brown horse, white sheep")
                bg = gr.Textbox(label="Mô tả nền", value="on the hillside at sunset")
                full = gr.Checkbox(label="OCSD đầy đủ (học định danh, chậm hơn ~1 phút)", value=False)
                al = gr.Slider(0.3, 1.0, value=0.5, step=0.05, label="alpha")
                seed = gr.Number(value=0, label="seed", precision=0)
                btn = gr.Button("Sinh ảnh", variant="primary")
            with gr.Column():
                out = gr.Image(label="Kết quả")
                fg = gr.Image(label="Ảnh tiền cảnh (M2 + M4)")
                rep = gr.Textbox(label="Kiểm tra nhất quán")
        btn.click(lambda a, b, c, d, e, f: run(a, b, c, d, e, f, backbone), [sk, ph, bg, full, al, seed], [out, fg, rep])
    demo.launch(share=share, debug=False)
