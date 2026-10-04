"""Gradio demo: sketch + descriptions -> object-consistent scene image."""
from __future__ import annotations

import numpy as np

from .sketch import (Scene, SceneObject, auto_decompose, bbox_of, decompose_by_color, infer_relations, make_caption,
                     mask_from_strokes, normalize_sketch, square_resize, to_binary)

_STATE = {}
METHODS = {
    "OCSD-v2 (training-free, recommended)": "ocsd_v2",
    "OCSD-lite (thesis sampler, no identity learning)": "ocsd_lite",
    "OCSD (thesis, identity learning, ~1 min slower)": "ocsd",
}


def _models(backbone):
    if "eng" not in _STATE:
        from .engine import Engine
        from .vision import Vision
        _STATE["eng"] = Engine.from_pretrained(backbone, "cuda", True)
        _STATE["vis"] = Vision("cuda")
    return _STATE["eng"], _STATE["vis"]


def build_scene_from_input(sketch_rgb: np.ndarray, phrases_text: str, bg: str) -> Scene:
    """One object per stroke colour when the sketch uses several colours (black strokes count as one object),
    otherwise connected groups of strokes. Phrases are matched to objects from left to right."""
    rgb = square_resize(np.asarray(sketch_rgb)[..., :3].astype(np.uint8), 512)
    groups = decompose_by_color(rgb)
    if groups:
        parts = [normalize_sketch(g, 512) for g in groups]
        sk = np.min(np.stack(parts), 0)
    else:
        sk = normalize_sketch(rgb, 512)
        parts = auto_decompose(sk)
    parts = sorted(parts, key=lambda p: bbox_of(to_binary(p))[0])        # left -> right
    phrases = [p.strip() for p in phrases_text.split(",") if p.strip()]
    if len(phrases) != len(parts):
        raise ValueError(f"Found {len(parts)} objects in the sketch but {len(phrases)} phrases. Enter exactly "
                         f"{len(parts)} comma-separated phrases, left to right (or draw each object in its own colour).")
    objs = [SceneObject(cls=ph.split()[-1], phrase=ph, box=bbox_of(to_binary(p)), mask=mask_from_strokes(p), sketch=p)
            for p, ph in zip(parts, phrases)]
    return Scene("app", sk, objs, bg, make_caption(phrases, bg), infer_relations([o.box for o in objs]),
                 dict(count_bin=str(len(objs)), complexity="user"))


def run(sketch, phrases_text, bg, method, alpha, seed, backbone="sd15"):
    from .matching import consistency
    from .method import generate, prepare
    from .methods import OCSD_VARIANTS
    if isinstance(sketch, dict):   # gradio ImageEditor
        sketch = sketch.get("composite")
    scene = build_scene_from_input(np.asarray(sketch), phrases_text, bg)
    eng, vis = _models(backbone)
    cfg = OCSD_VARIANTS[METHODS.get(method, "ocsd_v2")].replace(alpha=float(alpha))
    prep = prepare(eng, vis, scene, cfg, seed=int(seed) if cfg.m2_per_seed else 0)
    img = generate(eng, vis, scene, prep, cfg, int(seed))
    eng.reset_identity()
    c = consistency(scene, vis.gdino(img, [o.cls for o in scene.objects]))
    report = (f"Requested objects: {c['n_in']} | detected: {c['n_preserved']} | OPR = {100 * c['opr']:.0f}% | "
              f"count error = {c['oce_c']}" + (f" | missing: {', '.join(c['missing'])}" if c["missing"] else ""))
    return img, prep.fg[0], report


def launch(backbone="sd15", share=True):
    import gradio as gr
    from .methods import OCSD_VARIANTS
    with gr.Blocks(title="OCSD demo") as demo:
        gr.Markdown("## OCSD: object-consistent scene images from a sketch and text")
        with gr.Row():
            with gr.Column():
                sk = gr.Sketchpad(label="Sketch (one colour per object, or black strokes)", type="numpy",
                                  canvas_size=(512, 512))
                ph = gr.Textbox(label="One phrase per object, left to right, comma-separated",
                                value="brown horse, white sheep")
                bg = gr.Textbox(label="Background", value="on the hillside at sunset")
                method = gr.Radio(list(METHODS), value=list(METHODS)[0], label="Method")
                al = gr.Slider(0.0, 1.0, value=OCSD_VARIANTS["ocsd_v2"].alpha, step=0.05,
                               label="alpha (fraction of the steps generated freely)")
                seed = gr.Number(value=0, label="seed", precision=0)
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out = gr.Image(label="Result")
                fg = gr.Image(label="Object composite (M2 + M4)")
                rep = gr.Textbox(label="Consistency check")
        btn.click(lambda a, b, c, d, e, f: run(a, b, c, d, e, f, backbone), [sk, ph, bg, method, al, seed],
                  [out, fg, rep])
    demo.launch(share=share, debug=False)
