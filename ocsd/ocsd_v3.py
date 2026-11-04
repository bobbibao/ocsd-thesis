"""OCSD v3: the complete sketch + text -> scene image model.

Pipeline (one call, no per-scene training):
  1. Instance parsing (OCSD M1): one sketch per object (stroke colour or connected groups), phrase per object,
     box = ink extent, caption from the phrases and background (`ocsd.app.build_scene_from_input`).
  2. Sketch silhouette (OCSD parser, `ocsd.direct_instances.sketch_silhouette`): bridge open stroke ends, close,
     fill holes -> a per-object mask + 256-point polygon. Parsed from the user's sketch only, never oracle masks.
  3. Instance-conditioned generation: the pinned official InstanceDiffusion core (Wang et al., CVPR 2024) with
     boxes, centre points, phrases and the silhouettes; multi-instance sampler (MIS).
  4. Optional naturalness stages fixed by the dev v3 study (assets/direct_dev_v3/protocol.json):
     `layout_only` releases the silhouettes after the MIS layout steps; `refine` applies upstream's SDXL-Refiner
     cascade (strength 0.35, 20 steps).

FROZEN holds the configuration used for every reported OCSD v3 number. It is set once from the dev v3 decision
rule and never tuned on held-out scenes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

NEGATIVE = 'blurry, low quality, distorted, deformed, watermark, text'


@dataclass(frozen=True)
class Config:
    steps: int = 50
    mis: float = .36
    alpha: float = .8
    guidance: float = 7.5
    negative: str = NEGATIVE
    scribble: bool = False          # held-out v2: silhouettes without the 20 ink points
    layout_only: bool = False       # release silhouettes after the MIS merge (dev v3)
    refine: bool = False            # upstream SDXL-Refiner cascade (dev v3)
    refiner_strength: float = .35
    refiner_steps: int = 20
    refiner_revision: str | None = None
    decided_by: str = 'pending dev v3'


# Set from the dev v3 result; see docs/DIRECT_DEV3_PROTOCOL.md.
FROZEN = Config(layout_only=True, refine=True, refiner_revision='5d4cfe854c9a9a87939ff3653551c2b3c99a4356',
                decided_by='dev v3 run dev3_c4c37905e295bb1fb60f, kept arm sil_layout_refine (docs/DIRECT_DEV3_20261104_RESULTS.md)')


class OCSDv3:
    def __init__(self, model, refiner=None, refiner_revision=None, config: Config = FROZEN, device='cuda'):
        self.model, self.refiner, self.config, self.device = model, refiner, config, device
        self.refiner_revision = refiner_revision

    @classmethod
    def load(cls, cache='/content/ocsd-direct-cache', config: Config = FROZEN, device='cuda', lock=None,
             expected_checkpoint_sha256=None):
        from .direct_instances import load_components, load_refiner
        model = load_components(cache, lock=lock, expected_checkpoint_sha256=expected_checkpoint_sha256)
        refiner = revision = None
        if config.refine:
            refiner, revision = load_refiner(cache, config.refiner_revision, device=device)
        return cls(model, refiner, revision, config, device)

    def condition(self, scene):
        from .direct_instances import scene_condition
        return scene_condition(scene, silhouettes=True)

    def generate_scene(self, scene, seed=0, deadline=None, condition=None):
        """scene: objects with .phrase, .cls, .box (pixels), .sketch (H x W uint8, ink < 128). Returns (image, log)."""
        from .direct_instances import refine
        c = self.config
        condition = condition or self.condition(scene)
        pixels, log = self.model.generate(condition, seed, c.negative, scribble=c.scribble, device=self.device,
                                          steps=c.steps, mis=c.mis, alpha=c.alpha, deadline=deadline,
                                          silhouette=True, silhouette_layout_only=c.layout_only)
        log = dict(log, method='ocsd_v3', ocsd_v3_config=asdict(c))
        if c.refine:
            if self.refiner is None: raise RuntimeError('Config asks for the refiner but none is loaded')
            base = pixels
            pixels, rlog = refine(self.refiner, base, condition['caption'], seed, c.refiner_strength, c.refiner_steps,
                                  device=self.device)
            log.update(rlog, refiner_revision=self.refiner_revision)
        return pixels, log

    def __call__(self, sketch_rgb, phrases_text, background, seed=0):
        """Demo entry point: raw sketch image + comma-separated phrases + background text."""
        from .app import build_scene_from_input
        scene = build_scene_from_input(np.asarray(sketch_rgb), phrases_text, background)
        return self.generate_scene(scene, seed)


def with_overrides(**kw):
    return replace(FROZEN, **kw)
