"""Sketch-fidelity metric: does each generated object follow its own sketch's outline and facing direction?

Detector metrics (OWLv2 count/OPR/mIoU) cannot see pose; human reviews showed pose/shape to be the main failure.
For each instance: segment the generated image inside the instance's target box (SAM, box prompt), take the mask
outline, and measure how close the instance's raw sketch ink lies to that outline (chamfer distance, normalised by
the box diagonal). Facing: the ink should fit the outline better than its own left-right mirror image (mirrored
inside the box). Only instances whose ink differs clearly from its mirror (asymmetric drawings) count for facing.

The reference is the raw sketch ink, the same for every method; the silhouette parser is not used here.
Thresholds were fixed before any held-out v2 output.
"""
import numpy as np

ASYMMETRY_MIN = 0.04   # mean ink-to-mirrored-ink distance / box diagonal for a drawing to count for facing
EMPTY_MASK_MIN = 0.02  # a mask covering < 2% of its box counts as "object not found"


def _box_px(box, scale):
    x0, y0, x1, y1 = [float(v)*scale for v in box]
    return int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))


def _outline(mask):
    from scipy import ndimage
    return mask & ~ndimage.binary_erosion(mask, iterations=1, border_value=0)


def instance_fidelity(mask, ink, box):
    """mask: generated-object mask (H, W) bool; ink: sketch ink (H, W) bool, same frame; box: pixel x0,y0,x1,y1."""
    from scipy import ndimage
    x0, y0, x1, y1 = box
    h, w = mask.shape
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(w, x1), min(h, y1)
    diag = float(np.hypot(x1-x0, y1-y0))
    sub_ink = ink[y0:y1, x0:x1]; ys, xs = np.nonzero(sub_ink)
    if not len(xs): raise ValueError('Instance has no ink inside its box')
    mirror = np.zeros_like(sub_ink); mirror[ys, (x1-x0-1)-xs] = True
    dist_to_ink = ndimage.distance_transform_edt(~sub_ink)
    asymmetry = float(dist_to_ink[mirror].mean())/diag
    sub_mask = mask[y0:y1, x0:x1]
    found = float(sub_mask.mean()) >= EMPTY_MASK_MIN
    if not found:
        return dict(found=False, chamfer=1.0, chamfer_mirror=1.0, asymmetry=asymmetry,
                    facing_eligible=asymmetry >= ASYMMETRY_MIN, facing_correct=False)
    edge = _outline(mask)[y0:y1, x0:x1]
    if not edge.any(): edge = _outline(np.pad(sub_mask, 1))[1:-1, 1:-1]
    dist = ndimage.distance_transform_edt(~edge)
    chamfer = float(dist[sub_ink].mean())/diag; chamfer_mirror = float(dist[mirror].mean())/diag
    eligible = asymmetry >= ASYMMETRY_MIN
    return dict(found=True, chamfer=chamfer, chamfer_mirror=chamfer_mirror, asymmetry=asymmetry,
                facing_eligible=eligible, facing_correct=bool(eligible and chamfer < chamfer_mirror))


def image_fidelity(image, scene, segment):
    """image: HxWx3 uint8; scene: loaded Scene (objects with .box, .sketch in the scene frame);
    segment(image, box) -> bool mask in image frame. Returns per-instance rows and the image summary."""
    h = image.shape[0]; scale = h/scene.sketch.shape[0]
    rows = []
    for i, obj in enumerate(scene.objects):
        ink = np.asarray(obj.sketch) < 128
        if scale != 1:
            from PIL import Image
            ink = np.asarray(Image.fromarray(ink.astype(np.uint8)*255).resize((image.shape[1], h), Image.Resampling.NEAREST)) > 127
        box = _box_px(obj.box, scale)
        mask = np.asarray(segment(image, box), dtype=bool)
        rows.append(dict(index=i, cls=obj.cls, **instance_fidelity(mask, ink, box)))
    eligible = [r for r in rows if r['facing_eligible']]
    summary = dict(sketch_chamfer=float(np.mean([r['chamfer'] for r in rows])),
                   found_rate=float(np.mean([r['found'] for r in rows])),
                   facing_acc=float(np.mean([r['facing_correct'] for r in eligible])) if eligible else float('nan'),
                   facing_n=len(eligible))
    return rows, summary
