"""Build the clean farm example scene (layout B of the defense deck) as a normal OCSD scene folder.

Same rendering, masks, boxes and relations as ocsd/data.py; the five QuickDraw drawings are picked by hand so
that no two objects touch (the benchmark scene qd_5_medium_005 has a cow drawn over the truck).
Usage: python tools/make_farm_clean_scene.py QD_DIR OUT_DIR   (QD_DIR holds cow/tree/truck.ndjson from
ocsd.data.download_quickdraw; the indices below are line numbers in those files)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocsd.data import _save_scene
from ocsd.sketch import bbox_of, box_iou, infer_relations, make_caption, mask_from_strokes, render_strokes, to_binary

SID = "farm_clean"
BG = "on a farm"
# (class, line index, box x0 y0 x1 y1 on the 512 canvas, mirror, phrase)
LAYOUT = [("cow", 49, (35, 60, 195, 175), False, "small cow"),
          ("cow", 7, (320, 60, 480, 175), True, "small cow"),
          ("truck", 66, (30, 300, 215, 400), False, "red truck"),
          ("tree", 27, (220, 160, 335, 420), False, "tall tree"),
          ("cow", 33, (345, 290, 495, 415), True, "small cow")]


def build(qd_dir, out_dir, size=512, thickness=3, mask_mode="hull"):
    rows = {c: [json.loads(l) for _, l in zip(range(200), open(os.path.join(qd_dir, f"{c}.ndjson")))]
            for c in {o[0] for o in LAYOUT}}
    canvas = np.full((size, size), 255, np.uint8)
    objs, sk, masks, tight = [], [], [], []
    for cls, idx, box, flip, phrase in LAYOUT:
        d = rows[cls][idx]
        strokes = [[[255 - x for x in s[0]] if flip else s[0], s[1]] for s in d["drawing"]]
        single = render_strokes(strokes, size, thickness, box=box)
        canvas = np.minimum(canvas, single)
        sk.append(single)
        masks.append(mask_from_strokes(single, mode=mask_mode))
        tight.append(bbox_of(to_binary(single)))
        objs.append(dict(cls=cls, phrase=phrase, box=list(map(float, tight[-1])), qd_key=d["key_id"],
                         recognized=bool(d["recognized"]), n_strokes=len(d["drawing"])))
    phrases = [o["phrase"] for o in objs]
    scene = dict(sid=SID, split="quickdraw", generator_version="farm-clean-v1", n_obj=len(objs), count_bin="5",
                 complexity="simple", bg=BG, caption=make_caption(phrases, BG), objects=objs,
                 relations=[list(r) for r in infer_relations(tight, size)],
                 overlap_max=float(max(box_iou(a, b) for i, a in enumerate(tight) for b in tight[i + 1:])),
                 mask_overlap_px=int(sum((masks[i] & masks[j]).sum() for i in range(len(masks)) for j in range(i + 1, len(masks)))))
    _save_scene(os.path.join(out_dir, SID), canvas, sk, masks, scene)
    return scene


if __name__ == "__main__":
    s = build(sys.argv[1], sys.argv[2], mask_mode=sys.argv[3] if len(sys.argv) > 3 else "hull")
    print(s["caption"], "| overlap_max", round(s["overlap_max"], 3), "| mask overlap px", s["mask_overlap_px"])
