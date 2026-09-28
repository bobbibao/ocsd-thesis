"""Kiểm tra build_coco_sketch với một bộ COCO giả nhỏ (định dạng chuẩn COCO)."""
import json, os, sys, cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from ocsd.data import build_coco_sketch, load_scene, background_from_caption
D = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fakecoco"
os.makedirs(f"{D}/val2017", exist_ok=True); os.makedirs(f"{D}/annotations", exist_ok=True)
rng = np.random.default_rng(0)
imgs, anns, caps = [], [], []
aid = 1
for iid in range(1, 13):
    H, W = 480, 640
    im = np.full((H, W, 3), 200, np.uint8)
    n = 1 + (iid % 6)
    for k in range(n):
        x, y = int(rng.integers(100, 500)), int(rng.integers(60, 400))
        w, h = int(rng.integers(40, 90)), int(rng.integers(40, 70))
        poly = [x, y, x + w, y, x + w, y + h, x, y + h]
        cv2.rectangle(im, (x, y), (x + w, y + h), (int(rng.integers(0, 255)), 50, 50), -1)
        anns.append(dict(id=aid, image_id=iid, category_id=1 + k % 2, segmentation=[poly], area=w * h,
                         bbox=[x, y, w, h], iscrowd=0)); aid += 1
    cv2.imwrite(f"{D}/val2017/{iid:012d}.jpg", im)
    imgs.append(dict(id=iid, file_name=f"{iid:012d}.jpg", height=H, width=W))
    caps.append(dict(id=iid, image_id=iid, caption="A dog and a cat sitting on the green grass."))
cats = [dict(id=1, name="dog", supercategory="animal"), dict(id=2, name="cat", supercategory="animal")]
json.dump(dict(images=imgs, annotations=anns, categories=cats), open(f"{D}/annotations/instances_val2017.json", "w"))
json.dump(dict(images=imgs, annotations=caps), open(f"{D}/annotations/captions_val2017.json", "w"))
made = build_coco_sketch(D, f"{D}/bench/coco", n_scenes=8, ref_dir=f"{D}/ref", n_ref=5)
sc = load_scene(made[0]); print(sc.sid, sc.n, sc.bg, "|", sc.caption, sc.objects[0].box, os.listdir(made[0]))
print(background_from_caption("A man riding a horse on a sandy beach."), "|", background_from_caption("Two dogs play."))
print("COCO OK", len(made), len(os.listdir(f"{D}/ref")))
