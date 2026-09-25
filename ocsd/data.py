"""Xây dựng và nạp hai bộ kiểm thử:

1. QuickDraw-Scenes (tổng hợp, có kiểm soát): ma trận số đối tượng {1,3,5,8+} x độ phức tạp {simple, medium,
   complex}. Phác thảo đối tượng lấy từ Google QuickDraw (vẽ tay thật của người không chuyên), đặt lên khung
   512x512 theo bố cục có phối cảnh. Biết chính xác lớp, hộp, mặt nạ, quan hệ của từng đối tượng.
2. COCO-Sketch (ảnh thật): ảnh COCO val2017 có 1-8 đối tượng nổi bật; phác thảo sinh bằng PiDiNet (dạng
   scribble), mỗi đối tượng có mặt nạ phân đoạn thật; có ảnh thật để tính FID/KID và câu mô tả do người viết.

Mỗi cảnh lưu thành một thư mục: sketch.png, obj_XX.png, masks.npz, scene.json (+ real.png với COCO).
"""
from __future__ import annotations

import glob
import io
import json
import os
import random
import re
import zipfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .sketch import (Scene, SceneObject, bbox_of, box_iou, infer_relations, make_caption, mask_from_strokes,
                     normalize_sketch, render_strokes, to_binary)

# ============================================================================ QuickDraw-Scenes
# tên lớp QuickDraw -> tên lớp dùng trong văn bản / bộ phát hiện
QD_CLASSES: Dict[str, str] = {
    "cat": "cat", "dog": "dog", "horse": "horse", "sheep": "sheep", "cow": "cow", "elephant": "elephant",
    "bear": "bear", "zebra": "zebra", "giraffe": "giraffe", "bird": "bird", "duck": "duck", "rabbit": "rabbit",
    "car": "car", "bus": "bus", "truck": "truck", "bicycle": "bicycle", "motorbike": "motorcycle",
    "airplane": "airplane", "sailboat": "sailboat", "train": "train", "tree": "tree", "house": "house",
    "bench": "bench", "umbrella": "umbrella", "chair": "chair", "traffic light": "traffic light",
    "fire hydrant": "fire hydrant", "teddy-bear": "teddy bear",
}

ANIMAL_ATTR = ["brown", "black", "white", "gray", "golden", "small", "big"]
OBJECT_ATTR = ["red", "blue", "white", "black", "yellow", "green", "orange", "old", "shiny"]
NO_ATTR = {"zebra", "giraffe"}
CLASS_ATTR = {"tree": ["tall", "green", "small", "leafy"], "house": ["red", "white", "wooden", "small", "big"],
              "bench": ["wooden", "white", "old", "green"], "traffic light": ["small", "old"],
              "fire hydrant": ["red", "yellow", "old"]}
ANIMALS = {"cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "bird", "duck", "rabbit",
           "teddy bear"}

# bối cảnh -> các lớp phù hợp (để cảnh hợp lý về ngữ nghĩa)
BACKGROUNDS: Dict[str, List[str]] = {
    "in a green meadow": ["cat", "dog", "horse", "sheep", "cow", "rabbit", "bird", "tree", "house", "bench", "bicycle"],
    "on a farm": ["horse", "sheep", "cow", "dog", "cat", "duck", "truck", "tree", "house", "bird"],
    "in a savanna at sunset": ["elephant", "zebra", "giraffe", "tree", "bird", "truck", "car"],
    "on a city street": ["car", "bus", "truck", "bicycle", "motorcycle", "traffic light", "fire hydrant", "dog",
                         "bench", "tree", "house", "umbrella", "cat"],
    "in a park": ["dog", "cat", "bench", "tree", "bicycle", "bird", "umbrella", "chair", "duck", "rabbit"],
    "on a sandy beach": ["dog", "umbrella", "chair", "sailboat", "bird", "house", "cat"],
    "on a snowy field": ["dog", "bear", "tree", "house", "car", "horse", "rabbit"],
    "in a forest": ["bear", "tree", "rabbit", "bird", "dog", "house", "horse"],
    "by a lake": ["duck", "sailboat", "tree", "house", "bird", "dog", "bench", "cow"],
    "in a living room": ["cat", "dog", "chair", "teddy bear", "umbrella"],
    "near a train station": ["train", "bus", "car", "bench", "traffic light", "dog", "bicycle"],
    "at an airport": ["airplane", "bus", "truck", "car", "bird"],
}

COUNTS = {"1": (1, 1), "3": (3, 3), "5": (5, 5), "8+": (8, 10)}
COMPLEXITIES = ("simple", "medium", "complex")
COMPLEXITY_SPEC = {
    # tercile: nhóm độ chi tiết của nét vẽ (0 = ít nét nhất / trừu tượng nhất), overlap: IoU tối đa giữa 2 hộp
    "simple": dict(tercile=2, recognized=True, max_overlap=0.0, jitter=0.0, thickness=3),
    "medium": dict(tercile=1, recognized=True, max_overlap=0.15, jitter=0.0, thickness=3),
    "complex": dict(tercile=0, recognized=None, max_overlap=0.30, jitter=2.5, thickness=2),
}


def download_quickdraw(out_dir: str, classes: Iterable[str] = QD_CLASSES, per_class: int = 3000,
                       timeout: int = 60) -> None:
    """Tải N nét vẽ đầu tiên của mỗi lớp (định dạng simplified ndjson, đọc dạng stream - không tải cả file)."""
    import requests
    os.makedirs(out_dir, exist_ok=True)
    for c in classes:
        path = os.path.join(out_dir, f"{c}.ndjson")
        if os.path.exists(path) and sum(1 for _ in open(path)) >= per_class:
            continue
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{c.replace(' ', '%20')}.ndjson"
        n = 0
        with requests.get(url, stream=True, timeout=timeout) as r, open(path, "w") as f:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                f.write(line.decode() + "\n")
                n += 1
                if n >= per_class:
                    break
        print(f"QuickDraw {c}: {n} drawings")


def _load_qd(out_dir: str, qd_name: str) -> List[dict]:
    rows = [json.loads(l) for l in open(os.path.join(out_dir, f"{qd_name}.ndjson"))]
    for r in rows:
        r["n_strokes"] = len(r["drawing"])
        r["n_points"] = sum(len(s[0]) for s in r["drawing"])
    return rows


def _terciles(rows: List[dict]) -> List[List[dict]]:
    """Chia nét vẽ của một lớp thành 3 nhóm theo độ chi tiết (số điểm + 10 x số nét)."""
    score = np.array([r["n_points"] + 10 * r["n_strokes"] for r in rows])
    q1, q2 = np.quantile(score, [1 / 3, 2 / 3])
    out = [[], [], []]
    for r, s in zip(rows, score):
        out[0 if s <= q1 else (1 if s <= q2 else 2)].append(r)
    return out


def _jitter_strokes(strokes, sigma: float, rng: random.Random):
    """Nhiễu nét (độ phức tạp cao): dịch ngẫu nhiên từng điểm, mô phỏng nét run tay."""
    out = []
    for s in strokes:
        xs = [x + rng.gauss(0, sigma) for x in s[0]]
        ys = [y + rng.gauss(0, sigma) for y in s[1]]
        out.append([xs, ys])
    return out


def _sample_layout(n: int, aspects: Sequence[float], max_overlap: float, rng: random.Random,
                   size: int = 512, tries: int = 4000) -> Optional[List[Tuple[float, float, float, float]]]:
    """Bố cục có phối cảnh: đối tượng ở thấp (gần) lớn hơn đối tượng ở cao (xa); giới hạn chồng lấn."""
    base = {1: 0.62, 2: 0.45, 3: 0.40, 4: 0.34, 5: 0.32}.get(n, 0.24)
    for _ in range(tries):
        boxes = []
        ok = True
        for a in aspects:
            placed = False
            for _ in range(200):
                cy = rng.uniform(0.30, 0.80) * size
                persp = 0.65 + 0.7 * (cy / size - 0.3)              # 0.65 .. 1.0
                s = base * persp * rng.uniform(0.85, 1.15) * size   # cạnh dài
                w, h = (s, s / a) if a >= 1 else (s * a, s)
                cx = rng.uniform(w / 2 + 4, size - w / 2 - 4)
                cy = min(max(cy, h / 2 + 4), size - h / 2 - 4)
                b = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if all(box_iou(b, o) <= max_overlap + 1e-6 for o in boxes):
                    # với max_overlap = 0 thêm khoảng cách tối thiểu để các nét không chạm nhau
                    if max_overlap == 0 and any(_gap(b, o) < 6 for o in boxes):
                        continue
                    boxes.append(b)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return boxes
    return None


def _gap(a, b):
    return max(max(a[0], b[0]) - min(a[2], b[2]), max(a[1], b[1]) - min(a[3], b[3]))


def build_quickdraw_scenes(qd_dir: str, out_dir: str, per_cell: int = 25, seed: int = 2026,
                           size: int = 512, dup_prob: float = 0.4) -> List[str]:
    """Sinh bộ QuickDraw-Scenes: per_cell cảnh cho mỗi ô (số đối tượng x độ phức tạp).

    dup_prob: xác suất một đối tượng mới là bản sao lớp đã có (vd. 'three sheep') - để kiểm tra lỗi đếm.
    """
    rng = random.Random(seed)
    pools = {}
    for qd, name in QD_CLASSES.items():
        p = os.path.join(qd_dir, f"{qd}.ndjson")
        if os.path.exists(p):
            rows = _load_qd(qd_dir, qd)
            pools[name] = dict(all=rows, terc=_terciles(rows))
    if not pools:
        raise FileNotFoundError("Chưa tải QuickDraw - hãy chạy download_quickdraw trước.")
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for cnt_key, (lo, hi) in COUNTS.items():
        for cx in COMPLEXITIES:
            spec = COMPLEXITY_SPEC[cx]
            k = 0
            while k < per_cell:
                sid = f"qd_{cnt_key.replace('+', 'p')}_{cx}_{k:03d}"
                sdir = os.path.join(out_dir, sid)
                if os.path.exists(os.path.join(sdir, "scene.json")):
                    made.append(sdir)
                    k += 1
                    continue
                n = rng.randint(lo, hi)
                bg = rng.choice(list(BACKGROUNDS))
                pool = [c for c in BACKGROUNDS[bg] if c in pools]
                classes: List[str] = []
                dup_of: List[int] = []
                while len(classes) < n:
                    if classes and rng.random() < dup_prob:
                        src = rng.randrange(len(classes))   # bản sao: cùng lớp, cùng cụm từ
                        classes.append(classes[src])
                        dup_of.append(dup_of[src] if dup_of[src] >= 0 else src)
                    else:
                        classes.append(rng.choice(pool))
                        dup_of.append(-1)
                drawings = []
                for c in classes:
                    cand = pools[c]["terc"][spec["tercile"]]
                    if spec["recognized"] is not None:
                        cand = [r for r in cand if r["recognized"] == spec["recognized"]] or cand
                    drawings.append(rng.choice(cand))
                aspects = []
                for d in drawings:
                    xs = np.concatenate([s[0] for s in d["drawing"]])
                    ys = np.concatenate([s[1] for s in d["drawing"]])
                    aspects.append(float(np.clip((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1), 0.4, 2.5)))
                boxes = _sample_layout(n, aspects, spec["max_overlap"], rng, size)
                if boxes is None:
                    continue
                # vẽ: xa trước, gần sau (cạnh dưới nhỏ trước)
                order = sorted(range(n), key=lambda i: boxes[i][3])
                canvas = np.full((size, size), 255, np.uint8)
                obj_sketches = [None] * n
                for i in order:
                    strokes = drawings[i]["drawing"]
                    if spec["jitter"] > 0:
                        strokes = _jitter_strokes(strokes, spec["jitter"], rng)
                    single = render_strokes(strokes, size, spec["thickness"], box=boxes[i])
                    obj_sketches[i] = single
                    canvas = np.minimum(canvas, single)
                masks, tight = [], []
                for i in range(n):
                    m = mask_from_strokes(obj_sketches[i])
                    masks.append(m)
                    tight.append(bbox_of(to_binary(obj_sketches[i])))
                phrases = []
                for c, dsrc in zip(classes, dup_of):
                    if dsrc >= 0:
                        phrases.append(phrases[dsrc])
                    elif c in NO_ATTR or rng.random() < 0.25:
                        phrases.append(c)
                    else:
                        pool_a = CLASS_ATTR.get(c, ANIMAL_ATTR if c in ANIMALS else OBJECT_ATTR)
                        phrases.append(f"{rng.choice(pool_a)} {c}")
                rels = infer_relations(tight, size)
                scene = dict(
                    sid=sid, split="quickdraw", n_obj=n, count_bin=cnt_key, complexity=cx, bg=bg,
                    caption=make_caption(phrases, bg),
                    objects=[dict(cls=c, phrase=p, box=list(map(float, b)), qd_key=d["key_id"],
                                  recognized=bool(d["recognized"]), n_strokes=d["n_strokes"])
                             for c, p, b, d in zip(classes, phrases, tight, drawings)],
                    relations=[list(r) for r in rels],
                    overlap_max=float(max([box_iou(a, b) for ia, a in enumerate(tight) for b in tight[ia + 1:]] or [0])),
                )
                _save_scene(sdir, canvas, obj_sketches, masks, scene)
                made.append(sdir)
                k += 1
    print(f"QuickDraw-Scenes: {len(made)} cảnh tại {out_dir}")
    return made


# ============================================================================ COCO-Sketch
COCO_URLS = dict(
    ann="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    val="http://images.cocodataset.org/zips/val2017.zip",
)

PREP_RE = re.compile(r"\b(in|on|at|near|by|along|under|inside|across|through|beside|next to|in front of)\s+"
                     r"(the|a|an|some)\s+[a-z ,'-]+$")


def background_from_caption(caption: str) -> str:
    """Tách cụm từ nền từ câu mô tả COCO (cụm giới từ cuối câu), vd. '... on the beach' -> 'on the beach'."""
    c = caption.lower().strip().rstrip(".")
    m = None
    for mm in PREP_RE.finditer(c):
        m = mm
    if m is None:
        # thử tìm cụm giới từ bất kỳ gần cuối
        idx = max((c.rfind(f" {p} ") for p in ["in", "on", "at", "near", "by"]), default=-1)
        if idx > len(c) * 0.3:
            return c[idx + 1:]
        return "in a natural scene"
    return m.group(0)


def _pidinet():
    try:
        from controlnet_aux import PidiNetDetector
        return PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    except Exception as e:  # pragma: no cover
        print("Không nạp được PiDiNet, dùng Canny thay thế:", e)
        return None


def image_to_sketch(img_rgb: np.ndarray, detector=None) -> np.ndarray:
    """Ảnh thật -> phác thảo dạng scribble (nét đen nền trắng). Ưu tiên PiDiNet(scribble=True, safe=True)."""
    if detector is not None:
        from PIL import Image
        out = detector(Image.fromarray(img_rgb), detect_resolution=512, image_resolution=img_rgb.shape[0],
                       safe=True, scribble=True)
        edge = np.array(out.convert("L").resize((img_rgb.shape[1], img_rgb.shape[0])))
        ink = edge > 127
    else:
        g = cv2.GaussianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        ink = cv2.Canny(g, 80, 180) > 0
    return np.where(ink, 0, 255).astype(np.uint8)


def build_coco_sketch(coco_dir: str, out_dir: str, n_scenes: int = 200, seed: int = 2026, size: int = 512,
                      min_area_frac: float = 0.015, max_obj: int = 8, ref_dir: Optional[str] = None,
                      n_ref: int = 2000) -> List[str]:
    """Chọn ảnh COCO val2017 có 1..8 đối tượng nổi bật (diện tích >= 1,5% ảnh), không có đối tượng nổi bật
    nào khác bị bỏ sót; sinh phác thảo từ ảnh và mặt nạ phân đoạn thật. Phân bố đều theo số đối tượng.

    Đồng thời lưu `n_ref` ảnh thật 512x512 vào ref_dir làm tập tham chiếu FID/KID chung.
    """
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    rng = random.Random(seed)
    coco = COCO(os.path.join(coco_dir, "annotations", "instances_val2017.json"))
    caps = COCO(os.path.join(coco_dir, "annotations", "captions_val2017.json"))
    img_dir = os.path.join(coco_dir, "val2017")
    cat_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    buckets: Dict[str, List[Tuple[int, list]]] = {"1": [], "2-3": [], "4-5": [], "6-8": []}
    all_ids = sorted(coco.getImgIds())
    rng.shuffle(all_ids)
    for iid in all_ids:
        info = coco.loadImgs(iid)[0]
        H, W = info["height"], info["width"]
        s = min(H, W)
        ox, oy = (W - s) / 2, (H - s) / 2
        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=None))
        if any(a.get("iscrowd", 0) for a in anns):
            continue
        salient, ok = [], True
        for a in anns:
            if a["area"] < min_area_frac * H * W:
                continue
            x, y, w, h = a["bbox"]
            inter = max(0, min(x + w, ox + s) - max(x, ox)) * max(0, min(y + h, oy + s) - max(y, oy))
            if inter < 0.8 * w * h:
                ok = False  # đối tượng nổi bật bị cắt bởi crop vuông
                break
            salient.append(a)
        if not ok or not (1 <= len(salient) <= max_obj):
            continue
        n = len(salient)
        key = "1" if n == 1 else "2-3" if n <= 3 else "4-5" if n <= 5 else "6-8"
        buckets[key].append((iid, salient))
    per_bucket = max(1, n_scenes // len(buckets))
    det = _pidinet()
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for key, items in buckets.items():
        for iid, anns in items[:per_bucket]:
            sid = f"coco_{iid:012d}"
            sdir = os.path.join(out_dir, sid)
            if os.path.exists(os.path.join(sdir, "scene.json")):
                made.append(sdir)
                continue
            info = coco.loadImgs(iid)[0]
            img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, info["file_name"])), cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            s = min(H, W)
            ox, oy = (W - s) // 2, (H - s) // 2
            crop = cv2.resize(img[oy:oy + s, ox:ox + s], (size, size), interpolation=cv2.INTER_AREA)
            sk_full = image_to_sketch(crop, det)
            masks, obj_sk, objs = [], [], []
            union = np.zeros((size, size), bool)
            for a in sorted(anns, key=lambda a: a["bbox"][1] + a["bbox"][3]):
                rle = coco.annToRLE(a)
                m = mask_utils.decode(rle)[oy:oy + s, ox:ox + s]
                m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 0
                md = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                sk_i = np.where(md & (sk_full < 128), 0, 255).astype(np.uint8)
                if (sk_i < 128).sum() < 30:  # đối tượng quá ít nét -> vẽ đường bao
                    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(sk_i, cnts, -1, 0, 2)
                masks.append(md)
                obj_sk.append(sk_i)
                union |= md
                cls = cat_name[a["category_id"]]
                objs.append(dict(cls=cls, phrase=cls, box=list(map(float, bbox_of(m))), coco_ann=a["id"]))
            sketch = np.full((size, size), 255, np.uint8)
            for sk_i in obj_sk:
                sketch = np.minimum(sketch, sk_i)
            sketch = normalize_sketch(sketch, size)
            obj_sk = [np.where((sk < 128) & (sketch < 128), 0, 255).astype(np.uint8) for sk in obj_sk]
            cap = caps.loadAnns(caps.getAnnIds(imgIds=iid))[0]["caption"].strip()
            scene = dict(sid=sid, split="coco", n_obj=len(objs), count_bin=key, complexity="real",
                         bg=background_from_caption(cap), caption=cap, objects=objs,
                         relations=[list(r) for r in infer_relations([o["box"] for o in objs], size)],
                         coco_image_id=iid)
            _save_scene(sdir, sketch, obj_sk, masks, scene, real=crop)
            made.append(sdir)
    if ref_dir:
        os.makedirs(ref_dir, exist_ok=True)
        have = len(glob.glob(os.path.join(ref_dir, "*.jpg")))
        for iid in all_ids[:n_ref]:
            if have >= n_ref:
                break
            p = os.path.join(ref_dir, f"{iid:012d}.jpg")
            if os.path.exists(p):
                continue
            info = coco.loadImgs(iid)[0]
            img = cv2.imread(os.path.join(img_dir, info["file_name"]))
            H, W = img.shape[:2]
            s = min(H, W)
            crop = cv2.resize(img[(H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s], (size, size),
                              interpolation=cv2.INTER_AREA)
            cv2.imwrite(p, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            have += 1
    print(f"COCO-Sketch: {len(made)} cảnh tại {out_dir}")
    return made


# ============================================================================ IO
def _save_scene(sdir, sketch, obj_sketches, masks, scene: dict, real: Optional[np.ndarray] = None):
    os.makedirs(sdir, exist_ok=True)
    cv2.imwrite(os.path.join(sdir, "sketch.png"), sketch)
    for i, sk in enumerate(obj_sketches):
        cv2.imwrite(os.path.join(sdir, f"obj_{i:02d}.png"), sk)
    np.savez_compressed(os.path.join(sdir, "masks.npz"), masks=np.stack(masks).astype(bool))
    if real is not None:
        cv2.imwrite(os.path.join(sdir, "real.png"), cv2.cvtColor(real, cv2.COLOR_RGB2BGR))
    with open(os.path.join(sdir, "scene.json"), "w") as f:
        json.dump(scene, f, indent=1)


def load_scene(sdir: str) -> Scene:
    meta = json.load(open(os.path.join(sdir, "scene.json")))
    sketch = cv2.imread(os.path.join(sdir, "sketch.png"), cv2.IMREAD_GRAYSCALE)
    masks = np.load(os.path.join(sdir, "masks.npz"))["masks"]
    objs = []
    for i, o in enumerate(meta["objects"]):
        sk = cv2.imread(os.path.join(sdir, f"obj_{i:02d}.png"), cv2.IMREAD_GRAYSCALE)
        objs.append(SceneObject(cls=o["cls"], phrase=o["phrase"], box=tuple(o["box"]), mask=masks[i], sketch=sk))
    meta["dir"] = sdir
    return Scene(sid=meta["sid"], sketch=sketch, objects=objs, bg=meta["bg"], caption=meta["caption"],
                 relations=[tuple(r) for r in meta.get("relations", [])], meta=meta)


def list_scenes(bench_dir: str, split: Optional[str] = None, count_bins: Optional[Sequence[str]] = None,
                complexities: Optional[Sequence[str]] = None, limit: Optional[int] = None,
                seed: int = 0, exclude: Optional[Sequence[str]] = None) -> List[str]:
    """Liệt kê thư mục cảnh có lọc. Khi có limit, lấy mẫu phân tầng đều theo (count_bin, complexity)."""
    out = []
    for d in sorted(glob.glob(os.path.join(bench_dir, "*", "*", "scene.json"))):
        m = json.load(open(d))
        if split and m["split"] != split:
            continue
        if count_bins and m["count_bin"] not in count_bins:
            continue
        if complexities and m["complexity"] not in complexities:
            continue
        if exclude and os.path.basename(os.path.dirname(d)) in exclude:
            continue
        out.append((m["count_bin"], m["complexity"], os.path.dirname(d)))
    if limit is not None and limit < len(out):
        rng = random.Random(seed)
        cells: Dict[Tuple[str, str], List[str]] = {}
        for cb, cx, d in out:
            cells.setdefault((cb, cx), []).append(d)
        for v in cells.values():
            rng.shuffle(v)
        picked, i = [], 0
        while len(picked) < limit:
            for k in sorted(cells):
                if i < len(cells[k]) and len(picked) < limit:
                    picked.append(cells[k][i])
            i += 1
        return sorted(picked)
    return [d for _, _, d in out]
