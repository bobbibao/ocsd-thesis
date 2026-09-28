"""M1 - Tiền xử lý phác thảo, phân tách đối tượng, suy luận quan hệ và dựng câu lệnh."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
except Exception:  # pragma: no cover
    skeletonize = None

Box = Tuple[float, float, float, float]  # x0, y0, x1, y1 (pixel)


# ----------------------------------------------------------------------------- rendering
def render_strokes(strokes, size: int = 512, thickness: int = 3, box: Optional[Box] = None,
                   canvas: Optional[np.ndarray] = None, src_size: int = 256) -> np.ndarray:
    """Vẽ nét QuickDraw (định dạng simplified: [[xs],[ys]] trong khung 0..255) lên nền trắng.

    Nếu có `box`, phác thảo được co giãn (giữ tỉ lệ) để vừa khít hộp đó trong khung `size`.
    Trả về ảnh uint8 HxW, nét = 0 (đen), nền = 255.
    """
    if canvas is None:
        canvas = np.full((size, size), 255, np.uint8)
    xs = np.concatenate([np.asarray(s[0], np.float32) for s in strokes])
    ys = np.concatenate([np.asarray(s[1], np.float32) for s in strokes])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    if box is None:
        box = (0, 0, size - 1, size - 1)
    bx0, by0, bx1, by1 = box
    s = min((bx1 - bx0) / w, (by1 - by0) / h)
    ox = bx0 + ((bx1 - bx0) - w * s) / 2
    oy = by0 + ((by1 - by0) - h * s) / 2
    for st in strokes:
        pts = np.stack([(np.asarray(st[0]) - x0) * s + ox, (np.asarray(st[1]) - y0) * s + oy], 1)
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) == 1:
            cv2.circle(canvas, tuple(pts[0, 0]), max(thickness // 2, 1), 0, -1)
        else:
            cv2.polylines(canvas, [pts], False, 0, thickness, lineType=cv2.LINE_AA)
    return canvas


# ----------------------------------------------------------------------------- normalisation
def to_binary(img: np.ndarray, thr: int = 200) -> np.ndarray:
    """Ảnh phác thảo (nền trắng) -> mặt nạ nét bool (True = nét)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img < thr


def normalize_sketch(img: np.ndarray, size: int = 512, target_width: int = 3,
                     min_component: int = 12) -> np.ndarray:
    """Chuẩn hóa phác thảo: đệm về hình vuông, nhị phân hóa, làm mảnh, làm dày đều ~3px, lọc nhiễu.

    Trả về uint8 HxW, nét đen trên nền trắng.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    m = max(h, w)
    pad = np.full((m, m), 255, np.uint8)
    pad[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = img
    img = cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA)
    ink = to_binary(img)
    # lọc thành phần liên thông nhỏ
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_component:
            keep |= lab == i
    if skeletonize is not None:
        keep = skeletonize(keep)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (target_width, target_width))
    keep = cv2.dilate(keep.astype(np.uint8), k) > 0
    out = np.full((size, size), 255, np.uint8)
    out[keep] = 0
    return out


def sketch_to_control(img: np.ndarray) -> np.ndarray:
    """ControlNet Scribble v1.1 và T2I-Adapter Sketch nhận nét TRẮNG trên nền ĐEN, 3 kênh."""
    ink = to_binary(img)
    out = np.zeros(ink.shape + (3,), np.uint8)
    out[ink] = 255
    return out


# ----------------------------------------------------------------------------- masks / boxes
def mask_from_strokes(sketch_obj: np.ndarray, dilate: int = 9, mode: str = "hull") -> np.ndarray:
    """Mặt nạ vùng m_i: lấp đầy bao lồi (hoặc đường bao ngoài) của nét vẽ, rồi giãn nở nhẹ."""
    ink = to_binary(sketch_obj).astype(np.uint8)
    if ink.sum() == 0:
        return ink.astype(bool)
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(ink)
    if mode == "hull":
        pts = np.concatenate(cnts, 0)
        cv2.fillPoly(m, [cv2.convexHull(pts)], 1)
    else:
        cv2.drawContours(m, cnts, -1, 1, -1)
    if dilate > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    return m.astype(bool)


def bbox_of(mask: np.ndarray) -> Box:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def box_iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def auto_decompose(sketch: np.ndarray, dilate: int = 15, merge_gap: float = 0.02,
                   min_area: int = 200) -> List[np.ndarray]:
    """Phân tách tự động (khi không có nhãn lớp vẽ): gộp nét theo thành phần liên thông sau giãn nở,
    sau đó ghép các cụm có hộp bao gần nhau. Trả về danh sách ảnh phác thảo từng đối tượng."""
    ink = to_binary(sketch).astype(np.uint8)
    grown = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    n, lab = cv2.connectedComponents(grown, 8)
    groups = [(lab == i) & (ink > 0) for i in range(1, n)]
    groups = [g for g in groups if g.sum() > 0]
    H = sketch.shape[0]
    changed = True
    while changed and len(groups) > 1:
        changed = False
        boxes = [bbox_of(g) for g in groups]
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = boxes[i], boxes[j]
                gap_x = max(0, max(a[0], b[0]) - min(a[2], b[2]))
                gap_y = max(0, max(a[1], b[1]) - min(a[3], b[3]))
                if max(gap_x, gap_y) < merge_gap * H and (box_iou(a, b) > 0 or min(gap_x, gap_y) == 0):
                    groups[i] = groups[i] | groups[j]
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    out = []
    for g in groups:
        if mask_from_strokes(np.where(g, 0, 255).astype(np.uint8)).sum() < min_area:
            continue
        out.append(np.where(g, 0, 255).astype(np.uint8))
    return out


# ----------------------------------------------------------------------------- relations
REL_TYPES = ("left_of", "right_of", "above", "below", "in_front_of", "behind")


def infer_relations(boxes: Sequence[Box], size: int = 512, delta: float = 0.05) -> List[Tuple[int, int, str]]:
    """Quy tắc hình học (mục 3.3): trái/phải, trên/dưới theo tâm hộp (ngưỡng delta * size);
    trước/sau khi hai hộp chồng lấn, theo cạnh dưới (giả định mặt đất)."""
    rels = []
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            if i >= j:
                continue
            cax, cay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
            cbx, cby = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if cbx - cax > delta * size:
                rels.append((i, j, "left_of"))
            elif cax - cbx > delta * size:
                rels.append((i, j, "right_of"))
            if cby - cay > delta * size:
                rels.append((i, j, "above"))
            elif cay - cby > delta * size:
                rels.append((i, j, "below"))
            if box_iou(a, b) > 0.0:
                if a[3] - b[3] > delta * size / 2:
                    rels.append((i, j, "in_front_of"))
                elif b[3] - a[3] > delta * size / 2:
                    rels.append((i, j, "behind"))
    return rels


def relation_holds(a: Box, b: Box, r: str, size: int = 512, delta: float = 0.05) -> bool:
    """Kiểm tra quan hệ r giữa hai hộp (dùng cho độ đo RA). Dùng ngưỡng lỏng hơn (0) cho trái/phải/trên/dưới
    để không phạt các dịch chuyển nhỏ; trước/sau dựa trên cạnh dưới."""
    cax, cay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    cbx, cby = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    if r == "left_of":
        return cax < cbx
    if r == "right_of":
        return cax > cbx
    if r == "above":
        return cay < cby
    if r == "below":
        return cay > cby
    if r == "in_front_of":
        return a[3] > b[3]
    if r == "behind":
        return a[3] < b[3]
    raise ValueError(r)


# ----------------------------------------------------------------------------- object crops
@dataclass
class CropInfo:
    """Ánh xạ giữa khung cảnh (scene) và khung ảnh đối tượng 512x512 (crop)."""
    x0: int
    y0: int
    side: int      # cạnh của vùng vuông trong khung cảnh
    size: int      # cạnh của ảnh crop (512)

    def scene_to_crop(self, box: Box) -> Box:
        s = self.size / self.side
        return ((box[0] - self.x0) * s, (box[1] - self.y0) * s, (box[2] - self.x0) * s, (box[3] - self.y0) * s)


def square_crop(img: np.ndarray, box: Box, size: int = 512, margin: float = 0.12,
                fill: int = 255) -> Tuple[np.ndarray, CropInfo]:
    """Cắt vùng vuông bao quanh hộp (có lề), đệm nếu ra ngoài ảnh, phóng về size x size."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    side = int(round(max(x1 - x0, y1 - y0) * (1 + 2 * margin)))
    side = max(side, 16)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx0, sy0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    shape = (side, side) + img.shape[2:]
    out = np.full(shape, fill, img.dtype)
    ix0, iy0 = max(sx0, 0), max(sy0, 0)
    ix1, iy1 = min(sx0 + side, W), min(sy0 + side, H)
    if ix1 > ix0 and iy1 > iy0:
        out[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = img[iy0:iy1, ix0:ix1]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    out = cv2.resize(out, (size, size), interpolation=interp)
    return out, CropInfo(sx0, sy0, side, size)


def paste_back(canvas: np.ndarray, canvas_mask: np.ndarray, obj_img: np.ndarray, obj_mask: np.ndarray,
               info: CropInfo, scale: float = 1.0, dx: int = 0, dy: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Đặt ảnh đối tượng (khung crop) trở lại khung cảnh theo CropInfo (có thể co giãn / dịch nhẹ).
    Trả về (ảnh cảnh mới, mặt nạ của đối tượng trong khung cảnh)."""
    H, W = canvas.shape[:2]
    side = max(int(round(info.side * scale)), 4)
    img = cv2.resize(obj_img, (side, side), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(obj_mask.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST) > 0
    cx, cy = info.x0 + info.side / 2 + dx, info.y0 + info.side / 2 + dy
    sx0, sy0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    ix0, iy0 = max(sx0, 0), max(sy0, 0)
    ix1, iy1 = min(sx0 + side, W), min(sy0 + side, H)
    placed = np.zeros((H, W), bool)
    if ix1 <= ix0 or iy1 <= iy0:
        return canvas, placed
    sub_m = msk[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0]
    sub_i = img[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0]
    region = canvas[iy0:iy1, ix0:ix1]
    region[sub_m] = sub_i[sub_m]
    placed[iy0:iy1, ix0:ix1] = sub_m
    return canvas, placed


# ----------------------------------------------------------------------------- scene structure
@dataclass
class SceneObject:
    cls: str                      # tên lớp, ví dụ "dog"
    phrase: str                   # cụm từ người dùng, ví dụ "brown dog"
    box: Box
    mask: np.ndarray              # m_i (bool HxW)
    sketch: np.ndarray            # phác thảo riêng của đối tượng (uint8 HxW, nền trắng)


@dataclass
class Scene:
    sid: str
    sketch: np.ndarray            # phác thảo cảnh đã chuẩn hóa
    objects: List[SceneObject]
    bg: str                       # cụm từ nền p_bg
    caption: str                  # câu mô tả tự nhiên đầy đủ (dùng cho baseline và CLIP score)
    relations: List[Tuple[int, int, str]] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.objects)

    def class_counts(self) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for o in self.objects:
            d[o.cls] = d.get(o.cls, 0) + 1
        return d


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def join_phrases(phrases: Sequence[str]) -> str:
    items = [f"{article(p)} {p}" for p in phrases]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


NUM_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
IRREGULAR_PLURAL = {"sheep": "sheep", "mouse": "mice", "bus": "buses", "person": "people", "fish": "fish",
                    "knife": "knives", "giraffe": "giraffes", "wine glass": "wine glasses", "bench": "benches",
                    "couch": "couches", "sandwich": "sandwiches", "skis": "skis", "scissors": "scissors",
                    "traffic light": "traffic lights", "fire hydrant": "fire hydrants", "cow": "cows",
                    "teddy bear": "teddy bears", "motorcycle": "motorcycles", "sailboat": "sailboats",
                    "motorbike": "motorbikes", "bird": "birds", "duck": "ducks", "tree": "trees",
                    "horse": "horses", "dog": "dogs", "cat": "cats", "car": "cars", "truck": "trucks"}


def plural(phrase: str) -> str:
    words = phrase.split()
    for k in sorted(IRREGULAR_PLURAL, key=len, reverse=True):
        if phrase.endswith(k):
            return phrase[: -len(k)] + IRREGULAR_PLURAL[k]
    last = words[-1]
    if last.endswith(("s", "x", "ch", "sh")):
        last += "es"
    elif last.endswith("y") and last[-2:-1] not in "aeiou":
        last = last[:-1] + "ies"
    else:
        last += "s"
    return " ".join(words[:-1] + [last])


def make_caption(phrases: Sequence[str], bg: str) -> str:
    """Câu mô tả tự nhiên, gộp các cụm từ trùng nhau thành số đếm: 'a brown horse and two white sheep ...'.
    Số đếm tường minh giúp baseline có đủ thông tin số lượng (so sánh công bằng)."""
    order, cnt = [], {}
    for p in phrases:
        if p not in cnt:
            order.append(p)
            cnt[p] = 0
        cnt[p] += 1
    items = [f"{article(p)} {p}" if cnt[p] == 1 else f"{NUM_WORDS.get(cnt[p], str(cnt[p]))} {plural(p)}"
             for p in order]
    s = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    return f"{s} {bg}".strip()


# ----------------------------------------------------------------------------- prompt with token groups
@dataclass
class GroupedPrompt:
    """Câu lệnh kèm chỉ số token của từng nhóm: 'obj{i}' (T_i), 'id{i}' (token định danh), 'bg' (T_bg)."""
    text: str
    groups: Dict[str, List[int]]


def build_grouped_prompt(tokenizer, pieces: Sequence[Tuple[str, Optional[str]]], max_len: int = 77) -> GroupedPrompt:
    """Ghép các mảnh văn bản, theo dõi chỉ số token của từng mảnh (vị trí 0 là <|startoftext|>).

    Tokenizer CLIP tách theo khoảng trắng nên token của chuỗi ghép = ghép token của từng mảnh.
    Nếu vượt 77 token, các mảnh cuối bị cắt (nhóm của chúng bị rỗng).
    """
    groups: Dict[str, List[int]] = {}
    text_parts = []
    pos = 1
    for text, g in pieces:
        text = text.strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        idx = [p for p in range(pos, pos + len(ids)) if p < max_len - 1]
        if g is not None:
            for gg in g.split("|"):
                groups.setdefault(gg, []).extend(idx)
        pos += len(ids)
        text_parts.append(text)
    return GroupedPrompt(" ".join(text_parts), groups)


def scene_global_prompt(tokenizer, scene: Scene, id_tokens: Optional[Sequence[str]] = None,
                        use_phrase: bool = True, prefix: str = "a photo of") -> GroupedPrompt:
    """P_g = 'a photo of <o0> brown dog, <o1> red car and ... on the beach'.

    Nhóm 'obj{i}' gồm token định danh + thuộc tính + tên lớp của đối tượng i, nhóm 'id{i}' chỉ gồm token
    định danh (hoặc token tên lớp khi không có định danh) - dùng cho năng lượng chú ý ở M5(b)."""
    pieces: List[Tuple[str, Optional[str]]] = [(prefix, None)]
    n = scene.n
    for i, o in enumerate(scene.objects):
        if i > 0:
            pieces.append(("and" if i == n - 1 else ",", None))
        words = o.phrase if use_phrase else o.cls
        if id_tokens is not None:
            pieces.append((id_tokens[i], f"obj{i}|id{i}"))
            pieces.append((words, f"obj{i}"))
        else:
            attr = words[: -len(o.cls)].strip() if words.endswith(o.cls) else ""
            if attr:
                pieces.append((attr, f"obj{i}"))
            pieces.append((o.cls, f"obj{i}|id{i}"))
    pieces.append((scene.bg, "bg"))
    return build_grouped_prompt(tokenizer, pieces)
