"""Ghép cặp phát hiện - đối tượng phác thảo (Hungarian theo IoU, cùng lớp) và các độ đo nhất quán:
OPR, OCE, độ chính xác đếm, mIoU bố cục, độ chính xác quan hệ (RA). Dùng chung cho M5(d) và đánh giá."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .sketch import Scene, box_iou, relation_holds


def match_objects(gt: Sequence[Tuple[str, tuple]], dets: Sequence[dict], iou_thr: float = 0.1):
    """gt: [(cls, box)], dets: [{'cls','box','score'}]. Trả về list (det_index | -1, iou) cho mỗi gt."""
    res = [(-1, 0.0)] * len(gt)
    for c in sorted({g[0] for g in gt}):
        gi = [i for i, g in enumerate(gt) if g[0] == c]
        di = [j for j, d in enumerate(dets) if d["cls"] == c]
        if not di:
            continue
        iou = np.array([[box_iou(gt[i][1], dets[j]["box"]) for j in di] for i in gi])
        r, cidx = linear_sum_assignment(-iou)
        for a, b in zip(r, cidx):
            if iou[a, b] >= iou_thr:
                res[gi[a]] = (di[b], float(iou[a, b]))
    return res


def consistency(scene: Scene, dets: Sequence[dict], iou_thr: float = 0.1, size: int = 512) -> Dict:
    gt = [(o.cls, o.box) for o in scene.objects]
    m = match_objects(gt, dets, iou_thr)
    n_in = len(gt)
    preserved = sum(1 for j, _ in m if j >= 0)
    counts_in = scene.class_counts()
    counts_gen = {c: sum(1 for d in dets if d["cls"] == c) for c in counts_in}
    n_gen = sum(counts_gen.values())
    oce = abs(n_gen - n_in)
    oce_c = sum(abs(counts_gen[c] - counts_in[c]) for c in counts_in)
    miou = float(np.mean([iou if j >= 0 else 0.0 for j, iou in m])) if n_in else 0.0
    rel_ok = []
    for i, j, r in scene.relations:
        di, dj = m[i][0], m[j][0]
        rel_ok.append(di >= 0 and dj >= 0 and relation_holds(dets[di]["box"], dets[dj]["box"], r, size))
    ra = float(np.mean(rel_ok)) if rel_ok else float("nan")
    per_obj_missing = [scene.objects[i].cls for i, (j, _) in enumerate(m) if j < 0]
    return dict(opr=preserved / max(n_in, 1), oce=oce, oce_c=oce_c, count_acc=float(oce_c == 0), miou=miou,
                ra=ra, n_in=n_in, n_gen=n_gen, n_preserved=preserved, missing=per_obj_missing,
                matched=[(j, iou) for j, iou in m])


def verify_score(c: Dict) -> float:
    """Điểm nhất quán tổng hợp dùng trong M5(d) để chọn ảnh tốt nhất giữa các lần sinh lại."""
    ra = 1.0 if np.isnan(c["ra"]) else c["ra"]
    return c["opr"] + 0.5 * c["miou"] + 0.25 * ra - 0.25 * c["oce_c"] / max(c["n_in"], 1)


def verify_pass(c: Dict) -> bool:
    ra = 1.0 if np.isnan(c["ra"]) else c["ra"]
    return c["opr"] == 1.0 and c["oce_c"] == 0 and ra == 1.0
