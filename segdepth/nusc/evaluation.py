"""
evaluation.py
-------------
Scoring radar-segment association against nuScenes 3D box annotations, and
diagnosing why points are mis-assigned.

Ground truth per radar point (in the camera-time ego frame):
  * segment column j  inside a box (inflated by pos_margin) whose projected
                      convex hull matches a compatible segment
  * GT_UNMATCHED      inside a box, but no compatible segment was found
                      (segmentation miss or no equivalent Cityscapes class)
  * GT_BACKGROUND     outside every box inflated by neg_margin; correct answer
                      is clutter, occluded or a non-object segment
  * GT_IGNORE         between pos_margin and neg_margin of a box (ambiguous)

Box-to-segment matching: the 3D box is projected, its convex hull rasterised,
and compatible segments scored by overlap. Thing segments must lie mostly
inside the hull; stuff segments (e.g. fence for barriers) must cover part of it.
Things are matched one-to-one, nearest boxes first.

Functions:
  * evaluate_with_annotations  Per-frame metrics and raw counts for three
                               methods (naive, unary-only, optimised).
  * diagnose_failures          Classify each labelled point: correct,
                               unreachable, depth_mismatch, lost_to_segment,
                               lost_to_null.
  * summarise_counts           Turn summed counts into metrics.
"""

from collections import Counter

import numpy as np
from nuscenes.utils.geometry_utils import view_points
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

from ..config import EXTENT

GT_BACKGROUND, GT_UNMATCHED, GT_IGNORE = -1, -2, -3
METHODS = ("naive_zero_elevation", "unary_only", "optimised")

# nuScenes category prefix -> Cityscapes labels able to represent it
NUSC_TO_CITYSCAPES = {
    "vehicle.car": {"car"},
    "vehicle.truck": {"truck", "car"},
    "vehicle.trailer": {"truck", "car"},
    "vehicle.construction": {"truck", "car"},
    "vehicle.bus": {"bus", "truck"},
    "vehicle.emergency": {"car", "truck"},
    "vehicle.motorcycle": {"motorcycle", "rider"},
    "vehicle.bicycle": {"bicycle", "rider"},
    "human.pedestrian": {"person", "rider"},
    "movable_object.barrier": {"fence", "wall"},
    "movable_object.trafficcone": set(),
    "movable_object.pushable_pullable": set(),
    "movable_object.debris": set(),
    "static_object.bicycle_rack": {"bicycle", "fence"},
    "animal": set(),
}


def compatible_labels(name):
    best = ""
    for prefix in NUSC_TO_CITYSCAPES:
        if name.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return NUSC_TO_CITYSCAPES.get(best, set())


def _convex_hull(pts):
    """Andrew's monotone chain. pts: (n, 2). Returns hull vertices in order."""
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _hull_mask(corners_cam, K, H, W):
    c = corners_cam[:, corners_cam[2] > 0.5]       # corners behind the camera are dropped
    if c.shape[1] < 3:
        return None
    uv = view_points(c, K, normalize=True)[:2].T
    hull = _convex_hull(uv)
    if len(hull) < 3:
        return None
    img = Image.new("1", (W, H), 0)
    ImageDraw.Draw(img).polygon([(float(x), float(y)) for x, y in hull], fill=1)
    m = np.asarray(img, dtype=bool)
    return m if m.any() else None


def _in_footprint(xy, box, margin):
    yaw = box.orientation.yaw_pitch_roll[0]
    q = xy - box.center[:2]
    lx = np.cos(yaw) * q[:, 0] + np.sin(yaw) * q[:, 1]
    ly = -np.sin(yaw) * q[:, 0] + np.cos(yaw) * q[:, 1]
    w, l, _ = box.wlh
    return (np.abs(lx) <= l / 2 + margin) & (np.abs(ly) <= w / 2 + margin)


def _naive_assignment(assoc):
    """Zero-elevation pixel lookup on the original segmentation map."""
    geo0, sc = assoc._geometry({}), assoc.center
    seg = assoc.seg_map
    lut = np.full(max(int(seg.max()), int(assoc.ids.max(initial=0))) + 1, -1)
    lut[assoc.ids] = np.arange(assoc.M)
    sid = seg[geo0["v"][:, sc], geo0["u"][:, sc]]
    col = np.where(sid >= 0, lut[np.clip(sid, 0, None)], -1)
    ok = geo0["valid"][:, sc] & (col >= 0) & np.isin(col, assoc.cand_idx)
    return np.where(ok, col, assoc.CLUTTER)


def evaluate_with_annotations(nusc, sample_token, cam_token, assoc,
                              pos_margin=1.0, neg_margin=2.5,
                              min_seg_frac=0.5, min_hull_frac=0.3):
    """Returns (report, gt). `report["counts"]` can be summed across frames."""
    cam_rec = nusc.get("sample_data", cam_token)
    c_cal = nusc.get("calibrated_sensor", cam_rec["calibrated_sensor_token"])
    c_ego = nusc.get("ego_pose", cam_rec["ego_pose_token"])
    H, W = assoc.seg_map.shape
    xy = assoc._geometry({})["xy_ego"]     # uncorrected geometry: ground truth must not depend on fitted yaw
    N = len(assoc.pts)

    masks = {j: assoc.seg_map == assoc.ids[j] for j in assoc.cand_idx}
    sizes = {j: int(m.sum()) for j, m in masks.items()}

    boxes, pairs = [], []
    for ann in nusc.get("sample", sample_token)["anns"]:
        box = nusc.get_box(ann)                                   # global frame
        box.translate(-np.array(c_ego["translation"]))
        box.rotate(Quaternion(c_ego["rotation"]).inverse)         # camera-time ego frame
        in_neg = _in_footprint(xy, box, neg_margin)
        if not in_neg.any():
            continue
        k = len(boxes)
        boxes.append(dict(dist=float(np.linalg.norm(box.center[:2])), name=box.name,
                          in_pos=_in_footprint(xy, box, pos_margin), in_neg=in_neg, seg=-1))

        labels = compatible_labels(box.name)
        if not labels:
            continue
        cbox = box.copy()
        cbox.translate(-np.array(c_cal["translation"]))
        cbox.rotate(Quaternion(c_cal["rotation"]).inverse)        # camera frame
        hull = _hull_mask(cbox.corners(), assoc.cam.K, H, W)
        if hull is None:
            continue
        hull_area = int(hull.sum())
        for j in assoc.cand_idx:
            lab = assoc.labels[j]
            if lab not in labels:
                continue
            inter = int((masks[j] & hull).sum())
            if inter == 0:
                continue
            ok = (inter / sizes[j] >= min_seg_frac if lab in EXTENT
                  else inter / hull_area >= min_hull_frac)
            if ok:
                pairs.append((inter, k, j))

    # Greedy matching: largest overlap first; things one-to-one, stuff shared
    taken = set()
    for inter, k, j in sorted(pairs, reverse=True):
        if boxes[k]["seg"] >= 0 or (assoc.labels[j] in EXTENT and j in taken):
            continue
        boxes[k]["seg"] = j
        taken.add(j)

    gt = np.full(N, GT_BACKGROUND)
    for b in boxes:
        gt[b["in_neg"]] = GT_IGNORE
    for b in sorted(boxes, key=lambda b: -b["dist"]):             # nearer boxes win
        gt[b["in_pos"]] = b["seg"] if b["seg"] >= 0 else GT_UNMATCHED

    obj_cols = {j for j in assoc.cand_idx if assoc.labels[j] in EXTENT}
    obj_cols |= {b["seg"] for b in boxes if b["seg"] >= 0}
    obj_cols = np.array(sorted(obj_cols), dtype=int)

    def counts(A):
        pos = gt >= 0
        obj = np.isin(A, obj_cols)
        scored = (gt != GT_IGNORE) & (gt != GT_UNMATCHED)
        return {
            "n_pos": int(pos.sum()),
            "tp": int(np.sum(A[pos] == gt[pos])),
            "n_obj_assigned": int(np.sum(obj & scored)),
            "n_obj_correct": int(np.sum(obj & pos & (A == gt))),
            "n_bg": int(np.sum(gt == GT_BACKGROUND)),
            "bg_to_obj": int(np.sum(obj & (gt == GT_BACKGROUND))),
            "n_unmatched": int(np.sum(gt == GT_UNMATCHED)),
        }

    flat = {}
    for name, A in zip(METHODS, (_naive_assignment(assoc), assoc.A_init, assoc.A)):
        for key, val in counts(A).items():
            flat[f"{name}/{key}"] = val

    report = {
        "boxes_near_radar": len(boxes),
        "boxes_matched": int(sum(b["seg"] >= 0 for b in boxes)),
        "counts": flat,
        "metrics": summarise_counts(flat),
    }

    return report, gt


def summarise_counts(counts):
    def ratio(a, b):
        return round(a / b, 3) if b else None

    out = {}
    for m in METHODS:
        c = {k.split("/", 1)[1]: v for k, v in counts.items() if k.startswith(m + "/")}
        if not c:
            continue
        out[m] = {
            "recall": ratio(c["tp"], c["n_pos"]),
            "precision": ratio(c["n_obj_correct"], c["n_obj_assigned"]),
            "false_object_rate": ratio(c["bg_to_obj"], c["n_bg"]),
            "n_pos": c["n_pos"],
            "n_unmatched": c["n_unmatched"],
        }
    return out


def diagnose_failures(assoc, gt, depth_k=3.0):
    """
    For every point with a matched segment label, report why it was or was not
    assigned correctly:
      correct          assigned to the labelled segment
      unreachable      no elevation/azimuth sample lands on the labelled segment
                       (azimuth spread, mask boundary, calibration)
      depth_mismatch   samples land on it, but no depth within depth_k sigma
                       (depth scale or depth-map error)
      lost_to_segment  reachable and depth-consistent, another segment won
      lost_to_null     reachable and depth-consistent, clutter/occluded won
    `median_depth_ratio` (map depth / radar depth on the labelled segment) far
    from 1 indicates a depth-scale bias.
    """
    cfg, p, geo, extra = assoc.cfg, assoc.params, assoc.geo, assoc.extra
    col = np.where(geo["valid"], assoc.col_map[geo["v"], geo["u"]], -1)
    Z, D = geo["Z"], extra["D"]
    sZ = np.sqrt(cfg.sigma_range ** 2 + (Z ** 2 * p.sigma_inv) ** 2)

    cats, ratios, rows = Counter(), [], []
    for i in np.flatnonzero(gt >= 0):
        j, a = int(gt[i]), int(assoc.A[i])
        on = col[i] == j
        fin = on & np.isfinite(D[i])
        ratio = float(np.median(D[i][fin] / Z[i][fin])) if fin.any() else None
        if ratio is not None:
            ratios.append(ratio)

        near_miss = None
        if a == j:
            cat = "correct"
        elif not on.any():
            cat = "unreachable"
            mask = assoc.seg_map == assoc.ids[j]
            ok = geo["valid"][i]
            dists = [np.abs(np.flatnonzero(mask[vv]) - uu).min()
                     for uu, vv in zip(geo["u"][i][ok], geo["v"][i][ok]) if mask[vv].any()]
            near_miss = int(min(dists)) if dists else None
        elif not fin.any() or (np.abs(D[i][fin] - Z[i][fin]) / sZ[i][fin]).min() > depth_k:
            cat = "depth_mismatch"
        elif a < assoc.M:
            cat = "lost_to_segment"
        else:
            cat = "lost_to_null"
        cats[cat] += 1
        rows.append({
            "point": int(i),
            "range_m": round(float(np.linalg.norm(assoc.pts.xy[i])), 2),
            "gt_label": assoc.labels[j],
            "assigned": (assoc.labels[a] if a < assoc.M
                         else "clutter" if a == assoc.CLUTTER else "occluded"),
            "category": cat,
            "depth_ratio": None if ratio is None else round(ratio, 3),
            "near_miss_px": near_miss,
        })

    return {
        "counts": dict(cats),
        "median_depth_ratio": round(float(np.median(ratios)), 3) if ratios else None,
        "points": rows,
    }
