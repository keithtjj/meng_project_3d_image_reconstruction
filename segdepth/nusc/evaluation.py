"""
evaluation.py
-------------
Scoring radar-segment association against nuScenes 3D box annotations.

Ground truth
  * A radar point belongs to the annotated box whose bird's-eye-view footprint
    (inflated by `margin` m) contains it; the nearest box wins ties.
  * Each box is projected into the image as the convex hull of its 8 corners.
    It is matched to the 'thing' segment with the highest containment
    (fraction of the segment's pixels inside the hull), subject to a depth
    check, with one-to-one greedy matching.
  * Points on boxes with no matching segment (segmentation misses) are
    excluded from both recall and precision and counted separately.

Metrics (raw counts are returned so results can be pooled across frames)
  * recall     correct / points whose box has a matched segment
  * precision  correct / points assigned to a thing segment, excluding points
               on unmatched boxes
  * outcomes   where ground-truth points were assigned: correct, wrong_thing,
               stuff, clutter, occluded
  * causes     why misses happened: unreachable (no valid elevation/azimuth
               sample lands on the true segment), depth_mismatch (reachable,
               but the depth map disagrees with the radar range by > k sigma),
               outcompeted (reachable and depth-consistent, but another
               hypothesis scored better)

Methods scored: naive zero-elevation lookup, unary-only, optimised.
"""

import numpy as np
from nuscenes.utils.geometry_utils import view_points
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

from ..config import EXTENT
from ..depth_scale import closeness_to_metric

OUTCOMES = ("correct", "wrong_thing", "stuff", "clutter", "occluded")
CAUSES = ("unreachable", "depth_mismatch", "outcompeted")


def _convex_hull(p):
    """Monotone chain convex hull of (n, 2) points; returns (m, 2)."""
    p = np.unique(p, axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(points):
        h = []
        for q in points:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h

    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _hull_mask(uv, H, W):
    hull = _convex_hull(uv.T)
    img = Image.new("1", (W, H), 0)
    if len(hull) >= 3:
        ImageDraw.Draw(img).polygon([tuple(map(float, q)) for q in hull], fill=1)
    return np.asarray(img, dtype=bool)


def _ground_truth(nusc, sample_token, cam_token, assoc, margin, min_containment,
                  depth_rel_tol, depth_abs_tol):
    """Returns (gt_seg per point: segment column, -1 no box, -2 unmatched box),
    and box statistics."""
    cam_rec = nusc.get("sample_data", cam_token)
    c_cal = nusc.get("calibrated_sensor", cam_rec["calibrated_sensor_token"])
    c_ego = nusc.get("ego_pose", cam_rec["ego_pose_token"])
    H, W = assoc.seg_map.shape
    xy = assoc.geo["xy_ego"]
    N = len(assoc.pts)

    thing_cols = [j for j, l in enumerate(assoc.labels) if l in EXTENT]
    masks = {j: assoc.seg_map == assoc.ids[j] for j in thing_cols}
    sizes = {j: max(int(m.sum()), 1) for j, m in masks.items()}
    metric = closeness_to_metric(assoc.closeness, assoc.params.a, assoc.params.b,
                                 assoc.cfg.max_range)
    seg_depth = {}
    for j in thing_cols:
        d = metric[masks[j]]
        d = d[np.isfinite(d)]
        seg_depth[j] = float(np.median(d)) if len(d) else np.inf

    boxes = []              # (points inside, distance to point set, candidate scores)
    for ann in nusc.get("sample", sample_token)["anns"]:
        box = nusc.get_box(ann)
        box.translate(-np.array(c_ego["translation"]))
        box.rotate(Quaternion(c_ego["rotation"]).inverse)          # camera-time ego frame

        yaw = box.orientation.yaw_pitch_roll[0]
        q = xy - box.center[:2]
        lx = np.cos(yaw) * q[:, 0] + np.sin(yaw) * q[:, 1]
        ly = -np.sin(yaw) * q[:, 0] + np.cos(yaw) * q[:, 1]
        w, l, _ = box.wlh
        inside = (np.abs(lx) <= l / 2 + margin) & (np.abs(ly) <= w / 2 + margin)
        if not inside.any():
            continue
        dist = np.hypot(lx, ly)

        cbox = box.copy()
        cbox.translate(-np.array(c_cal["translation"]))
        cbox.rotate(Quaternion(c_cal["rotation"]).inverse)          # camera frame
        corners = cbox.corners()
        corners = corners[:, corners[2] > 0.5]
        scores = {}
        if corners.shape[1] >= 3:
            hull = _hull_mask(view_points(corners, assoc.cam.K, normalize=True)[:2], H, W)
            box_depth = float(cbox.center[2])
            tol = max(depth_abs_tol, depth_rel_tol * box_depth)
            for j in thing_cols:
                cont = float((masks[j] & hull).sum()) / sizes[j]
                if cont >= min_containment and abs(seg_depth[j] - box_depth) <= tol:
                    scores[j] = cont
        boxes.append((inside, dist, scores))

    # One-to-one greedy matching by containment
    pairs = sorted(((s, b, j) for b, (_, _, sc) in enumerate(boxes) for j, s in sc.items()),
                   reverse=True)
    box_to_seg, used = {}, set()
    for _, b, j in pairs:
        if b not in box_to_seg and j not in used:
            box_to_seg[b] = j
            used.add(j)

    # Assign each point to its nearest containing box
    gt = np.full(N, -1)
    best_dist = np.full(N, np.inf)
    for b, (inside, dist, _) in enumerate(boxes):
        closer = inside & (dist < best_dist)
        gt[closer] = box_to_seg.get(b, -2)
        best_dist[closer] = dist[closer]

    stats = {"boxes_with_points": len(boxes), "boxes_matched": len(box_to_seg)}
    return gt, thing_cols, stats


def _miss_cause(assoc, i, j, k_sigma):
    geo, extra, p, cfg = assoc.geo, assoc.extra, assoc.params, assoc.cfg
    on_seg = geo["valid"][i] & (assoc.col_map[geo["v"][i], geo["u"][i]] == j)
    if not on_seg.any():
        return "unreachable"
    Z, D = extra["Z"][i, on_seg], extra["D"][i, on_seg]
    sZ = np.sqrt(cfg.sigma_range ** 2 + (Z ** 2 * p.sigma_inv) ** 2)
    with np.errstate(invalid="ignore"):
        res = np.where(np.isfinite(D), np.abs(D - Z) / sZ, np.inf)
    return "depth_mismatch" if res.min() > k_sigma else "outcompeted"


def _score(A, gt, thing_cols, M, clutter, occluded):
    has_gt = gt >= 0
    to_thing = np.isin(A, thing_cols)
    verifiable = to_thing & (gt != -2)

    outcomes = dict.fromkeys(OUTCOMES, 0)
    for a, g in zip(A[has_gt], gt[has_gt]):
        if a == g:
            outcomes["correct"] += 1
        elif a in thing_cols:
            outcomes["wrong_thing"] += 1
        elif a < M:
            outcomes["stuff"] += 1
        elif a == clutter:
            outcomes["clutter"] += 1
        else:
            outcomes["occluded"] += 1

    n_gt, n_pred = int(has_gt.sum()), int(verifiable.sum())
    n_tp = int(np.sum(A[verifiable] == gt[verifiable]))
    return {
        "n_gt": n_gt, "n_pred": n_pred, "n_tp": n_tp,
        "recall": round(n_tp / n_gt, 3) if n_gt else None,
        "precision": round(n_tp / n_pred, 3) if n_pred else None,
        "outcomes": outcomes,
    }


def evaluate_with_annotations(nusc, sample_token, cam_token, assoc, margin=1.0,
                              min_containment=0.5, depth_rel_tol=0.3,
                              depth_abs_tol=3.0, k_sigma=3.0):
    gt, thing_cols, box_stats = _ground_truth(
        nusc, sample_token, cam_token, assoc, margin, min_containment,
        depth_rel_tol, depth_abs_tol)

    geo0 = assoc._geometry({})
    sc = assoc.center
    naive = np.where(geo0["valid"][:, sc],
                     assoc.col_map[geo0["v"][:, sc], geo0["u"][:, sc]], -1)
    naive = np.where(naive >= 0, naive, assoc.CLUTTER)

    args = (thing_cols, assoc.M, assoc.CLUTTER, assoc.OCCLUDED)
    out = {
        **box_stats,
        "points_total": int(len(gt)),
        "points_with_gt": int((gt >= 0).sum()),
        "points_on_unmatched_boxes": int((gt == -2).sum()),
        "naive_zero_elevation": _score(naive, gt, *args),
        "unary_only": _score(assoc.A_init, gt, *args),
        "optimised": _score(assoc.A, gt, *args),
    }

    causes = dict.fromkeys(CAUSES, 0)
    for i in np.nonzero((gt >= 0) & (assoc.A != gt))[0]:
        causes[_miss_cause(assoc, i, gt[i], k_sigma)] += 1
    out["optimised_miss_causes"] = causes
    return out
