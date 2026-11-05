"""
depth_scale.py
--------------
Converting between relative and metric depth, and measuring depth error.

Depth Anything V2 (relative) outputs affine-invariant inverse depth, so the
normalised closeness c relates to metric depth Z by c ≈ a / Z + b.

Functions:
  * fit_affine_inverse       Robust (Huber-reweighted) weighted least-squares fit
                             of (a, b) from closeness/depth pairs.
  * closeness_to_metric      Apply (a, b) to convert closeness to metres.
  * inverse_depth_closeness  Express a metric depth map as closeness with exact
                             (a, b), so metric output plugs into code that
                             works in closeness.
  * fit_relative_to_metric   Scale a relative closeness map using a metric
                             depth map as reference (no LiDAR needed).
  * depth_error_stats        AbsRel, RMSE, delta<1.25 and median scale ratio.
  * spearman                 Rank correlation, used to compare depth orderings.
  * ground_plane_anchors     Metric depth of road pixels from ray / ground-plane
                             intersection, using camera intrinsics/extrinsics.
"""

import numpy as np


def fit_affine_inverse(c, Z, w=None, huber=0.05, n_iter=10):
    c, Z = np.asarray(c, dtype=float), np.asarray(Z, dtype=float)
    w = np.ones_like(c) if w is None else np.asarray(w, dtype=float)
    A = np.stack([1.0 / Z, np.ones_like(Z)], axis=1)
    rw = np.ones_like(c)
    for _ in range(n_iter):
        sw = np.sqrt(w * rw)
        (a, b), *_ = np.linalg.lstsq(A * sw[:, None], c * sw, rcond=None)
        res = np.abs(c - (a / Z + b))
        rw = np.where(res <= huber, 1.0, huber / np.maximum(res, 1e-12))
    return float(a), float(b)


def closeness_to_metric(closeness, a, b, max_depth=200.0):
    """Metric depth in metres; np.inf where beyond max_depth or undefined (incl. NaN)."""
    closeness = np.asarray(closeness, dtype=float)
    if a <= 0:
        return np.full(closeness.shape, np.inf)
    inv = (closeness - b) / a
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(inv > 1.0 / max_depth, 1.0 / np.maximum(inv, 1e-12), np.inf)


def inverse_depth_closeness(depth, valid, max_depth, min_depth=0.1):
    """
    Returns (closeness, (a, b)). closeness = (1/Z - 1/max_depth) * a, in [0, 1]
    over valid pixels; invalid pixels are set to 0 (= max_depth).
    """
    inv = 1.0 / np.clip(depth, min_depth, max_depth)
    lo = 1.0 / max_depth
    hi = float(inv[valid].max()) if valid.any() else 1.0 / min_depth
    a = 1.0 / max(hi - lo, 1e-8)
    b = -lo * a
    c = np.where(valid, (inv - lo) * a, 0.0).astype(np.float32)
    return c, (float(a), float(b))


def fit_relative_to_metric(closeness, metric_depth, valid, n=20000,
                           z_range=(1.0, 60.0), seed=0):
    """Fit (a, b) for a relative closeness map using metric depth as reference.
    Returns None if too few usable pixels."""
    m = valid & (metric_depth > z_range[0]) & (metric_depth < z_range[1])
    ys, xs = np.nonzero(m)
    if len(xs) < 100:
        return None
    if len(xs) > n:
        pick = np.random.default_rng(seed).choice(len(xs), n, replace=False)
        ys, xs = ys[pick], xs[pick]
    return fit_affine_inverse(closeness[ys, xs], metric_depth[ys, xs])


def depth_error_stats(pred, gt, min_points=10):
    pred, gt = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    p, g = pred[ok], gt[ok]
    if len(g) < min_points:
        return None
    s = float(np.median(g / p))
    return {
        "n": int(len(g)),
        "abs_rel": round(float(np.mean(np.abs(p - g) / g)), 4),
        "rmse_m": round(float(np.sqrt(np.mean((p - g) ** 2))), 3),
        "delta_1.25": round(float(np.mean(np.maximum(p / g, g / p) < 1.25)), 4),
        "median_scale_gt_over_pred": round(s, 4),
        "abs_rel_after_scaling": round(float(np.mean(np.abs(s * p - g) / g)), 4),
    }


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def ground_plane_anchors(seg_map, closeness, results, K, T_ego_from_cam,
                         max_dist=40.0, n=600, seed=0):
    """
    Returns (closeness, depth) pairs for sampled road pixels. Assumes a flat
    road at ego z = 0 (nuScenes ego frame is at ground level).
    """
    road_ids = [r["segment_id"] for r in results if r["label"].lower() == "road"]
    ys, xs = np.nonzero(np.isin(seg_map, road_ids))
    if len(xs) == 0:
        return np.zeros(0), np.zeros(0)

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(xs), size=min(3 * n, len(xs)), replace=False)
    xs, ys = xs[pick], ys[pick]

    rays = np.linalg.inv(K) @ np.vstack([xs, ys, np.ones_like(xs)]).astype(float)
    R, t = T_ego_from_cam[:3, :3], T_ego_from_cam[:3, 3]
    d = R @ rays
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -t[2] / d[2]          # camera-frame rays have z = 1, so lam = camera depth
    ok = (d[2] < 0) & (lam > 1.0) & (lam < max_dist)
    return closeness[ys[ok], xs[ok]][:n], lam[ok][:n]
