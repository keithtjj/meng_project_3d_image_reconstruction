"""
depth_scale.py
--------------
Converting relative depth into metric depth.

Depth Anything V2 (relative) outputs affine-invariant inverse depth, so the
normalised closeness c relates to metric depth Z by c ≈ a / Z + b.

Functions:
  * fit_affine_inverse   Robust (Huber-reweighted) weighted least-squares fit of
                         (a, b) from closeness/depth pairs, e.g. LiDAR or radar.
  * closeness_to_metric  Apply a fit to convert closeness to metres.
  * spearman             Rank correlation, used to compare depth orderings.
  * ground_plane_anchors Metric depth of road pixels from ray / ground-plane
                         intersection, using camera intrinsics and extrinsics.
                         Provides a depth scale without LiDAR.
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
    """Metric depth in metres; np.inf where beyond max_depth or undefined."""
    closeness = np.asarray(closeness, dtype=float)
    if a <= 0:
        return np.full(closeness.shape, np.inf)
    inv = (closeness - b) / a
    with np.errstate(divide="ignore"):
        return np.where(inv > 1.0 / max_depth, 1.0 / np.maximum(inv, 1e-12), np.inf)


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
