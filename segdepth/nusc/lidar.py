"""
lidar.py
--------
LiDAR projection and validation of the relative depth estimates.

Functions:
  * project_lidar    Project the keyframe LIDAR_TOP sweep into a camera image
                     using the devkit; returns pixel coordinates and depth (m).
  * add_lidar_stats  For each segment: median LiDAR depth inside the eroded
                     mask, and an estimated metric depth from a robust fit of
                     closeness ≈ a / Z + b. Also reports the Spearman rank
                     correlation between relative depth and LiDAR depth.
"""

import numpy as np

from ..depth_scale import closeness_to_metric, fit_affine_inverse, spearman
from ..utils import erode_mask


def project_lidar(nusc, sample_token, cam_token, min_dist=1.0):
    lidar_token = nusc.get("sample", sample_token)["data"]["LIDAR_TOP"]
    points, depths, _ = nusc.explorer.map_pointcloud_to_image(
        lidar_token, cam_token, min_dist=min_dist)
    return (points[0].astype(np.int64), points[1].astype(np.int64),
            np.asarray(depths, dtype=np.float32))


def add_lidar_stats(results, seg_map, closeness, u, v, z,
                    erosion_px=3, min_points=5, max_depth=200.0):
    """Adds lidar_points, lidar_median_m and est_depth_m to each result (in place)."""
    a, b = fit_affine_inverse(closeness[v, u], z)

    for r in results:
        mask = seg_map == r["segment_id"]
        core = erode_mask(mask, erosion_px)
        if not core[v, u].any():
            core = mask
        in_seg = core[v, u]
        n = int(in_seg.sum())

        r["lidar_points"] = n
        r["lidar_median_m"] = round(float(np.median(z[in_seg])), 2) if n >= min_points else None
        est = float(closeness_to_metric(r["median_closeness"], a, b, max_depth))
        r["est_depth_m"] = round(est, 2) if np.isfinite(est) else None

    valid = [r for r in results if r["lidar_median_m"] is not None]
    rho = (spearman([r["relative_depth"] for r in valid],
                    [r["lidar_median_m"] for r in valid])
           if len(valid) >= 3 else None)

    return {
        "lidar_points_in_image": int(len(z)),
        "fit_a": round(a, 6),
        "fit_b": round(b, 6),
        "segments_with_lidar": len(valid),
        "spearman_rho_reldepth_vs_lidar": None if rho is None else round(rho, 3),
    }
