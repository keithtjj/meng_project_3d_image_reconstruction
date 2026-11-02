"""
radar_heuristic.py
------------------
Simple, non-iterative radar-to-segment association.

Radar has no elevation, so each detection is projected at zero elevation and
the segment is chosen from a vertical band in the image around that point.

Functions:
  * project_zero_elevation   Project RadarPoints into the camera at zero
                             elevation, keeping velocity, RCS and dyn_prop.
  * assign_window            Most frequent 'thing' (then any usable) segment
                             in a vertical window scaled by range.
  * assign_depth_consistent  Same band, but only pixels whose metric depth
                             agrees with the radar range are counted.
  * add_radar_stats          Per-segment radar depth, speed, RCS and moving
                             flag, using depth-consistent association if a
                             metric depth map is supplied.
"""

import numpy as np

from ..config import RADAR_IGNORE


def project_zero_elevation(pts, cam, min_dist=1.0):
    empty = dict(index=np.zeros(0, int), u=np.zeros(0, int), v=np.zeros(0, int),
                 z=np.zeros(0), rcs=np.zeros(0), vel=np.zeros((0, 3)),
                 speed=np.zeros(0), dyn_prop=np.zeros(0, np.int8), channel=np.zeros(0, str))
    if len(pts) == 0:
        return empty

    xyz_r = np.column_stack([pts.xy, np.zeros(len(pts))])
    vel_r = np.column_stack([pts.vel, np.zeros(len(pts))])
    xyz, vel = np.empty_like(xyz_r), np.empty_like(vel_r)
    for k in np.unique(pts.sweep):
        m = pts.sweep == k
        R, t = pts.T_cam[k][:3, :3], pts.T_cam[k][:3, 3]
        xyz[m] = xyz_r[m] @ R.T + t
        vel[m] = vel_r[m] @ R.T                       # vectors: rotate only

    Z = xyz[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.K[0, 0] * xyz[:, 0] / Z + cam.K[0, 2]
        v = cam.K[1, 1] * xyz[:, 1] / Z + cam.K[1, 2]
    keep = ((Z > min_dist) & (u > 1) & (u < cam.width - 1) &
            (v > 1) & (v < cam.height - 1))

    return dict(index=np.nonzero(keep)[0], u=u[keep].astype(int), v=v[keep].astype(int),
                z=Z[keep].astype(np.float32), rcs=pts.rcs[keep], vel=vel[keep],
                speed=np.linalg.norm(vel[keep], axis=1), dyn_prop=pts.dyn_prop[keep],
                channel=pts.channel[keep])


def assign_window(seg_map, results, proj, fy, height_above=2.0, height_below=0.6):
    H = seg_map.shape[0]
    things = {r["segment_id"] for r in results if r["category"] == "thing"}
    usable = {r["segment_id"] for r in results if r["label"].lower() not in RADAR_IGNORE}

    assigned = np.full(len(proj["z"]), -1, dtype=np.int64)
    for i, (u, v, z) in enumerate(zip(proj["u"], proj["v"], proj["z"])):
        top = max(0, int(v - fy * height_above / z))
        bottom = min(H, int(v + fy * height_below / z) + 1)
        ids, counts = np.unique(seg_map[top:bottom, u], return_counts=True)
        for candidates in (things, usable):
            sel = [(c, s) for s, c in zip(ids, counts) if s in candidates]
            if sel:
                assigned[i] = max(sel)[1]
                break
    return assigned


def assign_depth_consistent(seg_map, results, proj, fy, metric_depth,
                            height_above=2.5, height_below=1.0,
                            half_width=3, rel_tol=0.15, abs_tol=1.0):
    """Returns (segment_ids, rows); rows estimate the true reflection row."""
    H, W = seg_map.shape
    usable = np.array([r["segment_id"] for r in results
                       if r["label"].lower() not in RADAR_IGNORE])
    seg_ids = np.full(len(proj["z"]), -1, dtype=np.int64)
    rows = np.full(len(proj["z"]), -1, dtype=np.int64)

    for i, (u, v, z) in enumerate(zip(proj["u"], proj["v"], proj["z"])):
        top = max(0, int(v - fy * height_above / z))
        bottom = min(H, int(v + fy * height_below / z) + 1)
        left, right = max(0, u - half_width), min(W, u + half_width + 1)
        band_seg = seg_map[top:bottom, left:right]
        band_depth = metric_depth[top:bottom, left:right]

        ok = (np.abs(band_depth - z) < max(abs_tol, rel_tol * z)) & np.isin(band_seg, usable)
        if not ok.any():
            continue
        ids, counts = np.unique(band_seg[ok], return_counts=True)
        seg_ids[i] = ids[np.argmax(counts)]
        rows[i] = top + int(np.median(np.nonzero(ok)[0]))
    return seg_ids, rows


def add_radar_stats(results, seg_map, proj, fy, metric_depth=None,
                    min_points=1, moving_thresh=0.5):
    if metric_depth is not None:
        assigned, _ = assign_depth_consistent(seg_map, results, proj, fy, metric_depth)
        method = "depth_consistent"
    else:
        assigned = assign_window(seg_map, results, proj, fy)
        method = "window"

    for r in results:
        sel = assigned == r["segment_id"]
        n = int(sel.sum())
        r["radar_points"] = n
        if n >= min_points:
            r["radar_median_m"] = round(float(np.median(proj["z"][sel])), 2)
            r["radar_speed_mps"] = round(float(np.median(proj["speed"][sel])), 2)
            r["radar_rcs_dbsm"] = round(float(np.median(proj["rcs"][sel])), 1)
            r["radar_moving"] = bool(r["radar_speed_mps"] > moving_thresh)
        else:
            r["radar_median_m"] = r["radar_speed_mps"] = r["radar_rcs_dbsm"] = None
            r["radar_moving"] = None

    return {
        "method": method,
        "radar_points_in_image": int(len(proj["z"])),
        "radar_points_assigned": int((assigned >= 0).sum()),
        "segments_with_radar": sum(r["radar_points"] >= min_points for r in results),
    }
