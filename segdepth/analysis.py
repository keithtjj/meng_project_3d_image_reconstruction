"""
analysis.py
-----------
Per-segment relative depth statistics.

compute_segment_depths() combines a panoptic segmentation map with a
normalised closeness map (1 = nearest, 0 = farthest). For each segment it
erodes the mask to discard boundary pixels, then reports the median, mean,
spread and percentiles of closeness, a relative depth score
(relative_depth = 1 - median closeness), the centroid and bounding box.
Segments are ranked from nearest (depth_rank 1) to farthest.
"""

import numpy as np

from .config import CITYSCAPES_STUFF
from .utils import erode_mask


def compute_segment_depths(seg_map, segments_info, id2label, closeness,
                           erosion_px=2, min_pixels=50):
    results = []
    for seg in segments_info:
        seg_id = seg["id"]
        mask = seg_map == seg_id
        n_pixels = int(mask.sum())
        if n_pixels < min_pixels:
            continue

        core = erode_mask(mask, erosion_px)
        stat_mask = core if core.sum() >= min_pixels else mask
        values = closeness[stat_mask]

        ys, xs = np.nonzero(mask)
        label = id2label.get(seg["label_id"], str(seg["label_id"]))
        median_c = float(np.median(values))

        results.append({
            "segment_id": int(seg_id),
            "label": label,
            "category": "stuff" if label.lower() in CITYSCAPES_STUFF else "thing",
            "score": round(float(seg.get("score", float("nan"))), 4),
            "pixel_count": n_pixels,
            "median_closeness": round(median_c, 4),          # 1 = nearest
            "mean_closeness": round(float(values.mean()), 4),
            "std_closeness": round(float(values.std()), 4),
            "p10_closeness": round(float(np.percentile(values, 10)), 4),
            "p90_closeness": round(float(np.percentile(values, 90)), 4),
            "relative_depth": round(1.0 - median_c, 4),      # 0 = nearest
            "centroid_xy": [int(xs.mean()), int(ys.mean())],
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        })

    results.sort(key=lambda r: r["median_closeness"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["depth_rank"] = rank
    return results
