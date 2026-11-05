"""
reporting.py
------------
Text and file outputs.

Functions:
  * print_table   Console table of segments ranked by depth, with optional
                  metric depth, LiDAR and radar columns.
  * save_results  Write per-segment results to JSON and CSV.
  * save_json     Write any JSON-serialisable object to a file.
"""

import csv
import json
from pathlib import Path


def _fmt(x):
    return "-" if x is None else f"{x:.1f}"


def print_table(results, lidar=False, radar=False, metric=False):
    header = (f"{'Rank':>4}  {'ID':>3}  {'Label':<15} {'Type':<6} {'Pixels':>8}  "
              f"{'RelDepth':>8}")
    if metric:
        header += f"  {'Metric m':>8}  {'P10-P90 m':>11}"
    if lidar:
        header += f"  {'LiDAR pts':>9}  {'LiDAR m':>8}  {'Est. m':>7}"
        if metric:
            header += f"  {'M-L m':>6}"
    if radar:
        header += f"  {'Radar pts':>9}  {'Radar m':>8}  {'Speed':>6}  {'RCS':>6}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = (f"{r['depth_rank']:>4}  {r['segment_id']:>3}  {r['label']:<15} "
                f"{r['category']:<6} {r['pixel_count']:>8}  {r['relative_depth']:>8.3f}")
        if metric:
            if r.get("metric_median_m") is not None:
                spread = f"{r['metric_p10_m']:.1f}-{r['metric_p90_m']:.1f}"
                line += f"  {r['metric_median_m']:>8.1f}  {spread:>11}"
            else:
                line += f"  {'far' if r.get('metric_beyond_range') else '-':>8}  {'':>11}"
        if lidar:
            line += (f"  {r.get('lidar_points', 0):>9}  {_fmt(r.get('lidar_median_m')):>8}  "
                     f"{_fmt(r.get('est_depth_m')):>7}")
            if metric:
                line += f"  {_fmt(r.get('metric_err_m')):>6}"
        if radar:
            line += (f"  {r.get('radar_points', 0):>9}  {_fmt(r.get('radar_median_m')):>8}  "
                     f"{_fmt(r.get('radar_speed_mps')):>6}  {_fmt(r.get('radar_rcs_dbsm')):>6}")
        print(line)


def save_json(obj, path: Path):
    Path(path).write_text(json.dumps(obj, indent=2))
    print(f"Saved: {path}")


def save_results(results, out_dir: Path, stem: str):
    out_dir = Path(out_dir)
    save_json(results, out_dir / f"{stem}_segments.json")
    if not results:
        return
    fieldnames = list(dict.fromkeys(k for r in results for k in r))
    csv_path = out_dir / f"{stem}_segments.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {csv_path}")
