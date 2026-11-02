#!/usr/bin/env python3
"""
run_radar_eval.py
-----------------
Batch evaluation and tuning of the radar-segment association on nuScenes.

Step 1 (cached): runs segmentation and depth once per keyframe and saves
seg_map, closeness and segment results to --cache-dir.
Step 2: for each association setting in a small grid, runs the association on
every cached frame, scores it against the 3D annotations, and pools raw counts
into micro-averaged recall and precision, with outcome and miss-cause totals.
Scenes are split into tuning and held-out sets.

Usage:
    python run_radar_eval.py --step 2
    python run_radar_eval.py --step 2 --grid --val-scenes scene-0916 scene-1077
    python run_radar_eval.py --step 2 --lidar-init      # oracle depth-scale ablation
"""

import argparse
import csv
import itertools
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes

from segdepth.analysis import compute_segment_depths
from segdepth.config import DEFAULT_MODELS_DIR, AssocConfig
from segdepth.depth_scale import fit_affine_inverse, ground_plane_anchors
from segdepth.nusc.evaluation import CAUSES, OUTCOMES, evaluate_with_annotations
from segdepth.nusc.lidar import project_lidar
from segdepth.nusc.radar_association import Params, RadarSegmentAssociator
from segdepth.nusc.scenes import load_camera_frame, scene_sample_tokens
from segdepth.nusc.sensors import CameraModel, load_radar_points

GRID = {                                   # edit to explore other settings
    "rho_clutter": [0.02, 0.05, 0.1],
    "rho_occluded": [0.1, 0.3],
    "elev_sigma_deg": [0.0, 4.0],
    "az_sigma_deg": [0.3, 0.5],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataroot", default="nscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--step", type=int, default=2, help="Use every n-th keyframe")
    p.add_argument("--cache-dir", default="cache/frames")
    p.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--radar-sweeps", type=int, default=3)
    p.add_argument("--lidar-init", action="store_true",
                   help="Initialise depth scale from LiDAR (ablation)")
    p.add_argument("--grid", action="store_true", help="Evaluate the settings in GRID")
    p.add_argument("--val-scenes", nargs="*", default=[],
                   help="Held-out scenes (default: last two scenes)")
    p.add_argument("--out", default="results/radar_eval.csv")
    return p.parse_args()


def cache_frames(nusc, args):
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames, pipeline = [], None
    for scene in nusc.scene:
        for idx, token in list(enumerate(scene_sample_tokens(nusc, scene)))[::args.step]:
            stem = cache / f"{scene['name']}_{idx:03d}_{args.camera}"
            if not stem.with_suffix(".npz").exists():
                if pipeline is None:
                    from segdepth.pipeline import PanopticDepthPipeline
                    pipeline = PanopticDepthPipeline(models_dir=args.models_dir,
                                                     offline=args.offline)
                image, _, _ = load_camera_frame(nusc, scene, token, args.camera, idx)
                out = pipeline(image)
                results = compute_segment_depths(out["seg_map"], out["segments_info"],
                                                 pipeline.id2label, out["closeness"])
                np.savez_compressed(stem.with_suffix(".npz"), seg_map=out["seg_map"],
                                    closeness=out["closeness"].astype(np.float32))
                stem.with_suffix(".json").write_text(json.dumps(results))
                print(f"Cached {stem.name}")
            frames.append((scene["name"], token, stem))
    return frames


def evaluate_frame(nusc, args, cfg, scene_name, sample_token, stem):
    data = np.load(stem.with_suffix(".npz"))
    seg_map, closeness = data["seg_map"], data["closeness"]
    results = json.loads(stem.with_suffix(".json").read_text())
    cam_token = nusc.get("sample", sample_token)["data"][args.camera]
    cam = CameraModel.from_nusc(nusc, cam_token)
    pts = load_radar_points(nusc, sample_token, cam_token, nsweeps=args.radar_sweeps)
    if len(pts) == 0 or not results:
        return None

    c_g, Z_g = ground_plane_anchors(seg_map, closeness, results, cam.K, cam.T_ego_from_cam)
    if args.lidar_init:
        u, v, z = project_lidar(nusc, sample_token, cam_token)
        init = fit_affine_inverse(closeness[v, u], z)
    elif len(c_g) >= 20:
        init = fit_affine_inverse(c_g, Z_g)
    else:
        return None

    assoc = RadarSegmentAssociator(pts, cam, seg_map, closeness, results,
                                   Params(init[0], init[1], cfg.sigma_inv_init),
                                   cfg, ground=(c_g, Z_g))
    try:
        assoc.optimise(verbose=False)
    except ValueError:
        return None
    return evaluate_with_annotations(nusc, sample_token, cam_token, assoc)


def pool(evals):
    out = {"frames": len(evals),
           "points_with_gt": sum(e["points_with_gt"] for e in evals),
           "points_on_unmatched_boxes": sum(e["points_on_unmatched_boxes"] for e in evals),
           "boxes_matched": sum(e["boxes_matched"] for e in evals),
           "boxes_with_points": sum(e["boxes_with_points"] for e in evals)}
    for method in ("naive_zero_elevation", "unary_only", "optimised"):
        n_gt = sum(e[method]["n_gt"] for e in evals)
        n_pred = sum(e[method]["n_pred"] for e in evals)
        n_tp = sum(e[method]["n_tp"] for e in evals)
        out[f"{method}_recall"] = round(n_tp / n_gt, 3) if n_gt else None
        out[f"{method}_precision"] = round(n_tp / n_pred, 3) if n_pred else None
    oc = Counter()
    for e in evals:
        oc.update(e["optimised"]["outcomes"])
    out.update({f"outcome_{k}": oc[k] for k in OUTCOMES})
    mc = Counter()
    for e in evals:
        mc.update(e["optimised_miss_causes"])
    out.update({f"miss_{k}": mc[k] for k in CAUSES})
    return out


def main():
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    frames = cache_frames(nusc, args)

    val = set(args.val_scenes) or {s["name"] for s in nusc.scene[-2:]}
    print(f"Held-out scenes: {sorted(val)}")

    base = AssocConfig()
    configs = ([replace(base, **dict(zip(GRID, combo)))
                for combo in itertools.product(*GRID.values())] if args.grid else [base])

    rows = []
    for k, cfg in enumerate(configs, start=1):
        evals = {"tune": [], "val": []}
        for scene_name, token, stem in frames:
            e = evaluate_frame(nusc, args, cfg, scene_name, token, stem)
            if e is not None:
                evals["val" if scene_name in val else "tune"].append(e)
        settings = {g: getattr(cfg, g) for g in GRID}
        for split, ev in evals.items():
            if ev:
                rows.append({"config": k, "split": split, **settings, **pool(ev)})
        tune = next((r for r in rows if r["config"] == k and r["split"] == "tune"), None)
        if tune:
            print(f"[{k}/{len(configs)}] {settings} -> tune recall "
                  f"{tune['optimised_recall']}, precision {tune['optimised_precision']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {args.out}")
    Path(args.out).with_suffix(".config.json").write_text(json.dumps(asdict(base), indent=2,
                                                                     default=list))


if __name__ == "__main__":
    main()
