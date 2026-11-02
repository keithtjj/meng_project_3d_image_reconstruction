#!/usr/bin/env python3
"""
run_nuscenes.py
---------------
Command-line entry point for nuScenes (v1.0-mini by default).

For each selected keyframe of a scene, loads a camera image (CAM_FRONT by
default), runs the segmentation/depth pipeline and computes per-segment depth.
Optional sensor fusion:
  --lidar        Validate against projected LiDAR and recover a metric scale.
  --radar        Heuristic radar association (depth-consistent if --lidar).
  --radar-assoc  Optimised radar association by energy minimisation.
  --radar-eval   Score --radar-assoc against 3D box annotations.
  --bev-range    Forward range (m) of the top-down association plot.
  --radar-arcs   Draw each radar point's candidate elevation arc on the image overlay.


Usage:
    python run_nuscenes.py --list-scenes
    python run_nuscenes.py --scene scene-0061 --sample-index 10 --lidar
    python run_nuscenes.py --scene scene-0061 --radar-assoc --radar-eval
    python run_nuscenes.py --scene scene-0103 --all-samples --lidar --radar
"""

import argparse
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes

from segdepth.analysis import compute_segment_depths
from segdepth.config import DEFAULT_MODELS_DIR, DEPTH_MODEL_ID, SEG_MODEL_ID, AssocConfig
from segdepth.depth_scale import closeness_to_metric, fit_affine_inverse, ground_plane_anchors
from segdepth.nusc.evaluation import evaluate_with_annotations
from segdepth.nusc.lidar import add_lidar_stats, project_lidar
from segdepth.nusc.radar_association import Params, RadarSegmentAssociator
from segdepth.nusc.radar_heuristic import add_radar_stats, project_zero_elevation
from segdepth.nusc.scenes import (CAMERAS, list_scenes, load_camera_frame,
                                  resolve_scene, scene_sample_tokens)
from segdepth.nusc.sensors import CameraModel, load_radar_points
from segdepth.nusc.bev import annotation_footprints, render_bev
from segdepth.pipeline import PanopticDepthPipeline
from segdepth.reporting import print_table, save_json, save_results
from segdepth.visualisation import render_association, render_labeled_depth_image



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Dataset
    p.add_argument("--dataroot", default="nscenes")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--list-scenes", action="store_true")
    p.add_argument("--scene", default=None, help="Scene name or index (default: first)")
    p.add_argument("--camera", default="CAM_FRONT", choices=CAMERAS)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--sample-index", type=int, default=0)
    sel.add_argument("--all-samples", action="store_true")
    # Sensors
    p.add_argument("--lidar", action="store_true")
    radar = p.add_mutually_exclusive_group()
    radar.add_argument("--radar", action="store_true", help="Heuristic radar association")
    radar.add_argument("--radar-assoc", action="store_true", help="Optimised radar association")
    p.add_argument("--radar-eval", action="store_true")
    p.add_argument("--radar-sweeps", type=int, default=3)
    p.add_argument("--radar-all-points", action="store_true",
                   help="Disable the devkit's default radar quality filters")
    # Models
    p.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--seg-model", default=SEG_MODEL_ID)
    p.add_argument("--depth-model", default=DEPTH_MODEL_ID)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-fuse-stuff", action="store_true")
    p.add_argument("--erosion", type=int, default=2)
    p.add_argument("--min-pixels", type=int, default=50)
    # Output
    p.add_argument("--out-dir", default="results/nuscenes")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--cmap", default="turbo")
    p.add_argument("--label-min-pixels", type=int, default=500)
    p.add_argument("--save-arrays", action="store_true")

    p.add_argument("--bev-range", type=float, default=60.0,
                   help="Forward range (m) of the top-down association plot")
    p.add_argument("--no-bev-gt", action="store_true",
                   help="Do not draw annotated box footprints on the top-down plot")
    p.add_argument("--radar-arcs", action="store_true",
                   help="Draw candidate elevation arcs on the radar image overlay")

    args = p.parse_args()
    if args.radar_eval and not args.radar_assoc:
        p.error("--radar-eval requires --radar-assoc")
    return args


def run_radar_heuristic(meta, results, seg_map, closeness, pts, cam, lidar_fit):
    proj = project_zero_elevation(pts, cam)
    metric = closeness_to_metric(closeness, *lidar_fit) if lidar_fit else None
    meta["radar"] = add_radar_stats(results, seg_map, proj, cam.fy, metric)
    r = meta["radar"]
    print(f"Radar ({r['method']}): {r['radar_points_assigned']}/{r['radar_points_in_image']} "
          f"points assigned to {r['segments_with_radar']} segments")


def run_radar_association(args, nusc, meta, results, seg_map, closeness, pts, cam,
                          lidar_fit, image, sample_token, cam_token, out_dir, stem):
    c_g, Z_g = ground_plane_anchors(seg_map, closeness, results, cam.K, cam.T_ego_from_cam)
    init = lidar_fit or (fit_affine_inverse(c_g, Z_g) if len(c_g) >= 20 else None)
    if init is None or len(pts) == 0:
        print("Radar association skipped: no depth-scale initialisation or no radar points")
        return

    cfg = AssocConfig()
    assoc = RadarSegmentAssociator(pts, cam, seg_map, closeness, results,
                                   Params(init[0], init[1], cfg.sigma_inv_init),
                                   cfg, ground=(c_g, Z_g))
    print("Optimising radar association...")
    meta["radar_association"] = assoc.optimise()
    assoc.annotate_results()

    save_json(assoc.point_table(), out_dir / f"{stem}_radar_points.json")
    render_association(image, assoc, out_dir / f"{stem}_radar_assoc.png",
                       show_arcs=args.radar_arcs)

    gt = None if args.no_bev_gt else annotation_footprints(
        nusc, sample_token, cam_token, max_dist=1.5 * args.bev_range)
    render_bev(assoc, out_dir / f"{stem}_radar_bev.png", gt=gt, max_range=args.bev_range)


    if args.radar_eval:
        ev = evaluate_with_annotations(nusc, sample_token, cam_token, assoc)
        meta["radar_association"]["evaluation"] = ev
        print(f"Association vs annotations: {ev}")


def main():
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    if args.list_scenes:
        list_scenes(nusc)
        return

    scene = resolve_scene(nusc, args.scene)
    tokens = scene_sample_tokens(nusc, scene)
    if args.all_samples:
        selected = list(enumerate(tokens))
    else:
        if not 0 <= args.sample_index < len(tokens):
            raise SystemExit(f"--sample-index must be 0-{len(tokens) - 1} for {scene['name']}")
        selected = [(args.sample_index, tokens[args.sample_index])]

    print(f"Scene: {scene['name']} - {scene['description']}")
    print(f"Processing {len(selected)} of {len(tokens)} keyframes ({args.camera})")

    pipeline = PanopticDepthPipeline(
        models_dir=args.models_dir, seg_model_id=args.seg_model,
        depth_model_id=args.depth_model, offline=args.offline,
        threshold=args.threshold, fuse_stuff=not args.no_fuse_stuff,
    )
    print(f"Using device: {pipeline.device}")

    out_dir = Path(args.out_dir) / scene["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for idx, sample_token in selected:
        print(f"\n=== {scene['name']} | keyframe {idx} ===")
        image, cam_token, meta = load_camera_frame(nusc, scene, sample_token, args.camera, idx)
        stem = f"{scene['name']}_{idx:03d}_{args.camera}"
        cam = CameraModel.from_nusc(nusc, cam_token)

        out = pipeline(image)
        seg_map, closeness = out["seg_map"], out["closeness"]
        results = compute_segment_depths(
            seg_map, out["segments_info"], pipeline.id2label, closeness,
            erosion_px=args.erosion, min_pixels=args.min_pixels,
        )

        lidar_fit = None
        if args.lidar:
            u, v, z = project_lidar(nusc, sample_token, cam_token)
            lv = add_lidar_stats(results, seg_map, closeness, u, v, z)
            meta["lidar_validation"] = lv
            lidar_fit = (lv["fit_a"], lv["fit_b"])

        if (args.radar or args.radar_assoc) and results:
            pts = load_radar_points(nusc, sample_token, cam_token, nsweeps=args.radar_sweeps,
                                    use_default_filters=not args.radar_all_points)
            if args.radar:
                run_radar_heuristic(meta, results, seg_map, closeness, pts, cam, lidar_fit)
            else:
                run_radar_association(args, nusc, meta, results, seg_map, closeness, pts, cam,
                                      lidar_fit, image, sample_token, cam_token, out_dir, stem)

        print()
        print_table(results, lidar=args.lidar, radar=args.radar or args.radar_assoc)
        if args.lidar:
            lv = meta["lidar_validation"]
            print(f"\nSpearman rho (relative depth vs LiDAR): "
                  f"{lv['spearman_rho_reldepth_vs_lidar']} over {lv['segments_with_lidar']} segments")

        save_results(results, out_dir, stem)
        render_labeled_depth_image(
            image, seg_map, results, out_dir / f"{stem}_depth_labeled.png",
            alpha=args.alpha, cmap_name=args.cmap, label_min_pixels=args.label_min_pixels,
        )
        save_json(meta, out_dir / f"{stem}_meta.json")
        if args.save_arrays:
            np.save(out_dir / f"{stem}_depth_raw.npy", out["raw_depth"])
            np.save(out_dir / f"{stem}_panoptic.npy", seg_map)

        summary.append({**meta, "n_segments": len(results),
                        "nearest": results[0]["label"] if results else None})

    if len(summary) > 1:
        save_json(summary, out_dir / f"{scene['name']}_{args.camera}_summary.json")


if __name__ == "__main__":
    main()
