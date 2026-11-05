#!/usr/bin/env python3
"""
run_nuscenes.py
---------------
Segmentation and depth estimation on nuScenes camera keyframes (v1.0-mini by
default), with optional LiDAR and radar processing.

Per keyframe:
  * per-segment depth statistics and a labelled depth image
  * --lidar validate     compare depth estimates with projected LiDAR
  * --lidar init         also let LiDAR set the relative-depth scale used by
                         the radar association (comparisons are then not independent)
  * --radar heuristic    window / depth-consistent radar association
  * --radar optimised    energy-minimisation radar association, with
      --eval-assoc       scoring against 3D box annotations
      --dense-depth      whole-image metric depth from the association
                         (compared with LiDAR unless --lidar off)
Evaluation results are pooled over the selected keyframes into
<scene>_<camera>_aggregate.json.

Usage:
    python run_nuscenes.py --list-scenes
    python run_nuscenes.py --scene scene-0061 --samples 10 --lidar validate
    python run_nuscenes.py --scene scene-0061 --samples all --lidar validate \\
        --radar optimised --eval-assoc --dense-depth
    python run_nuscenes.py --scene scene-0061 --samples 0-9 --radar optimised \\
        --eval-assoc --set assoc.ordinal_weight=0 --set depth.shrink_k=4
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes

from segdepth.analysis import compute_segment_depths
from segdepth.cli import (add_model_args, apply_overrides, build_pipeline, metric_scale,
                          model_ids, parse_samples)
from segdepth.config import AssocConfig
from segdepth.depth_scale import (closeness_to_metric, fit_affine_inverse,
                                  fit_relative_to_metric, ground_plane_anchors)
from segdepth.nusc.bev import annotation_footprints, render_bev
from segdepth.nusc.evaluation import diagnose_failures, evaluate_with_annotations, summarise_counts
from segdepth.nusc.lidar import add_lidar_stats, project_lidar
from segdepth.nusc.radar_association import Params, RadarSegmentAssociator
from segdepth.nusc.radar_depth import (RadarDepthConfig, add_lidar_oracle, anchors_vs_lidar,
                                       compare_with_lidar, estimate_depth_maps,
                                       format_depth_table, pool_depth_sums,
                                       render_depth_comparison)
from segdepth.nusc.radar_heuristic import add_radar_stats, project_zero_elevation
from segdepth.nusc.scenes import (CAMERAS, list_scenes, load_camera_frame,
                                  resolve_scene, scene_sample_tokens)
from segdepth.nusc.sensors import CameraModel, load_radar_points
from segdepth.reporting import print_table, save_json, save_results
from segdepth.visualisation import render_association, render_labeled_depth_image


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("dataset")
    g.add_argument("--dataroot", default="nscenes", help="(default: %(default)s)")
    g.add_argument("--version", default="v1.0-mini", help="(default: %(default)s)")
    g.add_argument("--list-scenes", action="store_true", help="list scenes and exit")
    g.add_argument("--scene", default="0", help="scene name or index (default: first scene)")
    g.add_argument("--samples", default="0", metavar="SPEC",
                   help="keyframes: all, 5, 0-9 or 0,4,10-12 (default: %(default)s)")
    g.add_argument("--camera", default="CAM_FRONT", choices=CAMERAS)

    g = p.add_argument_group("sensors")
    g.add_argument("--lidar", choices=("off", "validate", "init"), default="off",
                   help="validate: compare with LiDAR; init: also set the relative-depth "
                        "scale of the radar association from LiDAR (default: off)")
    g.add_argument("--radar", choices=("off", "heuristic", "optimised"), default="off",
                   help="radar association method (default: off)")
    g.add_argument("--radar-sweeps", type=int, default=3,
                   help="sweeps accumulated per radar (default: %(default)s)")
    g.add_argument("--no-radar-filters", action="store_true",
                   help="keep radar points removed by the devkit's quality filters")
    g.add_argument("--assoc-depth", choices=("relative", "metric"), default=None,
                   help="depth map used by radar association "
                        "(default: relative if loaded, otherwise metric)")
    g.add_argument("--metric-scale", type=metric_scale, default=None, metavar="S|lidar",
                   help="scale applied to metric depth: a number, or 'lidar' (oracle)")

    g = p.add_argument_group("analyses (require --radar optimised)")
    g.add_argument("--eval-assoc", action="store_true",
                   help="score the association against 3D box annotations")
    g.add_argument("--dense-depth", action="store_true",
                   help="whole-image metric depth from the association")
    g.add_argument("--set", action="append", default=[], metavar="PREFIX.FIELD=VALUE",
                   help="override AssocConfig (assoc.FIELD) or RadarDepthConfig (depth.FIELD); "
                        "repeatable")

    add_model_args(p)

    g = p.add_argument_group("analysis")
    g.add_argument("--erosion", type=int, default=2, help="(default: %(default)s)")
    g.add_argument("--min-pixels", type=int, default=50, help="(default: %(default)s)")

    g = p.add_argument_group("output")
    g.add_argument("--out-dir", default="results/nuscenes", help="(default: %(default)s)")
    g.add_argument("--save-arrays", action="store_true",
                   help="save panoptic and depth maps (.npy)")
    g.add_argument("--bev-range", type=float, default=60.0,
                   help="forward range (m) of the BEV plot (default: %(default)s)")
    g.add_argument("--no-bev-gt", action="store_true", help="omit annotated boxes from the BEV plot")
    g.add_argument("--radar-arcs", action="store_true",
                   help="draw candidate elevation arcs on the radar overlay")
    g.add_argument("--alpha", type=float, default=0.6, help="(default: %(default)s)")
    g.add_argument("--cmap", default="turbo", help="(default: %(default)s)")
    g.add_argument("--label-min-pixels", type=int, default=500, help="(default: %(default)s)")

    args = p.parse_args()
    if args.list_scenes:
        return args

    relative, metric = model_ids(args)
    if (args.eval_assoc or args.dense_depth or args.radar_arcs) and args.radar != "optimised":
        p.error("--eval-assoc, --dense-depth and --radar-arcs require --radar optimised")
    if args.metric_scale is not None and metric is None:
        p.error("--metric-scale requires --depth metric or --depth both")
    if args.metric_scale == "lidar" and args.lidar == "off":
        p.error("--metric-scale lidar requires --lidar validate or --lidar init")
    if args.assoc_depth == "metric" and metric is None:
        p.error("--assoc-depth metric requires --depth metric or --depth both")
    if args.assoc_depth == "relative" and relative is None:
        p.error("--assoc-depth relative requires --depth relative or --depth both")
    args.assoc_depth = args.assoc_depth or ("relative" if relative else "metric")

    args.assoc_cfg, args.depth_cfg = AssocConfig(), RadarDepthConfig()
    try:
        apply_overrides(args.set, {"assoc": args.assoc_cfg, "depth": args.depth_cfg})
    except ValueError as e:
        p.error(str(e))

    if args.dense_depth and args.lidar == "off":
        print("Note: --dense-depth without LiDAR estimates depth but cannot evaluate it.")
    return args


# --------------------------------------------------------------------------- #
# Per-keyframe state and pooled totals
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    index: int
    sample_token: str
    cam_token: str
    image: object
    cam: CameraModel
    meta: dict
    stem: str
    out_dir: Path
    pred: dict = None
    results: list = None
    lidar_uvz: tuple = None
    lidar_fit: tuple = None
    lidar_scale: float = None


@dataclass
class Totals:
    eval_counts: Counter = field(default_factory=Counter)
    diag_counts: Counter = field(default_factory=Counter)
    depth_sums: Counter = field(default_factory=Counter)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def run_lidar(f):
    pred = f.pred
    u, v, z = project_lidar(nusc, f.sample_token, f.cam_token)
    f.lidar_uvz = (u, v, z)
    lv = add_lidar_stats(f.results, pred["seg_map"], pred["closeness"], u, v, z,
                         metric_depth=pred["metric_depth"], metric_valid=pred["metric_valid"])
    f.meta["lidar_validation"] = lv
    f.lidar_fit = (lv["fit_a"], lv["fit_b"])
    if lv.get("metric_vs_lidar"):
        f.lidar_scale = lv["metric_vs_lidar"]["median_scale_gt_over_pred"]


def get_metric_scale(args, f):
    """Scale for metric depth (None = unscaled)."""
    if args.metric_scale == "lidar":
        if f.lidar_scale is None:
            print("WARNING: no LiDAR metric scale for this frame; metric depth left unscaled")
        return f.lidar_scale
    return args.metric_scale


def choose_assoc_depth(args, f):
    """Returns (closeness map, initial (a, b) or None, description, whether LiDAR was used)."""
    pred = f.pred
    uses_lidar_scale = args.metric_scale == "lidar"
    if args.assoc_depth == "metric":
        c = np.where(pred["metric_valid"], pred["metric_closeness"], np.nan)
        a, b = pred["metric_affine"]
        s = get_metric_scale(args, f)
        if s is None:
            return c, (a, b), "metric model", False
        return c, (a * s, b), f"metric model x{s:.3f}", uses_lidar_scale

    c = pred["closeness"]
    if args.lidar == "init" and f.lidar_fit:
        return c, f.lidar_fit, "relative, scale from LiDAR", True
    if pred["metric_depth"] is not None:
        s = get_metric_scale(args, f) or 1.0
        fit = fit_relative_to_metric(c, pred["metric_depth"] * s, pred["metric_valid"])
        if fit and fit[0] > 0:
            return c, fit, "relative, scale from metric model", uses_lidar_scale
    return c, None, "relative, scale from ground plane", False


def run_radar_heuristic(args, f, pts):
    pred = f.pred
    proj = project_zero_elevation(pts, f.cam)
    if args.lidar == "init" and f.lidar_fit:
        metric, source = closeness_to_metric(pred["closeness"], *f.lidar_fit), "relative + LiDAR scale"
    elif pred["metric_depth"] is not None:
        metric = np.where(pred["metric_valid"], pred["metric_depth"], np.inf)
        metric = metric * (get_metric_scale(args, f) or 1.0)
        source = "metric model"
    else:
        metric, source = None, "none (window search)"
    rep = add_radar_stats(f.results, pred["seg_map"], proj, f.cam.fy, metric)
    rep["depth_source"] = source
    f.meta["radar"] = rep
    print(f"Radar ({rep['method']}, depth: {source}): {rep['radar_points_assigned']}/"
          f"{rep['radar_points_in_image']} points assigned to {rep['segments_with_radar']} segments")


def run_radar_optimised(args, f, pts, totals):
    seg_map = f.pred["seg_map"]
    c_assoc, init, source, uses_lidar = choose_assoc_depth(args, f)
    c_g, Z_g = ground_plane_anchors(seg_map, c_assoc, f.results, f.cam.K, f.cam.T_ego_from_cam)
    ok = np.isfinite(c_g)
    c_g, Z_g = c_g[ok], Z_g[ok]
    if init is None and len(c_g) >= 20:
        init = fit_affine_inverse(c_g, Z_g)
    if init is None or len(pts) == 0:
        print("Radar association skipped: no depth-scale initialisation or no radar points")
        return

    cfg = args.assoc_cfg
    assoc = RadarSegmentAssociator(pts, f.cam, seg_map, c_assoc, f.results,
                                   Params(init[0], init[1], cfg.sigma_inv_init),
                                   cfg, ground=(c_g, Z_g))
    print(f"Optimising radar association (depth: {source})...")
    try:
        summary = assoc.optimise()
    except ValueError as e:
        print(f"Radar association skipped: {e}")
        return
    summary.update(depth_source=source, lidar_used=uses_lidar)
    f.meta["radar_association"] = summary
    assoc.annotate_results()

    save_json(assoc.point_table(), f.out_dir / f"{f.stem}_radar_points.json")
    if hasattr(assoc, "pair_table"):
        save_json(assoc.pair_table(), f.out_dir / f"{f.stem}_radar_pairs.json")
    render_association(f.image, assoc, f.out_dir / f"{f.stem}_radar_assoc.png",
                       show_arcs=args.radar_arcs)
    boxes = None if args.no_bev_gt else annotation_footprints(
        nusc, f.sample_token, f.cam_token, max_dist=1.5 * args.bev_range)
    render_bev(assoc, f.out_dir / f"{f.stem}_radar_bev.png", gt=boxes, max_range=args.bev_range)

    if args.eval_assoc:
        ev, gt = evaluate_with_annotations(nusc, f.sample_token, f.cam_token, assoc)
        diag = diagnose_failures(assoc, gt)
        summary["evaluation"], summary["diagnosis"] = ev, diag
        totals.eval_counts.update(ev["counts"])
        totals.diag_counts.update(diag["counts"])
        print(f"Association vs annotations: {ev['metrics']}")
        print(f"Failure diagnosis: {diag['counts']} "
              f"(median depth ratio {diag['median_depth_ratio']})")

    if args.dense_depth:
        run_dense_depth(args, f, assoc, (c_g, Z_g), uses_lidar, totals)


def run_dense_depth(args, f, assoc, ground, uses_lidar, totals):
    pred, rc = f.pred, args.depth_cfg
    maps, anchors, info = estimate_depth_maps(assoc, ground, rc,
                                              pred["metric_depth"], pred["metric_valid"])
    main_method = rc.main_method if rc.main_method in maps else "radar_ground_opt"
    if args.save_arrays:
        np.save(f.out_dir / f"{f.stem}_depth_radar.npy", maps[main_method].astype(np.float32))
    rd = {"main_method": main_method, "info": info, "lidar_used_in_association": uses_lidar}
    f.meta["radar_depth"] = rd
    print(f"Dense depth: {info['n_anchors']} anchors, "
          f"{len(info['segment_corrections'])} segments corrected")
    if f.lidar_uvz is None:
        return

    if uses_lidar:
        print("WARNING: LiDAR set the association's depth scale; "
              "the comparison below is not independent.")
    u, v, z = f.lidar_uvz
    add_lidar_oracle(maps, assoc.closeness, u, v, z, rc)
    corrected = {c["segment_id"] for c in info["segment_corrections"]}
    rep = compare_with_lidar(maps, u, v, z, pred["seg_map"], f.results, corrected, rc)
    totals.depth_sums.update(rep.pop("sums"))
    rep["anchors_vs_lidar"] = anchors_vs_lidar(assoc, anchors, u, v, z)
    rd["lidar_comparison"] = rep

    print(f"\nWhole-image depth vs LiDAR ({rep['n_common']} common points):")
    print(format_depth_table(rep))
    av = rep["anchors_vs_lidar"]
    if av.get("n_checked"):
        print(f"Radar anchors vs LiDAR: {av['agreement_rate']:.0%} agree "
              f"({av['n_checked']} of {av['n_anchors']} checked), "
              f"median abs error {av['median_abs_error_m']} m")
    render_depth_comparison(f.image, maps, anchors, u, v, z, rep,
                            f.out_dir / f"{f.stem}_depth_vs_lidar.png", main=main_method)


def write_frame(args, f):
    pred = f.pred
    has_metric = pred["metric_depth"] is not None
    print()
    print_table(f.results, lidar=args.lidar != "off", radar=args.radar != "off", metric=has_metric)
    lv = f.meta.get("lidar_validation")
    if lv:
        print(f"\nSpearman rho (relative depth vs LiDAR): "
              f"{lv['spearman_rho_reldepth_vs_lidar']} over {lv['segments_with_lidar']} segments")
        if has_metric:
            print(f"Metric depth vs LiDAR: {lv.get('metric_vs_lidar')}")

    save_results(f.results, f.out_dir, f.stem)
    render_labeled_depth_image(
        f.image, pred["seg_map"], f.results, f.out_dir / f"{f.stem}_depth_labeled.png",
        alpha=args.alpha, cmap_name=args.cmap, label_min_pixels=args.label_min_pixels)
    save_json(f.meta, f.out_dir / f"{f.stem}_meta.json")
    if args.save_arrays:
        np.save(f.out_dir / f"{f.stem}_panoptic.npy", pred["seg_map"])
        if pred["raw_depth"] is not None:
            np.save(f.out_dir / f"{f.stem}_depth_raw.npy", pred["raw_depth"])
        if has_metric:
            np.save(f.out_dir / f"{f.stem}_depth_metric.npy", pred["metric_depth"])


def write_aggregate(args, scene, n_frames, totals, out_dir):
    agg = {"frames": n_frames, "settings": {"set": args.set, "assoc_depth": args.assoc_depth,
                                            "lidar": args.lidar}}
    if totals.eval_counts:
        agg["association"] = {"metrics": summarise_counts(totals.eval_counts),
                              "diagnosis": dict(totals.diag_counts)}
    if totals.depth_sums:
        agg["dense_depth"] = pool_depth_sums(totals.depth_sums)
    if len(agg) == 2:
        return

    print(f"\nAggregate over {n_frames} keyframe(s):")
    for m, s in agg.get("association", {}).get("metrics", {}).items():
        print(f"  {m:<22} recall {s['recall']}  precision {s['precision']}  "
              f"false-object {s['false_object_rate']}  (n_pos {s['n_pos']})")
    if "association" in agg:
        print(f"  diagnosis: {agg['association']['diagnosis']}")
    for m, parts in agg.get("dense_depth", {}).items():
        o = parts.get("overall")
        if o:
            print(f"  {m:<22} AbsRel {o['abs_rel']:.3f}  RMSE {o['rmse_m']:.2f} m  "
                  f"d<1.25 {o['delta_1_25']:.3f}  (n={o['n']})")
    save_json(agg, out_dir / f"{scene['name']}_{args.camera}_aggregate.json")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    global nusc
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    if args.list_scenes:
        list_scenes(nusc)
        return

    try:
        scene = resolve_scene(nusc, args.scene)
        tokens = scene_sample_tokens(nusc, scene)
        indices = parse_samples(args.samples, len(tokens))
    except ValueError as e:
        raise SystemExit(f"error: {e}")

    print(f"Scene: {scene['name']} - {scene['description']}")
    print(f"Processing {len(indices)} of {len(tokens)} keyframes ({args.camera})")

    pipeline = build_pipeline(args)
    out_dir = Path(args.out_dir) / scene["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, totals = [], Totals()

    for idx in indices:
        print(f"\n=== {scene['name']} | keyframe {idx} ===")
        image, cam_token, meta = load_camera_frame(nusc, scene, tokens[idx], args.camera, idx)
        f = Frame(idx, tokens[idx], cam_token, image, CameraModel.from_nusc(nusc, cam_token),
                  meta, f"{scene['name']}_{idx:03d}_{args.camera}", out_dir)

        f.pred = pipeline(image)
        f.results = compute_segment_depths(
            f.pred["seg_map"], f.pred["segments_info"], pipeline.id2label, f.pred["closeness"],
            erosion_px=args.erosion, min_pixels=args.min_pixels,
            metric_depth=f.pred["metric_depth"], metric_valid=f.pred["metric_valid"])

        if args.lidar != "off":
            run_lidar(f)
        if args.radar != "off" and f.results:
            pts = load_radar_points(nusc, f.sample_token, cam_token, nsweeps=args.radar_sweeps,
                                    use_default_filters=not args.no_radar_filters)
            if args.radar == "heuristic":
                run_radar_heuristic(args, f, pts)
            else:
                run_radar_optimised(args, f, pts, totals)

        write_frame(args, f)
        summary.append({**f.meta, "n_segments": len(f.results),
                        "nearest": f.results[0]["label"] if f.results else None})

    if len(summary) > 1:
        save_json(summary, out_dir / f"{scene['name']}_{args.camera}_summary.json")
    write_aggregate(args, scene, len(indices), totals, out_dir)


if __name__ == "__main__":
    main()
