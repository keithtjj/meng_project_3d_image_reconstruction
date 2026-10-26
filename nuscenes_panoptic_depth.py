#!/usr/bin/env python3
"""
Panoptic segmentation (Mask2Former, Cityscapes) + relative depth (Depth Anything V2)
on nuScenes camera images, with optional validation against projected LiDAR.

Requires panoptic_depth.py in the same directory.

Usage:
    python nuscenes_panoptic_depth.py --dataroot data/nuscenes --list-scenes
    python nuscenes_panoptic_depth.py --dataroot data/nuscenes --scene scene-0061
    python nuscenes_panoptic_depth.py --dataroot data/nuscenes --scene 3 --sample-index 10 --lidar
    python nuscenes_panoptic_depth.py --dataroot data/nuscenes --scene scene-0103 --all-samples --lidar
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes
from PIL import Image
from transformers import AutoModelForDepthEstimation, Mask2FormerForUniversalSegmentation

from panoptic_depth import (
    CITYSCAPES_STUFF,
    DEFAULT_MODELS_DIR,
    DEPTH_MODEL_ID,
    SEG_MODEL_ID,
    compute_segment_depths,
    erode_mask,
    get_device,
    load_model_and_processor,
    normalise,
    render_labeled_depth_image,
    save_results,
)

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


# --------------------------------------------------------------------------- #
# Pipeline: models loaded once, reused for every image
# --------------------------------------------------------------------------- #
class PanopticDepthPipeline:
    def __init__(self, device, models_dir, seg_model_id=SEG_MODEL_ID,
                 depth_model_id=DEPTH_MODEL_ID, offline=False,
                 threshold=0.5, fuse_stuff=True):
        self.device = device
        self.threshold = threshold

        print("Loading segmentation model...")
        self.seg_processor, self.seg_model = load_model_and_processor(
            seg_model_id, Mask2FormerForUniversalSegmentation, models_dir, offline)
        self.seg_model.to(device).eval()

        print("Loading depth model...")
        self.depth_processor, self.depth_model = load_model_and_processor(
            depth_model_id, AutoModelForDepthEstimation, models_dir, offline)
        self.depth_model.to(device).eval()

        self.id2label = {int(k): v for k, v in self.seg_model.config.id2label.items()}
        self.label_ids_to_fuse = (
            {i for i, n in self.id2label.items() if n.lower() in CITYSCAPES_STUFF}
            if fuse_stuff else None
        )

    @torch.inference_mode()
    def segment(self, image):
        inputs = self.seg_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.seg_model(**inputs)
        result = self.seg_processor.post_process_panoptic_segmentation(
            outputs,
            threshold=self.threshold,
            label_ids_to_fuse=self.label_ids_to_fuse,
            target_sizes=[image.size[::-1]],
        )[0]
        seg_map = result["segmentation"].cpu().numpy().astype(np.int32)
        return seg_map, result["segments_info"]

    @torch.inference_mode()
    def estimate_depth(self, image):
        inputs = self.depth_processor(images=image, return_tensors="pt").to(self.device)
        predicted = self.depth_model(**inputs).predicted_depth
        return F.interpolate(
            predicted.unsqueeze(1), size=image.size[::-1],
            mode="bilinear", align_corners=False,
        ).squeeze().float().cpu().numpy()


# --------------------------------------------------------------------------- #
# nuScenes helpers
# --------------------------------------------------------------------------- #
def list_scenes(nusc):
    print(f"{'Idx':>3}  {'Name':<11} {'Samples':>7}  {'Location':<22} Description")
    print("-" * 100)
    for i, s in enumerate(nusc.scene):
        location = nusc.get("log", s["log_token"])["location"]
        print(f"{i:>3}  {s['name']:<11} {s['nbr_samples']:>7}  {location:<22} {s['description']}")


def resolve_scene(nusc, scene_arg):
    if scene_arg is None:
        return nusc.scene[0]
    if scene_arg.isdigit():
        idx = int(scene_arg)
        if not 0 <= idx < len(nusc.scene):
            raise ValueError(f"Scene index {idx} out of range (0-{len(nusc.scene) - 1})")
        return nusc.scene[idx]
    for s in nusc.scene:
        if s["name"] == scene_arg:
            return s
    raise ValueError(f"Scene '{scene_arg}' not found. Use --list-scenes.")


def scene_sample_tokens(nusc, scene):
    """Keyframe sample tokens in temporal order."""
    tokens, token = [], scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = nusc.get("sample", token)["next"]
    return tokens


def load_camera_frame(nusc, scene, sample_token, camera, sample_index):
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"][camera]
    cam_data = nusc.get("sample_data", cam_token)
    calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", cam_data["ego_pose_token"])

    image = Image.open(nusc.get_sample_data_path(cam_token)).convert("RGB")

    meta = {
        "scene_name": scene["name"],
        "scene_description": scene["description"],
        "location": nusc.get("log", scene["log_token"])["location"],
        "sample_index": sample_index,
        "sample_token": sample_token,
        "sample_data_token": cam_token,
        "camera": camera,
        "timestamp_us": cam_data["timestamp"],
        "image_file": cam_data["filename"],
        "image_size_wh": list(image.size),
        "camera_intrinsic": calib["camera_intrinsic"],
        "ego_translation": ego["translation"],
        "ego_rotation_wxyz": ego["rotation"],
    }
    return image, cam_token, meta


# --------------------------------------------------------------------------- #
# LiDAR validation
# --------------------------------------------------------------------------- #
def project_lidar(nusc, sample_token, cam_token, min_dist=1.0):
    """Project the keyframe LIDAR_TOP sweep into the camera image.
    Returns integer pixel coordinates (u, v) and camera-frame depth z in metres."""
    lidar_token = nusc.get("sample", sample_token)["data"]["LIDAR_TOP"]
    points, depths, _ = nusc.explorer.map_pointcloud_to_image(
        lidar_token, cam_token, min_dist=min_dist)
    return (points[0].astype(np.int64), points[1].astype(np.int64),
            np.asarray(depths, dtype=np.float32))


def fit_inverse_depth_scale(closeness, u, v, z):
    """Least-squares fit of closeness ≈ a * (1/z) + b.
    Valid because Depth Anything V2 (relative) outputs affine-invariant inverse depth."""
    c = closeness[v, u]
    A = np.stack([1.0 / z, np.ones_like(z)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, c, rcond=None)
    return float(a), float(b)


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def add_lidar_stats(results, seg_map, closeness, u, v, z,
                    erosion_px=3, min_points=5, max_depth=200.0):
    """Adds per-segment LiDAR depth and scale-recovered depth estimates (in place).
    Returns image-level validation statistics."""
    a, b = fit_inverse_depth_scale(closeness, u, v, z)

    for r in results:
        mask = seg_map == r["segment_id"]
        # Erode to discard boundary points, where LiDAR/camera parallax and
        # mask inaccuracy cause points to land on the wrong object
        core = erode_mask(mask, erosion_px)
        if not core[v, u].any():
            core = mask
        in_seg = core[v, u]
        n = int(in_seg.sum())

        r["lidar_points"] = n
        r["lidar_median_m"] = round(float(np.median(z[in_seg])), 2) if n >= min_points else None

        inv = (r["median_closeness"] - b) / a if a > 0 else -1.0
        r["est_depth_m"] = round(1.0 / inv, 2) if inv > 1.0 / max_depth else None

    valid = [r for r in results if r["lidar_median_m"] is not None]
    rho = (spearman([r["relative_depth"] for r in valid],
                    [r["lidar_median_m"] for r in valid])
           if len(valid) >= 3 else None)

    return {
        "lidar_points_in_image": int(len(z)),
        "fit_a": round(a, 4),
        "fit_b": round(b, 4),
        "segments_with_lidar": len(valid),
        "spearman_rho_reldepth_vs_lidar": None if rho is None else round(rho, 3),
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def print_table(results, lidar=False):
    header = f"{'Rank':>4}  {'Label':<15} {'Type':<6} {'Pixels':>8}  {'RelDepth':>8}"
    if lidar:
        header += f"  {'LiDAR pts':>9}  {'LiDAR m':>8}  {'Est. m':>7}"
    print(header)
    print("-" * len(header))
    fmt = lambda x: f"{x:.1f}" if x is not None else "-"
    for r in results:
        line = (f"{r['depth_rank']:>4}  {r['label']:<15} {r['category']:<6} "
                f"{r['pixel_count']:>8}  {r['relative_depth']:>8.3f}")
        if lidar:
            line += (f"  {r['lidar_points']:>9}  {fmt(r['lidar_median_m']):>8}  "
                     f"{fmt(r['est_depth_m']):>7}")
        print(line)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # nuScenes
    parser.add_argument("--dataroot", default="nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--scene", default=None,
                        help="Scene name (e.g. scene-0061) or index. Default: first scene")
    parser.add_argument("--camera", default="CAM_FRONT", choices=CAMERAS)
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--sample-index", type=int, default=0,
                     help="Keyframe index within the scene (0 = first)")
    sel.add_argument("--all-samples", action="store_true",
                     help="Process every keyframe in the scene")
    parser.add_argument("--lidar", action="store_true",
                        help="Validate against projected LIDAR_TOP points")
    # Models / pipeline
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--seg-model", default=SEG_MODEL_ID)
    parser.add_argument("--depth-model", default=DEPTH_MODEL_ID)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-fuse-stuff", action="store_true")
    parser.add_argument("--erosion", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=50)
    # Output
    parser.add_argument("--out-dir", default="results/nuscenes")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument("--label-min-pixels", type=int, default=500)
    parser.add_argument("--save-arrays", action="store_true",
                        help="Also save raw depth and panoptic maps as .npy")
    args = parser.parse_args()

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
            parser.error(f"--sample-index must be 0-{len(tokens) - 1} for {scene['name']}")
        selected = [(args.sample_index, tokens[args.sample_index])]

    print(f"Scene: {scene['name']} - {scene['description']}")
    print(f"Processing {len(selected)} of {len(tokens)} keyframes ({args.camera})")

    device = get_device()
    print(f"Using device: {device}")
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PanopticDepthPipeline(
        device, models_dir, args.seg_model, args.depth_model,
        offline=args.offline, threshold=args.threshold,
        fuse_stuff=not args.no_fuse_stuff,
    )

    out_dir = Path(args.out_dir) / scene["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for idx, sample_token in selected:
        print(f"\n=== {scene['name']} | keyframe {idx} ===")
        image, cam_token, meta = load_camera_frame(nusc, scene, sample_token,
                                                   args.camera, idx)
        stem = f"{scene['name']}_{idx:03d}_{args.camera}"

        seg_map, segments_info = pipeline.segment(image)
        raw_depth = pipeline.estimate_depth(image)
        closeness = normalise(raw_depth)

        results = compute_segment_depths(
            seg_map, segments_info, pipeline.id2label, closeness,
            erosion_px=args.erosion, min_pixels=args.min_pixels,
        )

        if args.lidar:
            u, v, z = project_lidar(nusc, sample_token, cam_token)
            meta["lidar_validation"] = add_lidar_stats(results, seg_map, closeness, u, v, z)

        print_table(results, lidar=args.lidar)
        if args.lidar:
            lv = meta["lidar_validation"]
            print(f"\nSpearman rho (relative depth vs LiDAR): "
                  f"{lv['spearman_rho_reldepth_vs_lidar']} "
                  f"over {lv['segments_with_lidar']} segments")

        save_results(results, out_dir, stem)
        render_labeled_depth_image(
            image, seg_map, results, out_dir / f"{stem}_depth_labeled.png",
            alpha=args.alpha, cmap_name=args.cmap,
            label_min_pixels=args.label_min_pixels,
        )
        (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))
        if args.save_arrays:
            np.save(out_dir / f"{stem}_depth_raw.npy", raw_depth)
            np.save(out_dir / f"{stem}_panoptic.npy", seg_map)

        summary.append({**meta, "n_segments": len(results),
                        "nearest": results[0]["label"] if results else None})
        
        nusc.render_pointcloud_in_image(sample_token,
                                        pointsensor_channel="RADAR_FRONT",
                                        camera_channel="CAM_FRONT",
                                        out_path="radar_check.png")
        
    if len(summary) > 1:
        summary_path = out_dir / f"{scene['name']}_{args.camera}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved scene summary: {summary_path}")    


if __name__ == "__main__":
    main()
