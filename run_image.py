#!/usr/bin/env python3
"""
run_image.py
------------
Command-line entry point for a single image file.

Runs panoptic segmentation and relative depth estimation, computes per-segment
depth statistics, prints a ranked table, and saves JSON/CSV results, a
labelled depth-shaded image and (optionally) a four-panel overview.

Usage:
    python run_image.py street.jpg
    python run_image.py street.jpg --visualise --offline
    python run_image.py --download-only        # fetch models to ./models and exit
"""

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoModelForDepthEstimation, Mask2FormerForUniversalSegmentation

from segdepth.analysis import compute_segment_depths
from segdepth.config import DEFAULT_MODELS_DIR, DEPTH_MODEL_ID, SEG_MODEL_ID
from segdepth.models import load_model_and_processor
from segdepth.pipeline import PanopticDepthPipeline
from segdepth.reporting import print_table, save_results
from segdepth.utils import load_image
from segdepth.visualisation import render_labeled_depth_image, render_overview


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", nargs="?", help="Path to input image")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--seg-model", default=SEG_MODEL_ID, help="Hub ID or local path")
    p.add_argument("--depth-model", default=DEPTH_MODEL_ID, help="Hub ID or local path")
    p.add_argument("--offline", action="store_true", help="Only use local model copies")
    p.add_argument("--download-only", action="store_true", help="Save models locally, then exit")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-fuse-stuff", action="store_true")
    p.add_argument("--erosion", type=int, default=2)
    p.add_argument("--min-pixels", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--cmap", default="turbo")
    p.add_argument("--label-min-pixels", type=int, default=500)
    p.add_argument("--visualise", action="store_true", help="Also save the overview figure")
    p.add_argument("--save-arrays", action="store_true", help="Save raw depth/panoptic .npy")
    args = p.parse_args()
    if not args.download_only and not args.image:
        p.error("an image path is required unless --download-only is used")
    return args


def main():
    args = parse_args()

    if args.download_only:
        load_model_and_processor(args.seg_model, Mask2FormerForUniversalSegmentation, args.models_dir)
        load_model_and_processor(args.depth_model, AutoModelForDepthEstimation, args.models_dir)
        print("Models are available locally.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    pipeline = PanopticDepthPipeline(
        models_dir=args.models_dir, seg_model_id=args.seg_model,
        depth_model_id=args.depth_model, offline=args.offline,
        threshold=args.threshold, fuse_stuff=not args.no_fuse_stuff,
    )
    print(f"Using device: {pipeline.device}")

    image = load_image(args.image)
    print(f"Loaded image: {image.size[0]}x{image.size[1]}")
    out = pipeline(image)
    print(f"  Found {len(out['segments_info'])} segments\n")

    results = compute_segment_depths(
        out["seg_map"], out["segments_info"], pipeline.id2label, out["closeness"],
        erosion_px=args.erosion, min_pixels=args.min_pixels,
    )

    print_table(results)
    save_results(results, out_dir, stem)
    render_labeled_depth_image(
        image, out["seg_map"], results, out_dir / f"{stem}_depth_labeled.png",
        alpha=args.alpha, cmap_name=args.cmap, label_min_pixels=args.label_min_pixels,
    )
    if args.visualise:
        render_overview(image, out["seg_map"], out["closeness"], results,
                        out_dir / f"{stem}_overview.png")
    if args.save_arrays:
        np.save(out_dir / f"{stem}_depth_raw.npy", out["raw_depth"])
        np.save(out_dir / f"{stem}_panoptic.npy", out["seg_map"])


if __name__ == "__main__":
    main()
