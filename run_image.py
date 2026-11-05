#!/usr/bin/env python3
"""
run_image.py
------------
Panoptic segmentation and depth estimation for a single image file.

Computes per-segment depth statistics, prints a ranked table, and saves
JSON/CSV results and a labelled, depth-shaded image. With a metric depth
model, labels and the table show depth in metres.

Usage:
    python run_image.py street.jpg
    python run_image.py street.jpg --depth both --overview
    python run_image.py street.jpg --depth metric --offline
    python run_image.py --download-only --depth both
"""

import argparse
from pathlib import Path

import numpy as np

from segdepth.analysis import compute_segment_depths
from segdepth.cli import add_model_args, build_pipeline, download_models
from segdepth.reporting import print_table, save_results
from segdepth.utils import load_image
from segdepth.visualisation import render_labeled_depth_image, render_overview


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", nargs="?", help="path to the input image")
    p.add_argument("--download-only", action="store_true",
                   help="fetch the selected models into the models directory and exit")
    add_model_args(p)

    a = p.add_argument_group("analysis")
    a.add_argument("--erosion", type=int, default=2,
                   help="mask erosion (px) before statistics (default: %(default)s)")
    a.add_argument("--min-pixels", type=int, default=50,
                   help="ignore segments smaller than this (default: %(default)s)")

    o = p.add_argument_group("output")
    o.add_argument("--out-dir", default="results", help="(default: %(default)s)")
    o.add_argument("--overview", action="store_true", help="also save the four-panel overview figure")
    o.add_argument("--save-arrays", action="store_true", help="save depth and panoptic maps (.npy)")
    o.add_argument("--alpha", type=float, default=0.6, help="shading opacity (default: %(default)s)")
    o.add_argument("--cmap", default="turbo", help="depth colour map (default: %(default)s)")
    o.add_argument("--label-min-pixels", type=int, default=500,
                   help="no labels on smaller segments (default: %(default)s)")
    args = p.parse_args()
    if not args.download_only and not args.image:
        p.error("an image path is required unless --download-only is used")
    return args


def main():
    args = parse_args()
    if args.download_only:
        download_models(args)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    pipeline = build_pipeline(args)
    image = load_image(args.image)
    print(f"Loaded image: {image.size[0]}x{image.size[1]}")
    pred = pipeline(image)
    print(f"  Found {len(pred['segments_info'])} segments\n")

    has_metric = pred["metric_depth"] is not None
    results = compute_segment_depths(
        pred["seg_map"], pred["segments_info"], pipeline.id2label, pred["closeness"],
        erosion_px=args.erosion, min_pixels=args.min_pixels,
        metric_depth=pred["metric_depth"], metric_valid=pred["metric_valid"],
    )

    print_table(results, metric=has_metric)
    save_results(results, out_dir, stem)
    render_labeled_depth_image(
        image, pred["seg_map"], results, out_dir / f"{stem}_depth_labeled.png",
        alpha=args.alpha, cmap_name=args.cmap, label_min_pixels=args.label_min_pixels,
    )
    if args.overview:
        render_overview(image, pred["seg_map"], pred["closeness"], results,
                        out_dir / f"{stem}_overview.png")
    if args.save_arrays:
        np.save(out_dir / f"{stem}_panoptic.npy", pred["seg_map"])
        if pred["raw_depth"] is not None:
            np.save(out_dir / f"{stem}_depth_raw.npy", pred["raw_depth"])
        if has_metric:
            np.save(out_dir / f"{stem}_depth_metric.npy", pred["metric_depth"])


if __name__ == "__main__":
    main()
