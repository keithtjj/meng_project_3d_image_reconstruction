#!/usr/bin/env python3
"""
Panoptic segmentation (Mask2Former, Cityscapes) + relative depth estimation
(Depth Anything V2), followed by per-segment relative depth analysis.

Models are saved to a local directory on first use and loaded from there
on subsequent runs.

Usage:
    python panoptic_depth.py path/to/image.jpg --out-dir results --visualise
    python panoptic_depth.py --download-only            # just fetch models
    python panoptic_depth.py image.jpg --offline        # never touch the Hub
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    Mask2FormerForUniversalSegmentation,
)

SEG_MODEL_ID = "facebook/mask2former-swin-large-cityscapes-panoptic"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"  # or -Small-hf / -Base-hf
DEFAULT_MODELS_DIR = "models"

# Cityscapes "stuff" classes (amorphous regions); everything else is a "thing"
CITYSCAPES_STUFF = {
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
}


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Binary erosion via max-pooling. Removes boundary pixels where depth
    tends to bleed between neighbouring objects."""
    if pixels <= 0:
        return mask
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    k = 2 * pixels + 1
    eroded = -F.max_pool2d(-t, kernel_size=k, stride=1, padding=pixels)
    return eroded[0, 0].numpy() > 0.5


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    k = 2 * pixels + 1
    dilated = F.max_pool2d(t, kernel_size=k, stride=1, padding=pixels)
    return dilated[0, 0].numpy() > 0.5


def free_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Local model storage
# --------------------------------------------------------------------------- #
def local_model_dir(models_dir: Path, model_id: str) -> Path:
    """facebook/mask2former-... -> models/facebook--mask2former-..."""
    return models_dir / model_id.replace("/", "--")


def update_manifest(models_dir: Path, model_id: str, local_path: Path):
    """Record where each model came from, for reproducibility."""
    manifest_path = models_dir / "manifest.json"
    data = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    data[model_id] = {
        "local_path": str(local_path),
        "transformers_version": transformers.__version__,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(data, indent=2))


def load_model_and_processor(model_id: str, model_cls, models_dir: Path,
                             offline: bool = False):
    """
    Load a model + image processor from the local models directory if present.
    Otherwise download from the Hugging Face Hub and save a local copy.
    `model_id` may also be a path to an existing local model directory.
    """
    as_path = Path(model_id)
    local_path = as_path if as_path.is_dir() else local_model_dir(models_dir, model_id)

    if (local_path / "config.json").exists():
        print(f"  Loading local copy: {local_path}")
        processor = AutoImageProcessor.from_pretrained(local_path, local_files_only=True)
        model = model_cls.from_pretrained(local_path, local_files_only=True)
        return processor, model

    if offline:
        raise FileNotFoundError(
            f"No local copy of '{model_id}' found at {local_path} and --offline is set. "
            f"Run once without --offline (or with --download-only) first."
        )

    print(f"  Downloading {model_id} from the Hugging Face Hub...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = model_cls.from_pretrained(model_id)

    local_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(local_path)
    model.save_pretrained(local_path)  # safetensors by default
    update_manifest(models_dir, model_id, local_path)
    print(f"  Saved local copy to: {local_path}")
    return processor, model


# --------------------------------------------------------------------------- #
# Panoptic segmentation
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def run_panoptic_segmentation(image, device, model_id, models_dir,
                              threshold=0.5, fuse_stuff=True, offline=False):
    processor, model = load_model_and_processor(
        model_id, Mask2FormerForUniversalSegmentation, models_dir, offline
    )
    model.to(device).eval()

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label_ids_to_fuse = (
        {i for i, name in id2label.items() if name.lower() in CITYSCAPES_STUFF}
        if fuse_stuff else None
    )

    result = processor.post_process_panoptic_segmentation(
        outputs,
        threshold=threshold,
        label_ids_to_fuse=label_ids_to_fuse,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]

    seg_map = result["segmentation"].cpu().numpy().astype(np.int32)
    segments_info = result["segments_info"]

    del model
    free_memory(device)
    return seg_map, segments_info, id2label


# --------------------------------------------------------------------------- #
# Depth estimation
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def run_depth_estimation(image, device, model_id, models_dir, offline=False):
    processor, model = load_model_and_processor(
        model_id, AutoModelForDepthEstimation, models_dir, offline
    )
    model.to(device).eval()

    inputs = processor(images=image, return_tensors="pt").to(device)
    predicted = model(**inputs).predicted_depth  # (1, h, w)

    depth = F.interpolate(
        predicted.unsqueeze(1),
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    ).squeeze().float().cpu().numpy()

    del model
    free_memory(device)
    return depth


def normalise(depth: np.ndarray) -> np.ndarray:
    d_min, d_max = float(depth.min()), float(depth.max())
    return (depth - d_min) / (d_max - d_min + 1e-8)


# --------------------------------------------------------------------------- #
# Per-segment analysis
# --------------------------------------------------------------------------- #
def compute_segment_depths(seg_map, segments_info, id2label, closeness,
                           erosion_px=2, min_pixels=50):
    """
    `closeness` is the normalised Depth Anything V2 output in [0, 1].
    Depth Anything V2 (relative) predicts affine-invariant *inverse* depth,
    so higher values = closer to the camera.
    """
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
            "median_closeness": round(median_c, 4),        # 1 = nearest
            "mean_closeness": round(float(values.mean()), 4),
            "std_closeness": round(float(values.std()), 4),
            "p10_closeness": round(float(np.percentile(values, 10)), 4),
            "p90_closeness": round(float(np.percentile(values, 90)), 4),
            "relative_depth": round(1.0 - median_c, 4),    # 0 = nearest
            "centroid_xy": [int(xs.mean()), int(ys.mean())],
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        })

    results.sort(key=lambda r: r["median_closeness"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["depth_rank"] = rank
    return results


# --------------------------------------------------------------------------- #
# Labelled, depth-shaded image
# --------------------------------------------------------------------------- #
def load_font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf",
                 "DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def label_anchor(mask: np.ndarray, centroid_xy) -> tuple:
    """Pick a point well inside the segment, as close to the centroid as possible.
    (The centroid itself can fall outside non-convex shapes such as roads.)"""
    core = mask
    for px in (15, 8, 3):
        eroded = erode_mask(mask, px)
        if eroded.any():
            core = eroded
            break
    ys, xs = np.nonzero(core)
    cx, cy = centroid_xy
    i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return int(xs[i]), int(ys[i])


def render_labeled_depth_image(image, seg_map, results, out_path: Path,
                               alpha=0.6, cmap_name="turbo",
                               label_min_pixels=500, boundary_px=1):
    """
    Shade each segment by its median closeness (red = near, blue = far with
    'turbo'), draw segment boundaries, label each segment and add a legend.
    Pixels not belonging to any analysed segment are shown greyed out.
    """
    import matplotlib

    cmap = matplotlib.colormaps[cmap_name]
    img = np.asarray(image, dtype=np.float32) / 255.0
    h, w = seg_map.shape

    # Greyed, darkened background for unlabelled / filtered pixels
    grey = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2)
    out = grey * 0.5

    seg_colours = {}
    for r in results:
        m = seg_map == r["segment_id"]
        colour = np.array(cmap(r["median_closeness"])[:3], dtype=np.float32)
        seg_colours[r["segment_id"]] = colour
        out[m] = (1 - alpha) * img[m] + alpha * colour

    # Segment boundaries
    if boundary_px > 0:
        edges = np.zeros((h, w), dtype=bool)
        edges[:, 1:] |= seg_map[:, 1:] != seg_map[:, :-1]
        edges[1:, :] |= seg_map[1:, :] != seg_map[:-1, :]
        edges = dilate_mask(edges, boundary_px - 1) if boundary_px > 1 else edges
        out[edges] = 1.0

    shaded = Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))

    # Fonts and layout scale with image size
    font_size = max(12, w // 90)
    font = load_font(font_size)
    legend_font = load_font(max(12, w // 70))
    pad = max(2, font_size // 4)

    # ---- Legend panel on the right ----
    margin = legend_font.size if hasattr(legend_font, "size") else 14
    bar_w = max(16, w // 60)
    text_w = int(max(legend_font.getlength(t) for t in ("Near", "Far", "Depth")))
    legend_w = bar_w + text_w + margin * 3

    canvas = Image.new("RGB", (w + legend_w, h), (30, 30, 30))
    canvas.paste(shaded, (0, 0))
    draw = ImageDraw.Draw(canvas)

    bar_top, bar_bottom = margin * 3, h - margin * 2
    bar_h = max(1, bar_bottom - bar_top)
    gradient = cmap(np.linspace(1.0, 0.0, bar_h))[:, :3]  # top = near
    bar = np.repeat(gradient[:, None, :], bar_w, axis=1)
    canvas.paste(Image.fromarray((bar * 255).astype(np.uint8)), (w + margin, bar_top))

    tx = w + margin * 2 + bar_w
    draw.text((w + margin, margin), "Depth", fill="white", font=legend_font)
    draw.text((tx, bar_top), "Near", fill="white", font=legend_font)
    draw.text((tx, bar_top + bar_h), "Far", fill="white", font=legend_font, anchor="ls")

    # ---- Segment labels (farthest first so nearer labels are drawn on top) ----
    for r in reversed(results):
        if r["pixel_count"] < label_min_pixels:
            continue
        mask = seg_map == r["segment_id"]
        cx, cy = label_anchor(mask, r["centroid_xy"])
        text = f"#{r['depth_rank']} {r['label']}\nd={r['relative_depth']:.2f}"

        l, t, rgt, b = draw.multiline_textbbox((0, 0), text, font=font, align="center")
        tw, th = rgt - l, b - t
        x0 = int(np.clip(cx - tw / 2, pad, w - tw - pad))
        y0 = int(np.clip(cy - th / 2, pad, h - th - pad))

        outline = tuple(int(c * 255) for c in seg_colours[r["segment_id"]])
        draw.rectangle([x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad],
                       fill=(0, 0, 0), outline=outline, width=max(1, pad // 2))
        draw.multiline_text((x0 - l, y0 - t), text, fill="white",
                            font=font, align="center")

    canvas.save(out_path)
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------------- #
# Other outputs
# --------------------------------------------------------------------------- #
def print_table(results):
    header = f"{'Rank':>4}  {'ID':>3}  {'Label':<15} {'Type':<6} {'Pixels':>8}  " \
             f"{'RelDepth':>8}  {'Median':>7}  {'P10-P90':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['depth_rank']:>4}  {r['segment_id']:>3}  {r['label']:<15} "
              f"{r['category']:<6} {r['pixel_count']:>8}  {r['relative_depth']:>8.3f}  "
              f"{r['median_closeness']:>7.3f}  "
              f"{r['p10_closeness']:.3f}-{r['p90_closeness']:.3f}")


def save_results(results, out_dir: Path, stem: str):
    json_path = out_dir / f"{stem}_segments.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = out_dir / f"{stem}_segments.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    print(f"\nSaved: {json_path}\nSaved: {csv_path}")


def visualise(image, seg_map, closeness, results, out_path: Path):
    """Four-panel matplotlib overview (optional)."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    seg_rgb = np.zeros((*seg_map.shape, 3), dtype=np.float32)
    seg_depth = np.full(seg_map.shape, np.nan, dtype=np.float32)
    for r in results:
        m = seg_map == r["segment_id"]
        seg_rgb[m] = rng.random(3)
        seg_depth[m] = r["median_closeness"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input image")
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(seg_rgb, alpha=0.6)
    axes[0, 1].set_title("Panoptic segmentation")
    im = axes[1, 0].imshow(closeness, cmap="inferno")
    axes[1, 0].set_title("Depth Anything V2 (bright = near)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    im = axes[1, 1].imshow(seg_depth, cmap="inferno", vmin=0, vmax=1)
    axes[1, 1].set_title("Per-segment median closeness")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", help="Path to input image")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                        help="Directory where models are saved/loaded")
    parser.add_argument("--seg-model", default=SEG_MODEL_ID,
                        help="Hub ID or local path")
    parser.add_argument("--depth-model", default=DEPTH_MODEL_ID,
                        help="Hub ID or local path")
    parser.add_argument("--offline", action="store_true",
                        help="Only use local model copies; never download")
    parser.add_argument("--download-only", action="store_true",
                        help="Download and save models, then exit")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-fuse-stuff", action="store_true")
    parser.add_argument("--erosion", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Opacity of depth shading on the labelled image")
    parser.add_argument("--cmap", default="turbo",
                        help="Matplotlib colormap for depth shading")
    parser.add_argument("--label-min-pixels", type=int, default=500,
                        help="Do not draw labels for segments smaller than this")
    parser.add_argument("--visualise", action="store_true",
                        help="Also save the four-panel matplotlib overview")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.download_only:
        load_model_and_processor(args.seg_model, Mask2FormerForUniversalSegmentation,
                                 models_dir)
        load_model_and_processor(args.depth_model, AutoModelForDepthEstimation,
                                 models_dir)
        print("Models are available locally.")
        return

    if not args.image:
        parser.error("an image path is required unless --download-only is used")

    device = get_device()
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    image = load_image(args.image)
    print(f"Loaded image: {image.size[0]}x{image.size[1]}")

    print("Running panoptic segmentation...")
    seg_map, segments_info, id2label = run_panoptic_segmentation(
        image, device, args.seg_model, models_dir, args.threshold,
        fuse_stuff=not args.no_fuse_stuff, offline=args.offline,
    )
    print(f"  Found {len(segments_info)} segments")

    print("Running depth estimation...")
    raw_depth = run_depth_estimation(image, device, args.depth_model, models_dir,
                                     offline=args.offline)
    closeness = normalise(raw_depth)

    print("Computing per-segment depth...\n")
    results = compute_segment_depths(
        seg_map, segments_info, id2label, closeness,
        erosion_px=args.erosion, min_pixels=args.min_pixels,
    )

    print_table(results)
    save_results(results, out_dir, stem)
    np.save(out_dir / f"{stem}_depth_raw.npy", raw_depth)
    np.save(out_dir / f"{stem}_panoptic.npy", seg_map)

    render_labeled_depth_image(
        image, seg_map, results, out_dir / f"{stem}_depth_labeled.png",
        alpha=args.alpha, cmap_name=args.cmap,
        label_min_pixels=args.label_min_pixels,
    )

    if args.visualise:
        visualise(image, seg_map, closeness, results, out_dir / f"{stem}_overview.png")

if __name__ == "__main__":
    main()
