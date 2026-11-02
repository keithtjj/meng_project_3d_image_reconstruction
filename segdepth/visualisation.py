"""
visualisation.py
----------------
Image outputs.

Functions:
  * render_labeled_depth_image
        Shades each segment by its median closeness, draws segment boundaries,
        labels segments with depth rank, class and relative depth, and adds a
        Near/Far colour legend. Unanalysed pixels are greyed out.
  * render_overview
        Four-panel matplotlib figure: input, segmentation, depth map and
        per-segment depth.
  * render_association
        Overlay of optimised radar associations: estimated reflection points
        joined to their zero-elevation projections, plus clutter and
        occluded points.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils import dilate_mask, erode_mask


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


def label_anchor(mask, centroid_xy):
    """A point well inside the segment, as close to the centroid as possible.
    The centroid itself can fall outside non-convex shapes such as roads."""
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
    import matplotlib

    cmap = matplotlib.colormaps[cmap_name]
    img = np.asarray(image, dtype=np.float32) / 255.0
    h, w = seg_map.shape

    grey = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2)
    out = grey * 0.5

    seg_colours = {}
    for r in results:
        m = seg_map == r["segment_id"]
        colour = np.array(cmap(r["median_closeness"])[:3], dtype=np.float32)
        seg_colours[r["segment_id"]] = colour
        out[m] = (1 - alpha) * img[m] + alpha * colour

    if boundary_px > 0:
        edges = np.zeros((h, w), dtype=bool)
        edges[:, 1:] |= seg_map[:, 1:] != seg_map[:, :-1]
        edges[1:, :] |= seg_map[1:, :] != seg_map[:-1, :]
        edges = dilate_mask(edges, boundary_px - 1)
        out[edges] = 1.0

    shaded = Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))

    font = load_font(max(12, w // 90))
    legend_font = load_font(max(12, w // 70))
    font_px = getattr(font, "size", 12)
    pad = max(2, font_px // 4)

    # ---- Legend panel ----
    margin = getattr(legend_font, "size", 14)
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
    far_h = draw.textbbox((0, 0), "Far", font=legend_font)[3]
    draw.text((w + margin, margin), "Depth", fill="white", font=legend_font)
    draw.text((tx, bar_top), "Near", fill="white", font=legend_font)
    draw.text((tx, bar_top + bar_h - far_h), "Far", fill="white", font=legend_font)

    # ---- Segment labels (farthest first so nearer labels are on top) ----
    for r in reversed(results):
        if r["pixel_count"] < label_min_pixels:
            continue
        cx, cy = label_anchor(seg_map == r["segment_id"], r["centroid_xy"])
        text = f"#{r['depth_rank']} {r['label']}\nd={r['relative_depth']:.2f}"

        l, t, rgt, b = draw.multiline_textbbox((0, 0), text, font=font, align="center")
        tw, th = rgt - l, b - t
        x0 = int(np.clip(cx - tw / 2, pad, w - tw - pad))
        y0 = int(np.clip(cy - th / 2, pad, h - th - pad))

        outline = tuple(int(c * 255) for c in seg_colours[r["segment_id"]])
        draw.rectangle([x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad],
                       fill=(0, 0, 0), outline=outline, width=max(1, pad // 2))
        draw.multiline_text((x0 - l, y0 - t), text, fill="white", font=font, align="center")

    canvas.save(out_path)
    print(f"Saved: {out_path}")


def render_overview(image, seg_map, closeness, results, out_path: Path):
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


def render_association(image, assoc, out_path: Path, confident=0.8, show_arcs=False):
    """
    `assoc` is a fitted RadarSegmentAssociator.

    Assigned points: a line from the zero-elevation projection (free end) to the
    estimated reflection pixel (circle), i.e. the elevation correction implied by
    the association. Green = posterior > `confident`, amber = uncertain.
    Red crosses: clutter. Blue crosses: occluded.
    show_arcs: also draw each point's arc of valid elevation hypotheses (grey),
    at the centre azimuth sample, showing the full candidate region.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    geo, extra, sc = assoc.geo, assoc.extra, assoc.center
    n_az, n_elev = assoc.cfg.n_az, assoc.cfg.n_elev
    arc_idx = np.arange(n_elev) * n_az + n_az // 2      # centre-azimuth samples

    def cross(u, v, colour):
        draw.line([(u - 4, v - 4), (u + 4, v + 4)], fill=colour, width=2)
        draw.line([(u - 4, v + 4), (u + 4, v - 4)], fill=colour, width=2)

    if show_arcs:
        for i in range(len(assoc.A)):
            ok = geo["valid"][i, arc_idx]
            pts = list(zip(geo["u"][i, arc_idx][ok], geo["v"][i, arc_idx][ok]))
            if len(pts) >= 2:
                draw.line(pts, fill=(170, 170, 170), width=1)

    for i, j in enumerate(assoc.A):
        u0, v0, ok0 = geo["u"][i, sc], geo["v"][i, sc], geo["valid"][i, sc]
        if j < assoc.M:
            s = extra["best"][i, j]
            u, v = geo["u"][i, s], geo["v"][i, s]
            colour = (0, 230, 0) if assoc.post[i] > confident else (255, 190, 0)
            if ok0:
                draw.line([(u0, v0), (u, v)], fill=colour, width=1)
            draw.ellipse([u - 4, v - 4, u + 4, v + 4], outline=colour, width=2)
        elif ok0:
            cross(u0, v0, (255, 40, 40) if j == assoc.CLUTTER else (120, 140, 255))

    img.save(out_path)
    print(f"Saved: {out_path}")

