"""
bev.py
------
Bird's-eye-view (top-down) plot of the radar-segment association, drawn in the
ego frame at the camera timestamp (ego x forward, y left; plotted with forward
pointing up and right to the right).

Shows:
  * Ego vehicle outline (approximate), camera position and horizontal field of
    view, and radar mounting positions.
  * Image segments back-projected using the metric depth map from the
    association's fitted scale (a, b): a faint point cloud per candidate
    segment plus a marker at its median position.
  * Radar points at zero elevation, coloured by assigned segment (marker size
    scales with posterior); clutter and occluded points with distinct markers;
    optional radial-velocity arrows.
  * Association lines from each assigned radar point to the depth-map position
    of its best-matching pixel. Line length shows the disagreement between the
    radar range and the depth map.
  * Optional nuScenes 3D box footprints (dashed), colour-coded by category group.

Layout: all text is clipped to the axes and excluded from layout, only boxes
overlapping the view are drawn, overlapping labels are hidden, and the legend
sits outside the axes. Rendering uses the Agg canvas, so no display is needed.

Functions:
  * backproject_pixels     Pixels + metric depth -> ego-frame xyz.
  * segment_bev_points     Sampled ego-frame xy points for each segment.
  * annotation_footprints  nuScenes box footprints in the camera-time ego frame.
  * render_bev             Draw and save the figure.
"""

from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from ..config import EXTENT
from ..depth_scale import closeness_to_metric
from ..utils import erode_mask

# Approximate footprint of the nuScenes ego vehicle (Renault Zoe) relative to
# the ego origin at the rear axle: x from -0.9 to 3.2 m, y within ±0.87 m.
EGO_X, EGO_Y = (-0.9, 3.2), 0.87

# Colours for annotated boxes, by the first part of the nuScenes category name
GT_COLOURS = {
    "vehicle": "black",
    "human": "darkmagenta",
    "movable_object": "dimgrey",
    "static_object": "saddlebrown",
    "animal": "darkolivegreen",
}
GT_DEFAULT_COLOUR = "slategrey"


def _plot_xy(xy):
    """Ego (x forward, y left) -> plot (right, forward)."""
    xy = np.atleast_2d(xy)
    return -xy[:, 1], xy[:, 0]


def backproject_pixels(xs, ys, depth, K, T_ego_from_cam):
    rays = np.linalg.inv(K) @ np.vstack([xs, ys, np.ones_like(xs)]).astype(float)
    pc = rays * np.asarray(depth, dtype=float)[None]          # rays have z = 1
    return (T_ego_from_cam[:3, :3] @ pc + T_ego_from_cam[:3, 3:4]).T


def segment_bev_points(assoc, metric, n_per_seg=400, erosion_px=2,
                       max_range=80.0, candidates_only=True, seed=0):
    """Returns {segment column index: (n, 2) ego-frame xy}."""
    rng = np.random.default_rng(seed)
    out = {}
    cols = assoc.cand_idx if candidates_only else range(assoc.M)
    for j in cols:
        mask = assoc.seg_map == assoc.ids[j]
        core = erode_mask(mask, erosion_px)
        ys, xs = np.nonzero(core if core.any() else mask)
        if len(xs) == 0:
            continue
        if len(xs) > n_per_seg:
            pick = rng.choice(len(xs), n_per_seg, replace=False)
            xs, ys = xs[pick], ys[pick]
        d = metric[ys, xs]
        ok = np.isfinite(d) & (d < max_range)
        if ok.any():
            xyz = backproject_pixels(xs[ok], ys[ok], d[ok], assoc.cam.K, assoc.cam.T_ego_from_cam)
            out[j] = xyz[:, :2]
    return out


def annotation_footprints(nusc, sample_token, cam_token, max_dist=100.0):
    """Returns a list of (corners (4, 2) ego-frame xy, full nuScenes category name)."""
    cam_rec = nusc.get("sample_data", cam_token)
    c_ego = nusc.get("ego_pose", cam_rec["ego_pose_token"])
    q_inv = Quaternion(c_ego["rotation"]).inverse
    out = []
    for ann in nusc.get("sample", sample_token)["anns"]:
        box = nusc.get_box(ann)                                # global frame
        box.translate(-np.array(c_ego["translation"]))
        box.rotate(q_inv)                                      # camera-time ego frame
        if np.linalg.norm(box.center[:2]) > max_dist:
            continue
        out.append((box.bottom_corners()[:2].T, box.name))
    return out


def _add_label(ax, text, x, y, colour, fontsize=7):
    """Text that is clipped to the axes and ignored by the layout engine."""
    import matplotlib.patheffects as pe

    t = ax.annotate(text, (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=fontsize, color=colour, annotation_clip=True, clip_on=True,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                    zorder=6)
    t.set_in_layout(False)
    return t


def _declutter(texts, renderer):
    """Hide labels that overlap a higher-priority label (texts in priority order)."""
    placed = []
    for t in texts:
        bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
        if any(bb.overlaps(p) for p in placed):
            t.set_visible(False)
        else:
            placed.append(bb)


def render_bev(assoc, out_path: Path, gt=None, max_range=60.0, lateral=None,
               show_segments=True, show_velocity=True, vel_scale=1.0,
               moving_thresh=0.5, label_segments="radar", label_gt=False):
    """
    `assoc` is a fitted RadarSegmentAssociator. `gt` is the output of
    annotation_footprints (or None). Velocity arrows show the radial velocity
    over `vel_scale` seconds (radar measures only the line-of-sight component).

    label_segments: "radar" (segments with radar points), "things"
                    (also thing classes without radar), "all" or "none".
    label_gt:       also write category names next to annotated boxes.
    """
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, Rectangle

    cam, geo, extra, pts, A = assoc.cam, assoc.geo, assoc.extra, assoc.pts, assoc.A
    lateral = lateral or 0.6 * max_range
    xlim, ylim = (-lateral, lateral), (-5.0, max_range)
    metric = closeness_to_metric(assoc.closeness, assoc.params.a, assoc.params.b,
                                 assoc.cfg.max_range)

    def in_view(X, Y):
        return bool(np.any((X >= xlim[0]) & (X <= xlim[1]) &
                           (Y >= ylim[0]) & (Y <= ylim[1])))

    # Colours: one per segment that received radar points
    cmap = matplotlib.colormaps["tab20"]
    seg_colour = {j: cmap(k % 20) for k, j in
                  enumerate(j for j in assoc.cand_idx if np.any(A == j))}
    grey = (0.6, 0.6, 0.6, 1.0)

    fig = Figure(figsize=(13, 10), layout="constrained")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    seg_labels, gt_labels = [], []

    # ---- Ego vehicle, camera FOV, radar positions ----
    ax.add_patch(Rectangle((-EGO_Y, EGO_X[0]), 2 * EGO_Y, EGO_X[1] - EGO_X[0],
                           fc="lightgrey", ec="k", lw=1, zorder=2))
    cam_xy = cam.T_ego_from_cam[:2, 3]
    cy = cam.K[1, 2]
    for u in (0, cam.width):
        d = cam.T_ego_from_cam[:3, :3] @ (np.linalg.inv(cam.K) @ np.array([u, cy, 1.0]))
        d = d[:2] / np.linalg.norm(d[:2])
        ax.plot(*_plot_xy(np.vstack([cam_xy, cam_xy + d * 3 * max_range])),
                color="grey", ls=":", lw=1, zorder=1)
    ax.plot(*_plot_xy(cam_xy), "o", color="k", ms=5, zorder=5)
    for chan in np.unique(pts.channel):
        k = int(pts.sweep[pts.channel == chan].min())          # most recent sweep
        ax.plot(*_plot_xy(pts.T_egocam[k][:2, 3]), "^", color="k", ms=6, zorder=5)

    # ---- Ground-truth box footprints (only those overlapping the view) ----
    gt_groups = set()
    for corners, name in gt or []:
        X, Y = _plot_xy(corners)
        if not in_view(X, Y):
            continue
        group = name.split(".")[0]
        colour = GT_COLOURS.get(group, GT_DEFAULT_COLOUR)
        gt_groups.add(group)
        ax.add_patch(Polygon(np.c_[X, Y], closed=True, fill=False,
                             ec=colour, lw=1, ls="--", zorder=2))
        if label_gt:
            gt_labels.append(_add_label(ax, name.split(".")[-1], X.mean(), Y.max(),
                                        colour, fontsize=6))

    # ---- Back-projected segments ----
    if show_segments:
        candidates = []
        for j, xy in segment_bev_points(assoc, metric, max_range=max_range).items():
            c = seg_colour.get(j, grey)
            ax.scatter(*_plot_xy(xy), s=2, color=c, alpha=0.25, lw=0, zorder=1)
            has_radar = j in seg_colour
            if has_radar or assoc.labels[j] in EXTENT:
                mx, my = _plot_xy(np.median(xy, axis=0))
                ax.plot(mx, my, "s", color=c, ms=5, mec="k", mew=0.5, zorder=3)
                wanted = (label_segments == "all"
                          or (label_segments == "things" and (has_radar or assoc.labels[j] in EXTENT))
                          or (label_segments == "radar" and has_radar))
                if wanted:
                    candidates.append((int(np.sum(A == j)), j, mx[0], my[0], c))
        for _, j, mx, my, c in sorted(candidates, key=lambda t: -t[0]):   # priority order
            seg_labels.append(_add_label(ax, f"{assoc.labels[j]} ({assoc.ids[j]})", mx, my, c))

    # ---- Association lines: radar point -> depth-map position of best pixel ----
    xy = geo["xy_ego"]
    idx = np.nonzero(A < assoc.M)[0]
    if len(idx):
        s = extra["best"][idx, A[idx]]
        D = extra["D"][idx, s]
        ok = np.isfinite(D)
        ends = backproject_pixels(geo["u"][idx[ok], s[ok]], geo["v"][idx[ok], s[ok]],
                                  D[ok], cam.K, cam.T_ego_from_cam)[:, :2]
        for i, e in zip(idx[ok], ends):
            ax.plot(*_plot_xy(np.vstack([xy[i], e])), color=seg_colour[A[i]],
                    lw=0.9, alpha=0.8, zorder=3)

    # ---- Radar points ----
    if len(idx):
        ax.scatter(*_plot_xy(xy[idx]), c=[seg_colour[j] for j in A[idx]],
                   s=15 + 45 * assoc.post[idx], edgecolors="k", linewidths=0.5, zorder=4)
    for code, colour in ((assoc.CLUTTER, "red"), (assoc.OCCLUDED, "royalblue")):
        m = A == code
        if m.any():
            ax.scatter(*_plot_xy(xy[m]), marker="x", c=colour, s=25, zorder=4)

    if show_velocity:
        m = np.abs(pts.v_r) > moving_thresh
        if m.any():
            v = pts.v_r[m, None] * geo["n_ego"][m] * vel_scale
            X, Y = _plot_xy(xy[m])
            ax.quiver(X, Y, -v[:, 1], v[:, 0], angles="xy", scale_units="xy", scale=1,
                      width=0.002, color="k", alpha=0.6, zorder=4)

    # ---- Axes, legend, title ----
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Lateral (m, right positive)")
    ax.set_ylabel("Forward (m)")
    ax.grid(True, lw=0.3, alpha=0.5)

    handles = [
        Line2D([], [], marker="o", ls="", mfc="tab:blue", mec="k", label="radar point (assigned)"),
        Line2D([], [], marker="x", ls="", color="red", label="clutter"),
        Line2D([], [], marker="x", ls="", color="royalblue", label="occluded"),
        Line2D([], [], color="tab:blue", label="association\n(to depth-map position)"),
        Line2D([], [], marker="s", ls="", mfc="tab:blue", mec="k", label="segment median"),
        Line2D([], [], marker="o", ls="", color="k", label="camera"),
        Line2D([], [], marker="^", ls="", color="k", label="radar"),
        Line2D([], [], color="grey", ls=":", label="camera FOV"),
    ]
    for group in sorted(gt_groups):
        handles.append(Line2D([], [], color=GT_COLOURS.get(group, GT_DEFAULT_COLOUR), ls="--",
                              label=f"annotated: {group.replace('_', ' ')}"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, fontsize=8, framealpha=0.95)

    s = assoc.summary()
    ax.set_title(f"Radar association (BEV)\n{s['n_assigned']} assigned, "
                 f"{s['n_clutter']} clutter, {s['n_occluded']} occluded | "
                 f"mean posterior {s['mean_posterior']:.2f}", fontsize=10)

    # Layout first, then hide overlapping labels using their on-screen extents
    canvas.draw()
    _declutter(seg_labels + gt_labels, canvas.get_renderer())

    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
