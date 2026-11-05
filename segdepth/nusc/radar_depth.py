"""
radar_depth.py
--------------
Dense metric depth for the whole image from the radar association, and
comparison against projected LiDAR.

The depth map (relative or metric closeness, c ≈ a / Z + b) is converted to
metres using radar points as anchors. An anchor is a confidently assigned radar
point: its range gives the camera depth Z, its best-matching pixel gives c
(read from the association's per-column segment depth map).

Estimates (all LiDAR-free except the oracle):
  ground_plane         affine fit to road pixels only
  radar_global         robust affine fit to radar anchors only (scale-only if
                       too few anchors or too little depth spread)
  radar_ground_opt     (a, b) from the association optimiser (radar + road)
  radar_segment        radar_ground_opt plus a per-segment inverse-depth offset
                       for segments with anchors (shrunk, capped)
  metric_model         raw Depth Anything V2 metric output (if loaded)
  metric_radar_scaled  metric output times the radar/metric median ratio
  lidar_affine_oracle  affine fit to the LiDAR itself (upper bound for any
                       single global scale; uses the evaluation data)

Comparison is on a common set of LiDAR pixels where every estimate is finite:
AbsRel, RMSE, delta<1.25 and median scale, by LiDAR distance bin and segment
group. Summed statistics are returned so results can be pooled across frames.
Radar anchors are also checked directly against LiDAR in the same column and
segment, which tests the association itself.

Contents:
  RadarDepthConfig, radar_anchors, fit_radar_scale, segment_corrected_depth,
  estimate_depth_maps, add_lidar_oracle, anchors_vs_lidar, compare_with_lidar,
  pool_depth_sums, format_depth_table, render_depth_comparison
"""

from dataclasses import dataclass

import numpy as np

from ..config import EXTENT
from ..depth_scale import closeness_to_metric, fit_affine_inverse

METHOD_ORDER = ("ground_plane", "radar_global", "radar_ground_opt", "radar_segment",
                "metric_model", "metric_radar_scaled", "lidar_affine_oracle")


@dataclass
class RadarDepthConfig:
    min_posterior: float = 0.8         # anchors must be confidently assigned
    anchor_classes: str = "things"     # "things" or "all"
    correct_classes: str = "things"    # segments that receive per-segment offsets
    min_anchors_affine: int = 8        # fewer anchors -> fit scale only
    min_inv_spread: float = 0.01       # std of 1/Z (1/m) needed to fit the shift too
    shrink_k: float = 2.0              # offset shrinkage: n / (n + k)
    max_rel_change: float = 0.5        # cap on per-segment depth change (must be < 1)
    max_depth: float = 200.0
    eval_max_depth: float = 80.0
    main_method: str = "radar_segment"
    bins: tuple = (0, 10, 20, 40, 80)


def weighted_median(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


# --------------------------------------------------------------------------- #
# Anchors and fitting
# --------------------------------------------------------------------------- #
def radar_anchors(assoc, cfg):
    A, extra, geo = assoc.A, assoc.extra, assoc.geo
    idx = np.nonzero(A < assoc.M)[0]
    if cfg.anchor_classes == "things":
        keep = np.array([assoc.labels[j] in EXTENT for j in A[idx]], dtype=bool)
        idx = idx[keep]
    idx = idx[assoc.post[idx] >= cfg.min_posterior]
    s = extra["best"][idx, A[idx]]
    u, v = geo["u"][idx, s], geo["v"][idx, s]
    c = assoc.closeness_assoc[v, u]
    Z = geo["Z"][idx, s]
    ok = np.isfinite(c) & np.isfinite(Z) & (Z > 0)
    return {"point": idx[ok], "col": A[idx[ok]], "c": c[ok], "Z": Z[ok],
            "w": assoc.post[idx[ok]], "u": u[ok], "v": v[ok]}


def fit_radar_scale(anchors, b_init, cfg):
    """Returns (a, b, mode) or None."""
    c, Z, w = anchors["c"], anchors["Z"], anchors["w"]
    if len(c) == 0:
        return None
    if len(c) >= cfg.min_anchors_affine and np.std(1.0 / Z) >= cfg.min_inv_spread:
        a, b = fit_affine_inverse(c, Z, w)
        if a > 0:
            return a, b, "affine"
    a = weighted_median((c - b_init) * Z, w)
    return (a, b_init, "scale_only") if a > 0 else None


def segment_corrected_depth(closeness, a, b, assoc, anchors, cfg):
    """Global (a, b) plus per-segment inverse-depth offsets from radar anchors."""
    with np.errstate(invalid="ignore"):
        inv = (np.asarray(closeness, float) - b) / a
    corrections = []
    for j in np.unique(anchors["col"]):
        lab = assoc.labels[j]
        if cfg.correct_classes == "things" and lab not in EXTENT:
            continue
        m = anchors["col"] == j
        w = anchors["w"][m]
        r = 1.0 / anchors["Z"][m] - (anchors["c"][m] - b) / a
        n_eff = float(w.sum())
        rj = weighted_median(r, w) * n_eff / (n_eff + cfg.shrink_k)

        mask = assoc.seg_map == assoc.ids[j]
        vals = inv[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        im = float(np.median(vals))
        if im <= 0:
            continue
        x = cfg.max_rel_change
        rj = float(np.clip(rj, im / (1 + x) - im, im / (1 - x) - im))
        inv[mask] += rj
        corrections.append({"segment_id": int(assoc.ids[j]), "label": lab,
                            "anchors": int(m.sum()),
                            "depth_before_m": round(1.0 / im, 2),
                            "depth_after_m": round(1.0 / (im + rj), 2)})
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = np.where(inv > 1.0 / cfg.max_depth, 1.0 / np.maximum(inv, 1e-12), np.inf)
    return depth, corrections


def estimate_depth_maps(assoc, ground, cfg, metric_depth=None, metric_valid=None):
    """Returns (maps: name -> HxW depth in metres, anchors, info). LiDAR-free."""
    c, p = assoc.closeness, assoc.params
    anchors = radar_anchors(assoc, cfg)
    maps, info = {}, {"n_anchors": int(len(anchors["Z"]))}

    c_g, Z_g = ground
    if len(c_g) >= 20:
        ga, gb = fit_affine_inverse(c_g, Z_g)
        if ga > 0:
            maps["ground_plane"] = closeness_to_metric(c, ga, gb, cfg.max_depth)
            info["ground_plane"] = {"a": round(ga, 5), "b": round(gb, 5)}

    fit = fit_radar_scale(anchors, p.b, cfg)
    if fit:
        maps["radar_global"] = closeness_to_metric(c, fit[0], fit[1], cfg.max_depth)
        info["radar_global"] = {"a": round(fit[0], 5), "b": round(fit[1], 5), "mode": fit[2]}

    maps["radar_ground_opt"] = closeness_to_metric(c, p.a, p.b, cfg.max_depth)
    maps["radar_segment"], info["segment_corrections"] = segment_corrected_depth(
        c, p.a, p.b, assoc, anchors, cfg)

    if metric_depth is not None:
        md = np.where(metric_valid, metric_depth, np.inf)
        maps["metric_model"] = md
        ma = md[anchors["v"], anchors["u"]]
        ok = np.isfinite(ma) & (ma > 0)
        if ok.any():
            s = weighted_median(anchors["Z"][ok] / ma[ok], anchors["w"][ok])
            maps["metric_radar_scaled"] = md * s
            info["metric_radar_scale"] = round(s, 4)
    return maps, anchors, info


def add_lidar_oracle(maps, closeness, u, v, z, cfg):
    c = closeness[v, u]
    ok = np.isfinite(c) & (z > 0)
    if ok.sum() >= 10:
        a, b = fit_affine_inverse(c[ok], z[ok])
        if a > 0:
            maps["lidar_affine_oracle"] = closeness_to_metric(closeness, a, b, cfg.max_depth)


# --------------------------------------------------------------------------- #
# Comparison with LiDAR
# --------------------------------------------------------------------------- #
def _summary(p, g):
    if len(g) == 0:
        return None
    ratio = np.maximum(p / g, g / p)
    return {"n": int(len(g)),
            "abs_rel": round(float(np.mean(np.abs(p - g) / g)), 4),
            "rmse_m": round(float(np.sqrt(np.mean((p - g) ** 2))), 3),
            "delta_1_25": round(float(np.mean(ratio < 1.25)), 4),
            "median_scale_gt_over_pred": round(float(np.median(g / p)), 4)}


def _sums(p, g):
    ratio = np.maximum(p / g, g / p) if len(g) else np.zeros(0)
    return {"n": int(len(g)), "abs_rel": float(np.sum(np.abs(p - g) / g)) if len(g) else 0.0,
            "sq": float(np.sum((p - g) ** 2)), "delta": int(np.sum(ratio < 1.25))}


def compare_with_lidar(maps, u, v, z, seg_map, results, corrected_ids, cfg):
    preds = {k: np.asarray(m, float)[v, u] for k, m in maps.items()}
    common = np.isfinite(z) & (z > 0) & (z <= cfg.eval_max_depth)
    coverage = {}
    for k, p in preds.items():
        ok = np.isfinite(p) & (p > 0)
        coverage[k] = round(float(ok[common].mean()), 3) if common.any() else None
        common &= ok

    sid = seg_map[v, u]
    things = [r["segment_id"] for r in results if r["category"] == "thing"]
    labelled = [r["segment_id"] for r in results]
    corrected = list(corrected_ids)
    groups = {"thing": common & np.isin(sid, things),
              "stuff": common & np.isin(sid, labelled) & ~np.isin(sid, things),
              "radar_corrected": common & np.isin(sid, corrected),
              "uncorrected": common & ~np.isin(sid, corrected)}
    edges = cfg.bins
    bin_masks = {f"{lo}-{hi}m": common & (z >= lo) & (z < hi)
                 for lo, hi in zip(edges[:-1], edges[1:])}

    report = {"n_common": int(common.sum()), "coverage": coverage, "methods": {}}
    sums = {}
    for k, p in preds.items():
        res = {"overall": _summary(p[common], z[common]), "groups": {}, "bins": {}}
        for g, m in groups.items():
            s = _summary(p[m], z[m])
            res["groups"][g] = None if s is None else {f: s[f] for f in ("n", "abs_rel", "rmse_m")}
        for name, m in {"overall": common, **bin_masks}.items():
            if name != "overall":
                s = _summary(p[m], z[m])
                res["bins"][name] = None if s is None else {f: s[f] for f in ("n", "abs_rel", "rmse_m")}
            for f, val in _sums(p[m], z[m]).items():
                sums[f"{k}|{name}|{f}"] = val
        report["methods"][k] = res
    report["sums"] = sums
    return report


def anchors_vs_lidar(assoc, anchors, u, v, z, half_width=4, min_points=2,
                     rel_tol=0.1, abs_tol=1.0):
    """Radar anchor depth vs median LiDAR depth of the same segment within a few
    columns of the anchor. Tests the association, independent of annotations."""
    lid_sid = assoc.seg_map_assoc[v, u]
    rows = []
    for k in range(len(anchors["Z"])):
        j = anchors["col"][k]
        m = (np.abs(u - anchors["u"][k]) <= half_width) & (lid_sid == assoc.ids[j])
        if m.sum() < min_points:
            continue
        zl, zr = float(np.median(z[m])), float(anchors["Z"][k])
        rows.append({"point": int(anchors["point"][k]), "label": assoc.labels[j],
                     "radar_m": round(zr, 2), "lidar_m": round(zl, 2),
                     "error_m": round(zr - zl, 2),
                     "agrees": bool(abs(zr - zl) <= max(abs_tol, rel_tol * zl))})
    out = {"n_anchors": int(len(anchors["Z"])), "n_checked": len(rows)}
    if rows:
        err = np.array([r["error_m"] for r in rows])
        lid = np.array([r["lidar_m"] for r in rows])
        out.update(agreement_rate=round(float(np.mean([r["agrees"] for r in rows])), 3),
                   median_abs_error_m=round(float(np.median(np.abs(err))), 2),
                   median_rel_error=round(float(np.median(np.abs(err) / lid)), 3),
                   points=rows)
    return out


def pool_depth_sums(sums):
    nested = {}
    for key, val in sums.items():
        m, part, f = key.split("|")
        nested.setdefault(m, {}).setdefault(part, {})[f] = val
    out = {}
    for m, parts in nested.items():
        out[m] = {}
        for part, s in parts.items():
            n = s.get("n", 0)
            out[m][part] = None if not n else {
                "n": n, "abs_rel": round(s["abs_rel"] / n, 4),
                "rmse_m": round(float(np.sqrt(s["sq"] / n)), 3),
                "delta_1_25": round(s["delta"] / n, 4)}
    return out


def format_depth_table(report):
    methods = [m for m in METHOD_ORDER if m in report["methods"]]
    if not methods:
        return "(no estimates)"
    bins = list(report["methods"][methods[0]]["bins"].keys())
    head = (f"{'Method':<22}{'AbsRel':>8}{'RMSE m':>8}{'d<1.25':>8}{'Scale':>7}"
            + "".join(f"{b:>10}" for b in bins))
    lines = [head, "-" * len(head)]
    for m in methods:
        r, s = report["methods"][m], report["methods"][m]["overall"]
        if s is None:
            lines.append(f"{m:<22}  (no points)")
            continue
        line = (f"{m:<22}{s['abs_rel']:>8.3f}{s['rmse_m']:>8.2f}{s['delta_1_25']:>8.3f}"
                f"{s['median_scale_gt_over_pred']:>7.3f}")
        for b in bins:
            bb = r["bins"][b]
            line += f"{bb['abs_rel']:>10.3f}" if bb else f"{'-':>10}"
        lines.append(line)
    lines.append("(bin columns: AbsRel by LiDAR distance)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def render_depth_comparison(image, maps, anchors, u, v, z, report, out_path,
                            main="radar_segment", vmax=60.0, max_scatter=4000, seed=0):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    main = main if main in maps else "radar_ground_opt"
    img = np.asarray(image)
    fig = Figure(figsize=(19, 10))
    FigureCanvasAgg(fig)
    axs = fig.subplots(2, 3)

    ax = axs[0, 0]
    ax.imshow(img)
    sc = ax.scatter(u, v, c=z, s=1.5, cmap="turbo_r", vmin=0, vmax=vmax, lw=0)
    if len(anchors["u"]):
        ax.scatter(anchors["u"], anchors["v"], s=40, facecolors="none",
                   edgecolors="white", linewidths=1.2)
    ax.set_title("LiDAR depth (dots) and radar anchors (circles)")
    fig.colorbar(sc, ax=ax, fraction=0.03, label="depth (m)")

    ax = axs[0, 1]
    d = np.where(np.isfinite(maps[main]), maps[main], np.nan)
    im = ax.imshow(d, cmap="turbo_r", vmin=0, vmax=vmax)
    ax.set_title(f"Estimated depth: {main}")
    fig.colorbar(im, ax=ax, fraction=0.03, label="depth (m)")

    ax = axs[0, 2]
    p = maps[main][v, u]
    ok = np.isfinite(p) & (z > 0)
    ax.imshow(img.mean(axis=2), cmap="gray", vmin=0, vmax=255)
    sc = ax.scatter(u[ok], v[ok], c=(p[ok] - z[ok]) / z[ok], s=2, cmap="coolwarm",
                    vmin=-0.5, vmax=0.5, lw=0)
    ax.set_title(f"Relative error vs LiDAR: {main} (red = too far)")
    fig.colorbar(sc, ax=ax, fraction=0.03, label="(pred - LiDAR) / LiDAR")
    for a_ in axs[0]:
        a_.axis("off")

    ax = axs[1, 0]
    rng = np.random.default_rng(seed)
    shown = []
    for name, colour in (("ground_plane", "tab:grey"), ("lidar_affine_oracle", "tab:green"),
                         (main, "tab:red")):
        if name not in maps or name in shown:
            continue
        shown.append(name)
        pp = maps[name][v, u]
        idx = np.flatnonzero(np.isfinite(pp) & (z > 0))
        if len(idx) > max_scatter:
            idx = rng.choice(idx, max_scatter, replace=False)
        ax.scatter(z[idx], pp[idx], s=2, alpha=0.4, color=colour, label=name, lw=0)
    lim = vmax + 20
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("LiDAR depth (m)")
    ax.set_ylabel("Estimated depth (m)")
    ax.legend(markerscale=5, fontsize=8)
    ax.set_title("Estimated vs LiDAR depth")

    methods = [m for m in METHOD_ORDER if m in report["methods"]]
    ax = axs[1, 1]
    if methods:
        bins = list(report["methods"][methods[0]]["bins"].keys())
        x, width = np.arange(len(bins)), 0.8 / len(methods)
        for k, m in enumerate(methods):
            vals = [(b or {}).get("abs_rel", np.nan) for b in report["methods"][m]["bins"].values()]
            ax.bar(x - 0.4 + width * (k + 0.5), vals, width, label=m)
        ax.set_xticks(x, bins)
        ax.legend(fontsize=7)
    ax.set_ylabel("AbsRel")
    ax.set_title("Error by LiDAR distance")

    ax = axs[1, 2]
    ax.axis("off")
    rows = []
    for m in methods:
        s = report["methods"][m]["overall"]
        rows.append([m] + (["-"] * 4 if s is None else
                           [f"{s['abs_rel']:.3f}", f"{s['rmse_m']:.2f}",
                            f"{s['delta_1_25']:.3f}", f"{s['median_scale_gt_over_pred']:.3f}"]))
    if rows:
        t = ax.table(cellText=rows, colLabels=["method", "AbsRel", "RMSE (m)", "δ<1.25", "scale"],
                     loc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.6)
    ax.set_title(f"Overall, {report['n_common']} common LiDAR points")

    fig.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
