"""
cli.py
------
Command-line helpers shared by run_image.py and run_nuscenes.py.

Functions:
  * add_model_args   Model selection arguments (--depth, model IDs, offline mode).
  * model_ids        (relative_id, metric_id) for the chosen --depth; either may be None.
  * build_pipeline   PanopticDepthPipeline from the arguments.
  * download_models  Fetch the selected models into the local models directory.
  * apply_overrides  Apply 'PREFIX.FIELD=VALUE' strings to configuration dataclasses.
  * parse_samples    Parse keyframe selections: 'all', '5', '0-9', '0,4,10-12'.
  * metric_scale     argparse type for --metric-scale (a positive number or 'lidar').
"""

import argparse
import difflib
from dataclasses import fields

from .config import DEFAULT_MODELS_DIR, DEPTH_MODEL_ID, METRIC_DEPTH_MODEL_ID, SEG_MODEL_ID


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def add_model_args(p):
    g = p.add_argument_group("models")
    g.add_argument("--depth", choices=("relative", "metric", "both"), default="relative",
                   help="depth model(s) to run (default: %(default)s)")
    g.add_argument("--relative-model", default=DEPTH_MODEL_ID, metavar="ID",
                   help="relative depth model, Hub ID or local path")
    g.add_argument("--metric-model", default=METRIC_DEPTH_MODEL_ID, metavar="ID",
                   help="metric depth model, Hub ID or local path")
    g.add_argument("--seg-model", default=SEG_MODEL_ID, metavar="ID",
                   help="panoptic segmentation model, Hub ID or local path")
    g.add_argument("--seg-threshold", type=float, default=0.5,
                   help="segment score threshold (default: %(default)s)")
    g.add_argument("--no-fuse-stuff", action="store_true",
                   help="keep separate instances of stuff classes")
    g.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                   help="local model directory (default: %(default)s)")
    g.add_argument("--offline", action="store_true", help="only use local model copies")
    return g


def model_ids(args):
    relative = args.relative_model if args.depth in ("relative", "both") else None
    metric = args.metric_model if args.depth in ("metric", "both") else None
    return relative, metric


def build_pipeline(args):
    from .pipeline import PanopticDepthPipeline

    relative, metric = model_ids(args)
    pipeline = PanopticDepthPipeline(
        models_dir=args.models_dir, seg_model_id=args.seg_model,
        depth_model_id=relative, metric_model_id=metric, offline=args.offline,
        threshold=args.seg_threshold, fuse_stuff=not args.no_fuse_stuff,
    )
    print(f"Using device: {pipeline.device}")
    return pipeline


def download_models(args):
    from transformers import AutoModelForDepthEstimation, Mask2FormerForUniversalSegmentation

    from .models import load_model_and_processor

    load_model_and_processor(args.seg_model, Mask2FormerForUniversalSegmentation, args.models_dir)
    for model_id in model_ids(args):
        if model_id:
            load_model_and_processor(model_id, AutoModelForDepthEstimation, args.models_dir)
    print("Models are available locally.")


# --------------------------------------------------------------------------- #
# Configuration overrides
# --------------------------------------------------------------------------- #
def _convert(raw, current, key):
    if isinstance(current, bool):
        v = raw.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key}: expected true or false, got {raw!r}")
    if isinstance(current, (int, float, str)):
        try:
            return type(current)(raw)
        except ValueError:
            raise ValueError(f"{key}: cannot parse {raw!r} as {type(current).__name__}") from None
    if isinstance(current, (tuple, list, set, frozenset)):
        items = [x.strip() for x in raw.split(",") if x.strip()]
        proto = next(iter(current), "")
        if isinstance(proto, (int, float)) and not isinstance(proto, bool):
            items = [type(proto)(float(x)) for x in items]
        return type(current)(items)
    raise ValueError(f"{key} cannot be set from the command line")


def apply_overrides(items, targets):
    """
    items:   'PREFIX.FIELD=VALUE' strings; list values are comma-separated.
    targets: {prefix: dataclass instance}, modified in place.
    Raises ValueError with a readable message.
    """
    for item in items:
        key, sep, raw = item.partition("=")
        prefix, dot, name = key.strip().partition(".")
        if not sep or not dot:
            raise ValueError(f"expected PREFIX.FIELD=VALUE, got {item!r}")
        if prefix not in targets:
            raise ValueError(f"unknown prefix {prefix!r}; use "
                             + " or ".join(f"{p}." for p in targets))
        cfg = targets[prefix]
        names = [f.name for f in fields(cfg)]
        if name not in names:
            hint = difflib.get_close_matches(name, names, n=1)
            raise ValueError(f"{type(cfg).__name__} has no field {name!r}"
                             + (f" (did you mean {prefix}.{hint[0]}?)" if hint else ""))
        setattr(cfg, name, _convert(raw, getattr(cfg, name), key))


# --------------------------------------------------------------------------- #
# Argument types
# --------------------------------------------------------------------------- #
def parse_samples(spec, n):
    """'all', '5', '0-9', '-9', '10-', '0,4,10-12' -> sorted keyframe indices in [0, n)."""
    chosen = set()
    for part in str(spec).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "all":
            chosen.update(range(n))
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            indices = range(int(lo) if lo else 0, (int(hi) if hi else n - 1) + 1)
        else:
            indices = [int(part)]
        bad = [i for i in indices if not 0 <= i < n]
        if bad:
            raise ValueError(f"keyframe {bad[0]} out of range 0-{n - 1}")
        chosen.update(indices)
    if not chosen:
        raise ValueError(f"--samples {spec!r} selects no keyframes")
    return sorted(chosen)


def metric_scale(text):
    if text.strip().lower() == "lidar":
        return "lidar"
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive number or 'lidar'") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value
