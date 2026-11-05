"""
segdepth
========
Panoptic segmentation (Mask2Former, Cityscapes) combined with Depth Anything V2
depth estimation (relative and/or metric) to estimate the depth of each segment.

Core modules (no nuScenes dependency):
    config         Model IDs, class sets, association settings
    utils          Device selection, image loading, mask morphology
    models         Save/load Hugging Face models to a local directory
    pipeline       PanopticDepthPipeline: segmentation + depth model(s)
    analysis       Per-segment depth statistics (relative and metric)
    depth_scale    Relative/metric conversion, scale fitting, error metrics
    visualisation  Labelled depth image, overview figure, radar overlay
    reporting      Console tables and JSON/CSV output
    cli            Shared command-line model arguments

Optional subpackage:
    segdepth.nusc  nuScenes loading, LiDAR validation and radar association
                   (requires nuscenes-devkit)
"""

from .analysis import compute_segment_depths
from .config import DEFAULT_MODELS_DIR, DEPTH_MODEL_ID, METRIC_DEPTH_MODEL_ID, SEG_MODEL_ID
from .pipeline import PanopticDepthPipeline
from .utils import get_device, load_image, normalise

__all__ = [
    "PanopticDepthPipeline", "compute_segment_depths", "get_device", "load_image",
    "normalise", "SEG_MODEL_ID", "DEPTH_MODEL_ID", "METRIC_DEPTH_MODEL_ID",
    "DEFAULT_MODELS_DIR",
]
