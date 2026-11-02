"""
segdepth
========
Panoptic segmentation (Mask2Former, Cityscapes) combined with relative depth
estimation (Depth Anything V2) to estimate the relative depth of each segment.

Core modules (no nuScenes dependency):
    config         Model IDs, class sets, radar priors, association settings
    utils          Device selection, image loading, mask morphology
    models         Save/load Hugging Face models to a local directory
    pipeline       PanopticDepthPipeline: runs both models on an image
    analysis       Per-segment depth statistics and near-to-far ranking
    depth_scale    Affine inverse-depth fitting and metric conversion
    visualisation  Labelled depth image, overview figure, radar overlay
    reporting      Console tables and JSON/CSV output

Optional subpackage:
    segdepth.nusc  nuScenes loading, LiDAR validation and radar association
                   (requires nuscenes-devkit)
"""

from .analysis import compute_segment_depths
from .config import DEFAULT_MODELS_DIR, DEPTH_MODEL_ID, SEG_MODEL_ID
from .pipeline import PanopticDepthPipeline
from .utils import get_device, load_image, normalise

__all__ = [
    "PanopticDepthPipeline",
    "compute_segment_depths",
    "get_device",
    "load_image",
    "normalise",
    "SEG_MODEL_ID",
    "DEPTH_MODEL_ID",
    "DEFAULT_MODELS_DIR",
]
