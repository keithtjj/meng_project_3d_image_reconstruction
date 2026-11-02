"""
config.py
---------
Central configuration for the segdepth package. Contains values only, no logic.

Contents:
  * Default Hugging Face model IDs (segmentation, relative depth, metric depth)
    and the local model directory.
  * METRIC_SATURATION: fraction of a metric model's max_depth above which its
    output is treated as invalid (sky / beyond range).
  * Cityscapes "stuff" classes, used to fuse stuff regions during panoptic
    post-processing and to label segments as stuff or thing.
  * RADAR_IGNORE: classes that are never candidate segments for radar points.
  * EXTENT: approximate maximum length of 'thing' classes, used by the
    optional extent-consistency term in the radar association.
  * AssocConfig: tunable parameters for the depth-only radar association.
"""

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
SEG_MODEL_ID = "facebook/mask2former-swin-large-cityscapes-panoptic"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"            # relative
METRIC_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
# Alternatives: ...-Metric-Outdoor-Small-hf / -Base-hf (max 80 m),
#               ...-Metric-Indoor-*-hf (max 20 m)
DEFAULT_MODELS_DIR = "models"
METRIC_SATURATION = 0.98

# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
CITYSCAPES_STUFF = frozenset({
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
})
RADAR_IGNORE = frozenset({"sky", "road", "sidewalk", "terrain"})

EXTENT = {  # approximate maximum length (m) of 'thing' classes
    "car": 5.5, "truck": 12.0, "bus": 14.0, "train": 30.0,
    "motorcycle": 2.5, "bicycle": 2.2, "rider": 2.2, "person": 1.2,
}


# --------------------------------------------------------------------------- #
# Depth-only association settings
# --------------------------------------------------------------------------- #
@dataclass
class AssocConfig:
    # Elevation / azimuth hypotheses (odd counts so a centre sample exists)
    beam_half_elev_deg: float = 10.0   # check against the ARS 408 datasheet
    n_elev: int = 25
    az_sigma_deg: float = 0.6
    n_az: int = 5
    h_min: float = -0.3                # plausible reflection height above ground (m)
    h_max: float = 4.0
    max_range: float = 250.0
    # Depth comparison
    sigma_range: float = 0.4           # radar range noise (m)
    sigma_inv_init: float = 0.01       # inverse-depth error of the depth map (1/m)
    sigma_inv_bounds: tuple = (0.005, 0.05)
    outlier_eps: float = 0.05
    occlusion_rel: float = 0.25        # occluded if map depth < radar depth by this fraction
    occlusion_abs: float = 2.0         # ... and by at least this many metres
    # Candidate segments and boundary robustness
    exclude_classes: frozenset = RADAR_IGNORE
    profile_depth: bool = True         # per-column segment depth + grown masks
    mask_dilate_px: int = 6
    profile_erode_px: int = 3
    # Extent consistency (geometric coupling between points on one object)
    use_extent: bool = True
    sigma_extent: float = 0.5
    # Null hypotheses
    rho_clutter: float = 0.05
    rho_occluded: float = 0.3
    # Parameter priors
    sigma_closeness: float = 0.03
    ground_weight: float = 50.0        # total weight of ground-plane anchors
    sigma_yaw_deg: float = 0.5
    yaw_steps_deg: tuple = (-0.5, -0.2, 0.2, 0.5)
    # Which parameters the optimiser may refit
    fit_depth_scale: bool = True
    fit_sigma: bool = False            # refitting to own assignments causes overconfidence
    fit_yaw: bool = True
    # Optimisation
    cand_margin: float = 25.0          # only consider hypotheses within this many nats
    max_iter: int = 15
    tol: float = 1e-4
    seed: int = 0
