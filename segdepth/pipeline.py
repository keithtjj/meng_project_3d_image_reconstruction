"""
pipeline.py
-----------
PanopticDepthPipeline: loads Mask2Former (panoptic segmentation) and up to two
Depth Anything V2 models once, then runs them on any number of images.

Depth models:
  * relative  affine-invariant inverse depth, no scale (default model).
  * metric    depth in metres (Metric-Outdoor: up to 80 m, Metric-Indoor: 20 m).
Either may be disabled, but at least one is required. The type of each
checkpoint is read from its config (depth_estimation_type), not its name.

Classes:
  * DepthEstimator         One Depth Anything V2 model; returns its output
                           resized to the image.
  * PanopticDepthPipeline  Runs segmentation and the depth model(s).
        __call__(image) returns a dict:
          seg_map, segments_info
          closeness         [0, 1], higher = nearer. From the relative model,
                            or from the metric model if relative is disabled.
          raw_depth         relative model output (None if disabled)
          metric_depth      metres (None without a metric model)
          metric_valid      bool mask; False where saturated (sky, beyond range)
          metric_closeness  normalised inverse metric depth in [0, 1]
          metric_affine     (a, b) such that
                            closeness_to_metric(metric_closeness, a, b)
                            reproduces metric_depth exactly on valid pixels
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation, Mask2FormerForUniversalSegmentation

from .config import (CITYSCAPES_STUFF, DEFAULT_MODELS_DIR, DEPTH_MODEL_ID,
                     METRIC_SATURATION, SEG_MODEL_ID)
from .depth_scale import inverse_depth_closeness
from .models import load_model_and_processor
from .utils import get_device, normalise


class DepthEstimator:
    def __init__(self, model_id, models_dir, device, offline=False):
        self.model_id, self.device = model_id, device
        self.processor, self.model = load_model_and_processor(
            model_id, AutoModelForDepthEstimation, models_dir, offline)
        self.model.to(device).eval()
        cfg = self.model.config
        self.kind = getattr(cfg, "depth_estimation_type", None) or "relative"
        self.max_depth = float(getattr(cfg, "max_depth", 80.0)) if self.kind == "metric" else None

    @torch.inference_mode()
    def __call__(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pred = self.model(**inputs).predicted_depth                     # (1, h, w)
        return F.interpolate(
            pred.unsqueeze(1), size=image.size[::-1], mode="bilinear", align_corners=False,
        ).squeeze().float().cpu().numpy()


class PanopticDepthPipeline:
    def __init__(self, device=None, models_dir=DEFAULT_MODELS_DIR,
                 seg_model_id=SEG_MODEL_ID, depth_model_id=DEPTH_MODEL_ID,
                 metric_model_id=None, offline=False, threshold=0.5, fuse_stuff=True,
                 metric_saturation=METRIC_SATURATION):
        if depth_model_id is None and metric_model_id is None:
            raise ValueError("At least one of the relative or metric depth models is required.")
        self.device = device or get_device()
        self.threshold = threshold
        self.metric_saturation = metric_saturation
        models_dir = Path(models_dir)

        print("Loading segmentation model...")
        self.seg_processor, self.seg_model = load_model_and_processor(
            seg_model_id, Mask2FormerForUniversalSegmentation, models_dir, offline)
        self.seg_model.to(self.device).eval()

        self.relative = self.metric = None
        if depth_model_id:
            print("Loading relative depth model...")
            self.relative = DepthEstimator(depth_model_id, models_dir, self.device, offline)
            if self.relative.kind != "relative":
                raise ValueError(f"'{depth_model_id}' is a {self.relative.kind} checkpoint. "
                                 f"Pass metric checkpoints with --metric-depth-model.")
        if metric_model_id:
            print("Loading metric depth model...")
            self.metric = DepthEstimator(metric_model_id, models_dir, self.device, offline)
            if self.metric.kind != "metric":
                raise ValueError(f"'{metric_model_id}' is not a metric checkpoint "
                                 f"(depth_estimation_type={self.metric.kind}).")
            print(f"  Metric model max depth: {self.metric.max_depth:.0f} m")

        self.id2label = {int(k): v for k, v in self.seg_model.config.id2label.items()}
        self.label_ids_to_fuse = (
            {i for i, n in self.id2label.items() if n.lower() in CITYSCAPES_STUFF}
            if fuse_stuff else set()
        )

    @torch.inference_mode()
    def segment(self, image):
        inputs = self.seg_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.seg_model(**inputs)
        result = self.seg_processor.post_process_panoptic_segmentation(
            outputs,
            threshold=self.threshold,
            label_ids_to_fuse=self.label_ids_to_fuse,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]
        seg_map = result["segmentation"].cpu().numpy().astype(np.int32)
        return seg_map, result["segments_info"]

    def estimate_depth(self, image):
        """Relative depth output (kept for backwards compatibility)."""
        if self.relative is None:
            raise RuntimeError("No relative depth model loaded.")
        return self.relative(image)

    def __call__(self, image):
        seg_map, segments_info = self.segment(image)
        out = {"seg_map": seg_map, "segments_info": segments_info,
               "raw_depth": None, "closeness": None, "metric_depth": None,
               "metric_valid": None, "metric_closeness": None, "metric_affine": None}

        if self.relative is not None:
            out["raw_depth"] = self.relative(image)
            out["closeness"] = normalise(out["raw_depth"])

        if self.metric is not None:
            z = self.metric(image)
            zmax = self.metric.max_depth
            valid = np.isfinite(z) & (z > 0) & (z < self.metric_saturation * zmax)
            c, affine = inverse_depth_closeness(z, valid, zmax)
            out.update(metric_depth=z, metric_valid=valid,
                       metric_closeness=c, metric_affine=affine)
            if out["closeness"] is None:
                out["closeness"] = c

        return out
