"""
pipeline.py
-----------
PanopticDepthPipeline: loads Mask2Former (panoptic segmentation) and
Depth Anything V2 (relative depth) once, then runs both on any number of images.

Methods:
  * segment(image)         -> (seg_map, segments_info)
                              seg_map is an HxW int array of segment IDs.
  * estimate_depth(image)  -> HxW raw depth output, resized to the image.
                              Depth Anything V2 (relative) predicts affine-
                              invariant inverse depth: higher = closer.
  * __call__(image)        -> dict with seg_map, segments_info, raw_depth and
                              closeness (raw depth normalised to [0, 1]).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation, Mask2FormerForUniversalSegmentation

from .config import CITYSCAPES_STUFF, DEFAULT_MODELS_DIR, DEPTH_MODEL_ID, SEG_MODEL_ID
from .models import load_model_and_processor
from .utils import get_device, normalise


class PanopticDepthPipeline:
    def __init__(self, device=None, models_dir=DEFAULT_MODELS_DIR,
                 seg_model_id=SEG_MODEL_ID, depth_model_id=DEPTH_MODEL_ID,
                 offline=False, threshold=0.5, fuse_stuff=True):
        self.device = device or get_device()
        self.threshold = threshold
        models_dir = Path(models_dir)

        print("Loading segmentation model...")
        self.seg_processor, self.seg_model = load_model_and_processor(
            seg_model_id, Mask2FormerForUniversalSegmentation, models_dir, offline)
        self.seg_model.to(self.device).eval()

        print("Loading depth model...")
        self.depth_processor, self.depth_model = load_model_and_processor(
            depth_model_id, AutoModelForDepthEstimation, models_dir, offline)
        self.depth_model.to(self.device).eval()

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

    @torch.inference_mode()
    def estimate_depth(self, image):
        inputs = self.depth_processor(images=image, return_tensors="pt").to(self.device)
        predicted = self.depth_model(**inputs).predicted_depth  # (1, h, w)
        return F.interpolate(
            predicted.unsqueeze(1), size=image.size[::-1],
            mode="bilinear", align_corners=False,
        ).squeeze().float().cpu().numpy()

    def __call__(self, image):
        seg_map, segments_info = self.segment(image)
        raw_depth = self.estimate_depth(image)
        return {
            "seg_map": seg_map,
            "segments_info": segments_info,
            "raw_depth": raw_depth,
            "closeness": normalise(raw_depth),
        }
