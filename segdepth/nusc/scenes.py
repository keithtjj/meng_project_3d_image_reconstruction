"""
scenes.py
---------
Navigating the nuScenes dataset and loading camera frames.

Functions:
  * list_scenes          Print all scenes with sample count, location and description.
  * resolve_scene        Find a scene by name (e.g. scene-0061) or index.
  * scene_sample_tokens  Keyframe sample tokens of a scene in temporal order.
  * load_camera_frame    Load a camera image for a keyframe plus metadata
                         (tokens, timestamp, intrinsics, ego pose).
"""

from PIL import Image

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def list_scenes(nusc):
    print(f"{'Idx':>3}  {'Name':<11} {'Samples':>7}  {'Location':<22} Description")
    print("-" * 100)
    for i, s in enumerate(nusc.scene):
        location = nusc.get("log", s["log_token"])["location"]
        print(f"{i:>3}  {s['name']:<11} {s['nbr_samples']:>7}  {location:<22} {s['description']}")


def resolve_scene(nusc, scene_arg):
    if scene_arg is None:
        return nusc.scene[0]
    if scene_arg.isdigit():
        idx = int(scene_arg)
        if not 0 <= idx < len(nusc.scene):
            raise ValueError(f"Scene index {idx} out of range (0-{len(nusc.scene) - 1})")
        return nusc.scene[idx]
    for s in nusc.scene:
        if s["name"] == scene_arg:
            return s
    raise ValueError(f"Scene '{scene_arg}' not found. Use --list-scenes.")


def scene_sample_tokens(nusc, scene):
    tokens, token = [], scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = nusc.get("sample", token)["next"]
    return tokens


def load_camera_frame(nusc, scene, sample_token, camera, sample_index):
    """Returns (PIL image, camera sample_data token, metadata dict)."""
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"][camera]
    cam_data = nusc.get("sample_data", cam_token)
    calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", cam_data["ego_pose_token"])

    image = Image.open(nusc.get_sample_data_path(cam_token)).convert("RGB")
    meta = {
        "scene_name": scene["name"],
        "scene_description": scene["description"],
        "location": nusc.get("log", scene["log_token"])["location"],
        "sample_index": sample_index,
        "sample_token": sample_token,
        "sample_data_token": cam_token,
        "camera": camera,
        "timestamp_us": cam_data["timestamp"],
        "image_file": cam_data["filename"],
        "image_size_wh": list(image.size),
        "camera_intrinsic": calib["camera_intrinsic"],
        "ego_translation": ego["translation"],
        "ego_rotation_wxyz": ego["rotation"],
    }
    return image, cam_token, meta
