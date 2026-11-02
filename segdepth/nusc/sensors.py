"""
sensors.py
----------
Sensor geometry and radar loading for nuScenes.

Contents:
  * RADARS_FOR_CAMERA  Radars whose field of view overlaps each camera.
  * pose_matrix        4x4 transform from a calibrated_sensor or ego_pose record.
  * radar_filters      Context manager that sets the devkit's radar quality
                       filters and always restores the defaults afterwards.
  * CameraModel        Intrinsics, image size and camera-to-ego extrinsic.
  * RadarPoints        Radar detections from one or more radars and sweeps,
                       kept in each radar's own frame together with the
                       transforms needed to reach the camera and ego frames.
                       Retains velocity, RCS and timing information.
  * load_radar_points  Build RadarPoints for a keyframe and camera.

Transforms account for the physical radar-camera offset and for the ego
vehicle's motion between the radar and camera timestamps.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

RADARS_FOR_CAMERA = {
    "CAM_FRONT": ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"],
    "CAM_FRONT_LEFT": ["RADAR_FRONT_LEFT", "RADAR_FRONT"],
    "CAM_FRONT_RIGHT": ["RADAR_FRONT_RIGHT", "RADAR_FRONT"],
    "CAM_BACK": ["RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"],
    "CAM_BACK_LEFT": ["RADAR_BACK_LEFT", "RADAR_FRONT_LEFT"],
    "CAM_BACK_RIGHT": ["RADAR_BACK_RIGHT", "RADAR_FRONT_RIGHT"],
}

# RadarPointCloud row indices (18 fields)
R_X, R_Y, R_DYN_PROP, R_RCS, R_VX_COMP, R_VY_COMP = 0, 1, 3, 5, 8, 9


def pose_matrix(rec, inverse=False):
    return transform_matrix(rec["translation"], Quaternion(rec["rotation"]), inverse=inverse)


@contextmanager
def radar_filters(use_default=True):
    if use_default:
        RadarPointCloud.default_filters()
    else:
        RadarPointCloud.disable_filters()
    try:
        yield
    finally:
        RadarPointCloud.default_filters()


@dataclass
class CameraModel:
    K: np.ndarray
    width: int
    height: int
    T_ego_from_cam: np.ndarray

    @property
    def fx(self):
        return float(self.K[0, 0])

    @property
    def fy(self):
        return float(self.K[1, 1])

    @classmethod
    def from_nusc(cls, nusc, cam_token):
        rec = nusc.get("sample_data", cam_token)
        cal = nusc.get("calibrated_sensor", rec["calibrated_sensor_token"])
        return cls(np.array(cal["camera_intrinsic"]), rec["width"], rec["height"],
                   pose_matrix(cal))


@dataclass
class RadarPoints:
    xy: np.ndarray        # (N, 2) position in its radar frame (m)
    vel: np.ndarray       # (N, 2) ego-motion-compensated velocity, radar frame (m/s)
    v_r: np.ndarray       # compensated radial velocity (m/s)
    rcs: np.ndarray       # dBsm
    dyn_prop: np.ndarray  # dynamic property code
    dt: np.ndarray        # camera timestamp - radar timestamp (s)
    channel: np.ndarray   # radar channel name
    sweep: np.ndarray     # index into the transform lists
    T_cam: list           # camera <- radar
    T_ego: list           # ego (radar time) <- radar, for height above ground
    T_egocam: list        # ego (camera time) <- radar, common frame for all sweeps

    def __len__(self):
        return len(self.v_r)


def load_radar_points(nusc, sample_token, cam_token, channels=None, nsweeps=3,
                      use_default_filters=True, min_range=1.0) -> RadarPoints:
    cam_rec = nusc.get("sample_data", cam_token)
    c_cal = nusc.get("calibrated_sensor", cam_rec["calibrated_sensor_token"])
    c_ego = nusc.get("ego_pose", cam_rec["ego_pose_token"])
    cam_from_ego = pose_matrix(c_cal, inverse=True)
    egocam_from_global = pose_matrix(c_ego, inverse=True)
    channels = channels or RADARS_FOR_CAMERA[cam_rec["channel"]]
    sample = nusc.get("sample", sample_token)

    cols = {k: [] for k in ("xy", "vel", "v_r", "rcs", "dyn_prop", "dt", "channel", "sweep")}
    T_cam, T_ego, T_egocam = [], [], []

    with radar_filters(use_default_filters):
        for chan in channels:
            sd_token = sample["data"][chan]
            for _ in range(nsweeps):
                if not sd_token:
                    break
                sd = nusc.get("sample_data", sd_token)
                s_cal = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                s_ego = nusc.get("ego_pose", sd["ego_pose_token"])
                ego_from_radar = pose_matrix(s_cal)
                egocam_from_radar = egocam_from_global @ pose_matrix(s_ego) @ ego_from_radar

                k = len(T_cam)
                T_ego.append(ego_from_radar)
                T_egocam.append(egocam_from_radar)
                T_cam.append(cam_from_ego @ egocam_from_radar)

                pc = RadarPointCloud.from_file(nusc.get_sample_data_path(sd_token)).points
                xy = pc[[R_X, R_Y]].T
                r = np.linalg.norm(xy, axis=1)
                keep = r > min_range
                xy, r = xy[keep], r[keep]
                vel = pc[[R_VX_COMP, R_VY_COMP]].T[keep]

                cols["xy"].append(xy)
                cols["vel"].append(vel)
                cols["v_r"].append(np.sum(xy * vel, axis=1) / r)
                cols["rcs"].append(pc[R_RCS, keep])
                cols["dyn_prop"].append(pc[R_DYN_PROP, keep].astype(np.int8))
                cols["dt"].append(np.full(len(r), (cam_rec["timestamp"] - sd["timestamp"]) * 1e-6))
                cols["channel"].append(np.full(len(r), chan))
                cols["sweep"].append(np.full(len(r), k))
                sd_token = sd["prev"]

    def cat(key, empty):
        return np.concatenate(cols[key]) if cols[key] else empty

    return RadarPoints(
        xy=cat("xy", np.zeros((0, 2))), vel=cat("vel", np.zeros((0, 2))),
        v_r=cat("v_r", np.zeros(0)), rcs=cat("rcs", np.zeros(0)),
        dyn_prop=cat("dyn_prop", np.zeros(0, dtype=np.int8)),
        dt=cat("dt", np.zeros(0)), channel=cat("channel", np.zeros(0, dtype=str)),
        sweep=cat("sweep", np.zeros(0, dtype=int)).astype(int),
        T_cam=T_cam, T_ego=T_ego, T_egocam=T_egocam,
    )
