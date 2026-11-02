"""
segdepth.nusc
=============
nuScenes integration. Requires nuscenes-devkit and pyquaternion.

Modules:
    scenes             Scene/keyframe selection and camera frame loading
    sensors            Pose transforms, CameraModel, radar point loading
    lidar              LiDAR projection and per-segment validation
    radar_heuristic    Simple radar association (vertical window or
                       depth-consistent search)
    radar_association  Depth-only radar association by energy minimisation
    evaluation         Scoring radar association against 3D box annotations
    bev                Top-down plot of segments, radar points and associations
"""
