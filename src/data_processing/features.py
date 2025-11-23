from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


JOINT_PAIRS_FOR_ANGLES: dict[str, tuple[int, int, int]] = {
    "elbow_left": (5, 7, 9),  # shoulder, elbow, wrist
    "elbow_right": (6, 8, 10),
    "knee_left": (11, 13, 15),  # hip, knee, ankle
    "knee_right": (12, 14, 16),
    "shoulder_left": (11, 5, 7),  # hip, shoulder, elbow
    "shoulder_right": (12, 6, 8),
    "wrist_left_head": (7, 9, 0),  # elbow, wrist, nose(head proxy)
    "wrist_right_head": (8, 10, 0),
}

JOINT_INDEX_TO_NAME = {
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


def _vector_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Return the angle between two vectors in degrees."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-6 or b_norm < 1e-6:
        return float("nan")
    cos_theta = np.clip(np.dot(a, b) / (a_norm * b_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_joint_angles(keypoints: np.ndarray) -> dict[str, float]:
    """
    Compute joint angles given the YOLO keypoint tensor.

    Parameters
    ----------
    keypoints:
        Array of shape (num_keypoints, 3). Only x, y are used.
    """
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        raise ValueError("Keypoints array must have shape (num_keypoints, >=2)")

    angles: dict[str, float] = {}
    coordinates = keypoints[:, :2]

    for name, (a_idx, b_idx, c_idx) in JOINT_PAIRS_FOR_ANGLES.items():
        vec1 = coordinates[a_idx] - coordinates[b_idx]
        vec2 = coordinates[c_idx] - coordinates[b_idx]
        angles[name] = _vector_angle(vec1, vec2)
    return angles


def compute_joint_velocities(
    keypoint_sequence: Iterable[np.ndarray],
    *,
    fps: float,
) -> list[dict[str, float]]:
    """
    Compute per-frame joint velocities given a sequence of keypoints.
    """
    velocities: list[dict[str, float]] = []
    keypoint_sequence = list(keypoint_sequence)
    if len(keypoint_sequence) < 2:
        return velocities

    for prev, current in zip(keypoint_sequence[:-1], keypoint_sequence[1:]):
        dt = 1.0 / fps
        diff = (current[:, :2] - prev[:, :2]) / dt
        joint_velocities = {
            JOINT_INDEX_TO_NAME[idx]: float(np.linalg.norm(diff[idx]))
            for idx in JOINT_INDEX_TO_NAME
        }
        velocities.append(joint_velocities)
    return velocities


def compute_force_metrics(
    joint_velocities: list[dict[str, float]],
    *,
    mass: float = 80.0,
    limb_weight_ratio: Optional[dict[str, float]] = None,
) -> list[dict[str, float]]:
    """
    Estimate force-related metrics heuristically from joint velocities.

    Parameters
    ----------
    joint_velocities:
        Output of `compute_joint_velocities`.
    mass:
        Athlete mass in kilograms. Used for approximate force estimation.
    limb_weight_ratio:
        Proportion of body mass associated with each joint segment.
    """
    limb_weight_ratio = limb_weight_ratio or {
        "left_shoulder": 0.08,
        "right_shoulder": 0.08,
        "left_elbow": 0.03,
        "right_elbow": 0.03,
        "left_knee": 0.07,
        "right_knee": 0.07,
        "left_ankle": 0.04,
        "right_ankle": 0.04,
    }

    force_metrics: list[dict[str, float]] = []
    for frame_velocities in joint_velocities:
        forces = {
            joint: frame_velocities.get(joint, 0.0) * mass * limb_weight_ratio.get(joint, 0.05)
            for joint in frame_velocities
        }
        force_metrics.append(forces)
    return force_metrics


@dataclass(slots=True)
class FeatureSet:
    frame_index: int
    angles: dict[str, float]
    velocities: dict[str, float]
    forces: dict[str, float]
    keypoints: np.ndarray


def compute_feature_vector(
    keypoint_sequence: Iterable[np.ndarray],
    *,
    fps: float,
) -> list[FeatureSet]:
    """
    Aggregate angle, velocity and force metrics for a keypoint sequence.
    """
    keypoint_sequence = list(keypoint_sequence)
    if not keypoint_sequence:
        return []

    angles = [compute_joint_angles(kps) for kps in keypoint_sequence]
    velocities = compute_joint_velocities(keypoint_sequence, fps=fps)
    if velocities:
        velocities = [velocities[0]] + velocities  # repeat first to align length
    else:
        velocities = [{} for _ in keypoint_sequence]

    forces = compute_force_metrics([v for v in velocities if v], mass=80.0)
    if forces:
        forces = [forces[0]] + forces
    else:
        forces = [{} for _ in keypoint_sequence]

    feature_sets: list[FeatureSet] = []
    for idx, (kps, angle_dict, velocity_dict, force_dict) in enumerate(
        zip(keypoint_sequence, angles, velocities, forces)
    ):
        feature_sets.append(
            FeatureSet(
                frame_index=idx,
                angles=angle_dict,
                velocities=velocity_dict,
                forces=force_dict,
                keypoints=kps,
            )
        )
    return feature_sets

