from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(env_key: str, default: Optional[Path] = None) -> Optional[Path]:
    value = os.getenv(env_key)
    if value:
        candidate = Path(value)
        if candidate.exists():
            return candidate
        # If env var is set but file doesn't exist, fall back to default if it exists
        # This handles cases where env var points to wrong/old path
        if default and default.exists():
            return default
        # If neither exists, return the env var path (for training scenarios)
        return candidate
    return default


def _default_run_path(run_name: str) -> Path:
    return PROJECT_ROOT / "runs" / "train" / run_name / "weights" / "best.pt"


def _default_reference_path(filename: str) -> Path:
    return PROJECT_ROOT / "artifacts" / filename


def _find_reference_features(primary: Path, fallback: Path) -> Optional[Path]:
    """Try primary path first, then fallback if primary doesn't exist."""
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary  # Return primary even if it doesn't exist (for error messages)


@dataclass(slots=True)
class ActivitySpec:
    key: str
    label: str
    model_path: Path
    reference_features_path: Optional[Path] = None
    reference_video_path: Optional[Path] = None
    key_node_angles: tuple[str, ...] = ()
    focus_angle_keys: tuple[str, ...] = ()
    enable_focus_metrics: bool = True
    recommendation_thresholds: dict[str, float] = field(default_factory=dict)
    recommendation_messages: dict[str, str] = field(default_factory=dict)
    default_recommendation: str = "动作整体表现良好，请保持稳定节奏。"


def get_activity_specs() -> dict[str, ActivitySpec]:
    """Return the default activity specifications for shooting and running."""
    shooting_model = _path_from_env("MODEL_PATH", _default_run_path("kobe_pose2"))
    # Try reference_features_shooting.json first, fallback to reference_features.json
    shooting_ref_primary = _default_reference_path("reference_features_shooting.json")
    shooting_ref_fallback = _default_reference_path("reference_features.json")
    shooting_ref_default = _find_reference_features(shooting_ref_primary, shooting_ref_fallback)
    shooting_reference_features = _path_from_env("REFERENCE_FEATURES_PATH", shooting_ref_default)
    shooting_reference_video = _path_from_env("REFERENCE_VIDEO_PATH")

    running_model = _path_from_env("RUN_MODEL_PATH", _default_run_path("running_pose"))
    running_reference_features = _path_from_env(
        "RUN_REFERENCE_FEATURES_PATH", _default_reference_path("reference_features_running.json")
    )
    running_reference_video = _path_from_env("RUN_REFERENCE_VIDEO_PATH")

    shooting_spec = ActivitySpec(
        key="shooting",
        label="投篮动作",
        model_path=shooting_model,
        reference_features_path=shooting_reference_features,
        reference_video_path=shooting_reference_video,
        key_node_angles=(
            "elbow_left",
            "elbow_right",
            "shoulder_left",
            "shoulder_right",
            "knee_left",
            "knee_right",
        ),
        focus_angle_keys=(
            "elbow_left",
            "elbow_right",
            "wrist_left_head",
            "wrist_right_head",
            "knee_left",
            "knee_right",
        ),
        enable_focus_metrics=True,
        recommendation_thresholds={
            "elbow_left": 8.0,
            "elbow_right": 8.0,
            "shoulder_left": 10.0,
            "shoulder_right": 10.0,
            "knee_left": 12.0,
            "knee_right": 12.0,
            "__default__": 10.0,
        },
        recommendation_messages={
            "elbow_left": "左肘角度偏大，注意收肘保持垂直。",
            "elbow_right": "右肘角度偏大，注意收肘保持垂直。",
            "shoulder_left": "左肩发力不稳定，尝试保持肩部稳定。",
            "shoulder_right": "右肩发力不稳定，尝试保持肩部稳定。",
            "knee_left": "左膝弯曲不足，起跳时需要更多下蹲。",
            "knee_right": "右膝弯曲不足，起跳时需要更多下蹲。",
        },
        default_recommendation="投篮动作整体表现良好，保持稳定节奏。",
    )

    running_spec = ActivitySpec(
        key="running",
        label="跑步动作",
        model_path=running_model,
        reference_features_path=running_reference_features,
        reference_video_path=running_reference_video,
        key_node_angles=(
            "hip_left",
            "hip_right",
            "knee_left",
            "knee_right",
            "ankle_left",
            "ankle_right",
        ),
        focus_angle_keys=(),  # 自定义指标由推理服务计算
        enable_focus_metrics=True,
        recommendation_thresholds={
            "hip_left": 8.0,
            "hip_right": 8.0,
            "knee_left": 6.0,
            "knee_right": 6.0,
            "ankle_left": 6.0,
            "ankle_right": 6.0,
            "__default__": 8.0,
        },
        recommendation_messages={
            "hip_left": "左髋摆动幅度偏差较大，请收紧核心保持髋部稳定。",
            "hip_right": "右髋摆动幅度偏差较大，请收紧核心保持髋部稳定。",
            "knee_left": "左膝摆动高度不一致，注意保持一致的步幅。",
            "knee_right": "右膝摆动高度不一致，注意保持一致的步幅。",
            "ankle_left": "左脚踝离地角度波动较大，放松脚踝保持顺畅摆动。",
            "ankle_right": "右脚踝离地角度波动较大，放松脚踝保持顺畅摆动。",
        },
        default_recommendation="跑步姿势整体表现良好，保持步频和髋部稳定。",
    )

    return {
        shooting_spec.key: shooting_spec,
        running_spec.key: running_spec,
    }


__all__ = ["ActivitySpec", "get_activity_specs", "PROJECT_ROOT"]

