from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

from src.data_processing.features import FeatureSet


def render_angle_curves(
    user_features: Iterable[FeatureSet],
    reference_features: Iterable[FeatureSet],
    *,
    output_path: str | Path,
    joints: Optional[list[str]] = None,
) -> Path:
    user_features = list(user_features)
    reference_features = list(reference_features)
    if not user_features:
        raise ValueError("User features cannot be empty.")

    joints = joints or list(user_features[0].angles.keys())
    output_path = Path(output_path)

    plt.figure(figsize=(12, 6))
    for joint in joints:
        user_series = np.array([f.angles.get(joint, np.nan) for f in user_features])
        plt.plot(user_series, label=f"用户 {joint}")
        if reference_features:
            reference_series = np.array([f.angles.get(joint, np.nan) for f in reference_features])
            plt.plot(reference_series, linestyle="--", label=f"参考 {joint}")

    plt.title("关节角度变化曲线")
    plt.xlabel("帧序号")
    plt.ylabel("角度 (°)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def render_similarity_bars(
    similarity_scores: dict[str, float],
    *,
    output_path: str | Path,
    title: str = "各关键关节与标准动作的相似度",
) -> Optional[Path]:
    items = [(joint, score) for joint, score in similarity_scores.items() if score == score]
    if not items:
        return None

    joints, values = zip(*items)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(joints, values, color="#1d4ed8")
    plt.ylim(0, 1.05)
    plt.ylabel("相似度")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path

