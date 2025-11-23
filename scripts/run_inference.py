from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import ActivitySpec, get_activity_specs
from src.models.inference import PoseInferenceService, build_reference_features_from_video
from src.utils.serialization import load_features_from_file, save_features_to_file


def ensure_reference_features(
    spec: ActivitySpec,
    *,
    reference_features_path: Path | None,
    reference_video: Path | None,
    device: str | None = None,
    target_fps: float | None = None,
) -> list:
    if reference_features_path and reference_features_path.exists():
        return load_features_from_file(reference_features_path)

    if reference_video and reference_video.exists():
        features = build_reference_features_from_video(
            video_path=reference_video,
            model_path=spec.model_path,
            device=device,
            target_fps=target_fps,
        )
        if reference_features_path:
            reference_features_path.parent.mkdir(parents=True, exist_ok=True)
            save_features_to_file(features, reference_features_path)
        return features

    raise RuntimeError(
        f"活动「{spec.label}」缺少参考特征，请提供 --reference-video 或 --reference-features 路径。"
    )


def run_inference(
    *,
    activity_key: str,
    user_video: Path,
    reference_features_path: Path | None = None,
    reference_video: Path | None = None,
    device: str | None = None,
    target_fps: float | None = None,
) -> None:
    activity_specs = get_activity_specs()
    if activity_key not in activity_specs:
        raise KeyError(f"未知的动作类型: {activity_key}")

    spec = activity_specs[activity_key]
    if not spec.model_path.exists():
        raise FileNotFoundError(
            f"未找到模型文件: {spec.model_path}\n请先训练 {spec.label} 模型或更新 RUN_MODEL_PATH/MODEL_PATH 配置。"
        )

    reference_features = ensure_reference_features(
        spec,
        reference_features_path=reference_features_path or spec.reference_features_path,
        reference_video=reference_video or spec.reference_video_path,
        device=device,
        target_fps=target_fps,
    )

    service = PoseInferenceService(
        model_path=spec.model_path,
        reference_features=reference_features,
        device=device,
        activity_key=spec.key,
        key_node_angles=spec.key_node_angles,
        focus_angle_keys=spec.focus_angle_keys,
        enable_focus_metrics=spec.enable_focus_metrics,
        recommendation_thresholds=spec.recommendation_thresholds,
        recommendation_messages=spec.recommendation_messages,
        default_recommendation=spec.default_recommendation,
    )

    result = service.compare(user_video, target_fps=target_fps)
    print(f"[{spec.label}] Similarity scores:")
    for joint, score in result.similarity_scores.items():
        print(f"  {joint}: {score:.3f}")

    print("\nRecommendations:")
    for advice in result.recommendation:
        print(f" - {advice}")

    print(f"\nAngle plot saved to: {result.angle_plot_path}")
    if result.key_node_similarity_plot_path:
        print(f"Key node similarity plot saved to: {result.key_node_similarity_plot_path}")
    if result.focus_similarity_plot_path:
        print(f"Focus similarity plot saved to: {result.focus_similarity_plot_path}")
    if result.annotated_video_path:
        print(f"Annotated video saved to: {result.annotated_video_path}")

    if result.focus_metrics:
        print("\nFocus metrics:")
        for name, value in result.focus_metrics.items():
            print(f"  {name}: {value:.2f}" if value == value else f"  {name}: N/A")
    if result.focus_similarity:
        print("\nFocus metric similarity:")
        for name, value in result.focus_similarity.items():
            print(f"  {name}: {value:.2f}")


def parse_args() -> argparse.Namespace:
    specs = get_activity_specs()
    parser = argparse.ArgumentParser(description="运行单次离线推理")
    parser.add_argument(
        "--activity",
        choices=specs.keys(),
        default=os.getenv("ACTIVITY", "shooting"),
        help="选择动作类型 (shooting/running)",
    )
    parser.add_argument("--user-video", type=Path, required=True, help="待分析的视频路径")
    parser.add_argument(
        "--reference-video",
        type=Path,
        help="可选：覆盖默认的标准动作视频路径",
    )
    parser.add_argument(
        "--reference-features",
        type=Path,
        help="可选：覆盖默认的参考特征文件路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DEVICE"),
        help="推理设备 (如 '0' 或 'cpu')",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="抽帧 FPS，默认使用配置文件中的设置",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        activity_key=args.activity,
        user_video=args.user_video,
        reference_features_path=args.reference_features,
        reference_video=args.reference_video,
        device=args.device,
        target_fps=args.target_fps,
    )
