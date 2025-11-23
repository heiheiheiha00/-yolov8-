from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from src.data_processing.features import FeatureSet, compute_feature_vector
from src.data_processing.keypoints import PoseEstimator
from src.data_processing.pipeline import DataProcessingPipeline, ProcessedSample
from src.visualization.plots import render_angle_curves, render_similarity_bars
from src.visualization.video import annotate_video_with_pose


@dataclass(slots=True)
class PoseComparisonResult:
    processed_sample: ProcessedSample
    reference_features: list[FeatureSet]
    similarity_scores: dict[str, float]
    deviation_angles: dict[str, float]
    recommendation: list[str]
    angle_plot_path: Path
    key_node_similarity_plot_path: Optional[Path] = None
    focus_similarity_plot_path: Optional[Path] = None
    annotated_video_path: Optional[Path] = None
    focus_metrics: Optional[dict[str, float]] = None
    focus_similarity: Optional[dict[str, float]] = None


class PoseInferenceService:
    """
    End-to-end service that extracts features from a user video and compares them
    with a reference template.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        reference_features: list[FeatureSet],
        device: Optional[str] = None,
        temp_dir: str | Path = "artifacts/inference_frames",
        fps: float = 30.0,
        activity_key: str = "shooting",
        key_node_angles: Optional[Iterable[str]] = None,
        focus_angle_keys: Optional[Iterable[str]] = None,
        enable_focus_metrics: bool = True,
        recommendation_thresholds: Optional[dict[str, float]] = None,
        recommendation_messages: Optional[dict[str, str]] = None,
        default_recommendation: Optional[str] = None,
        pipeline_kwargs: Optional[dict[str, object]] = None,
    ) -> None:
        pipeline_args: dict[str, object] = {
            "model_path": model_path,
            "device": device,
            "temp_dir": temp_dir,
            "fps": fps,
            "enable_tracking": True,
            "enable_smoothing": True,
            "smoothing_window": 5,
            "tracking_max_distance": 0.5,
            "tracking_mode": "simple",
        }
        if activity_key == "running":
            pipeline_args.setdefault("smoothing_window", 3)
            pipeline_args.setdefault("smoothing_mode", "ema")
            pipeline_args.setdefault("tracking_max_distance", 0.35)
            pipeline_args["tracking_mode"] = "enhanced"
        if pipeline_kwargs:
            pipeline_args.update(pipeline_kwargs)
        self.pipeline = DataProcessingPipeline(**pipeline_args)
        if not reference_features:
            raise ValueError("Reference features must not be empty.")
        self.reference_features = reference_features
        self.activity_key = activity_key
        self.enable_key_node_plot = activity_key != "running"
        default_focus = {
            "elbow_left",
            "elbow_right",
            "wrist_left_head",
            "wrist_right_head",
            "knee_left",
            "knee_right",
        }
        default_key_nodes = {
            "elbow_left",
            "elbow_right",
            "shoulder_left",
            "shoulder_right",
            "knee_left",
            "knee_right",
        }
        self.key_node_angles = set(key_node_angles) if key_node_angles else default_key_nodes
        self.focus_similarity_title = (
            "跑步关键指标相似度" if activity_key == "running" else "重点关节相似度"
        )
        if focus_angle_keys is None:
            self.focus_angle_keys = default_focus
        else:
            self.focus_angle_keys = set(focus_angle_keys)
        self.enable_focus_metrics = enable_focus_metrics
        self.recommendation_thresholds = recommendation_thresholds or {
            "elbow_left": 8.0,
            "elbow_right": 8.0,
            "shoulder_left": 10.0,
            "shoulder_right": 10.0,
            "knee_left": 12.0,
            "knee_right": 12.0,
            "__default__": 10.0,
        }
        self.recommendation_messages = recommendation_messages or {
            "elbow_left": "左肘角度偏大，注意收肘保持垂直。",
            "elbow_right": "右肘角度偏大，注意收肘保持垂直。",
            "shoulder_left": "左肩发力不稳定，尝试保持肩部稳定。",
            "shoulder_right": "右肩发力不稳定，尝试保持肩部稳定。",
            "knee_left": "左膝弯曲不足，起跳时需要更多下蹲。",
            "knee_right": "右膝弯曲不足，起跳时需要更多下蹲。",
        }
        self.default_recommendation = (
            default_recommendation or "动作整体表现良好，保持稳定的投篮节奏。"
        )

    def compare(
        self,
        user_video: str | Path,
        *,
        output_dir: str | Path = "artifacts/results",
        target_fps: Optional[float] = None,
    ) -> PoseComparisonResult:
        processed = self.pipeline.process_video(user_video, target_fps=target_fps)
        user_features = processed.features
        if not user_features:
            raise RuntimeError("未检测到人体关键点，请提供更清晰的投篮视频。")

        similarity_scores_raw = self._compute_similarity(user_features, self.reference_features)
        similarity_scores = {k: v for k, v in similarity_scores_raw.items() if v == v}
        deviations = self._compute_angle_deviations(user_features, self.reference_features)
        recommendation = self._generate_recommendations(deviations)
        focus_metrics: dict[str, float] = {}
        reference_focus: dict[str, float] = {}
        focus_similarities: dict[str, float] = {}
        if self.enable_focus_metrics:
            focus_metrics = self._compute_focus_metrics(user_features)
            reference_focus = self._compute_focus_metrics(self.reference_features)
            focus_similarities_raw = self._compute_focus_similarity(focus_metrics, reference_focus)
            focus_similarities = {k: v for k, v in focus_similarities_raw.items() if v == v}

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        angle_plot_path = output_dir / f"{Path(user_video).stem}_angles.png"
        render_angle_curves(user_features, self.reference_features, output_path=angle_plot_path)
        key_node_plot_path: Optional[Path] = None
        if self.enable_key_node_plot:
            key_node_similarities = {
                k: v for k, v in similarity_scores.items() if k in self.key_node_angles
            }
            if key_node_similarities:
                key_node_plot_path = output_dir / f"{Path(user_video).stem}_keynode_similarity.png"
                render_similarity_bars(
                    key_node_similarities,
                    output_path=key_node_plot_path,
                    title="关键节点相似度",
                )

        advice_path = output_dir / "advice.txt"
        advice_path.write_text("\n".join(recommendation), encoding="utf-8")

        annotated_video_path: Optional[Path] = None
        try:
            annotated_video_path = output_dir / f"{Path(user_video).stem}_annotated.mp4"
            annotate_video_with_pose(
                sample=processed.features,
                frame_paths=processed.frame_paths,
                output_path=annotated_video_path,
            )
        except Exception:
            annotated_video_path = None

        focus_similarity_plot_path: Optional[Path] = None
        if focus_similarities:
            focus_similarity_plot_path = output_dir / f"{Path(user_video).stem}_focus_similarity.png"
            render_similarity_bars(
                focus_similarities,
                output_path=focus_similarity_plot_path,
                title=self.focus_similarity_title,
            )

        return PoseComparisonResult(
            processed_sample=processed,
            reference_features=self.reference_features,
            similarity_scores=similarity_scores,
            deviation_angles=deviations,
            recommendation=recommendation,
            angle_plot_path=angle_plot_path,
            key_node_similarity_plot_path=key_node_plot_path,
            focus_similarity_plot_path=focus_similarity_plot_path,
            annotated_video_path=annotated_video_path,
            focus_metrics=focus_metrics or None,
            focus_similarity=focus_similarities or None,
        )

    def _compute_similarity(
        self,
        user_features: list[FeatureSet],
        reference_features: list[FeatureSet],
    ) -> dict[str, float]:
        if not user_features or not reference_features:
            return {}

        angle_names = user_features[0].angles.keys()
        similarities: dict[str, float] = {}

        def _resample(series: np.ndarray, target_len: int) -> np.ndarray:
            if len(series) == target_len or len(series) == 0:
                return series
            x_original = np.linspace(0.0, 1.0, num=len(series))
            x_target = np.linspace(0.0, 1.0, num=target_len)
            return np.interp(x_target, x_original, series)

        target_len = min(len(user_features), len(reference_features))
        if target_len == 0:
            return {}

        for name in angle_names:
            user_series = np.array([f.angles.get(name, np.nan) for f in user_features], dtype=float)
            ref_series = np.array([f.angles.get(name, np.nan) for f in reference_features], dtype=float)

            if len(user_series) != target_len:
                user_series = _resample(user_series, target_len)
            if len(ref_series) != target_len:
                ref_series = _resample(ref_series, target_len)

            mask = ~np.isnan(user_series) & ~np.isnan(ref_series)
            if mask.sum() == 0:
                similarities[name] = float("nan")
                continue
            masked_user = user_series[mask]
            masked_ref = ref_series[mask]
            numerator = float(np.dot(masked_user, masked_ref))
            user_norm = float(np.linalg.norm(masked_user))
            ref_norm = float(np.linalg.norm(masked_ref))
            denom = user_norm * ref_norm
            if denom == 0.0:
                similarities[name] = float("nan")
            else:
                similarities[name] = float(numerator / denom)
        return similarities

    def _compute_angle_deviations(
        self,
        user_features: list[FeatureSet],
        reference_features: list[FeatureSet],
    ) -> dict[str, float]:
        if not user_features or not reference_features:
            return {}

        angle_names = user_features[0].angles.keys()
        deviations: dict[str, float] = {}

        target_len = min(len(user_features), len(reference_features))
        if target_len == 0:
            return {}

        def _resample(series: np.ndarray, target_len: int) -> np.ndarray:
            if len(series) == target_len or len(series) == 0:
                return series
            x_original = np.linspace(0.0, 1.0, num=len(series))
            x_target = np.linspace(0.0, 1.0, num=target_len)
            return np.interp(x_target, x_original, series)

        for name in angle_names:
            user_series = np.array([f.angles.get(name, np.nan) for f in user_features], dtype=float)
            ref_series = np.array([f.angles.get(name, np.nan) for f in reference_features], dtype=float)

            if len(user_series) != target_len:
                user_series = _resample(user_series, target_len)
            if len(ref_series) != target_len:
                ref_series = _resample(ref_series, target_len)

            mask = ~np.isnan(user_series) & ~np.isnan(ref_series)
            if mask.sum() == 0:
                deviations[name] = float("nan")
                continue
            deviations[name] = float(np.nanmean(np.abs(user_series[mask] - ref_series[mask])))
        return deviations

    def _generate_recommendations(self, deviations: dict[str, float]) -> list[str]:
        recs: list[str] = []
        thresholds = self.recommendation_thresholds
        default_threshold = thresholds.get("__default__", 10.0)
        messages = self.recommendation_messages

        def _joint_label_cn(joint_name: str) -> str:
            mapping = {
                "hip_left": "左髋",
                "hip_right": "右髋",
                "knee_left": "左膝",
                "knee_right": "右膝",
                "ankle_left": "左脚踝",
                "ankle_right": "右脚踝",
                "elbow_left": "左肘",
                "elbow_right": "右肘",
                "shoulder_left": "左肩",
                "shoulder_right": "右肩",
            }
            return mapping.get(joint_name, joint_name)

        for joint, deviation in deviations.items():
            threshold = thresholds.get(joint, default_threshold)
            if np.isnan(deviation):
                continue
            if deviation > threshold:
                base_message = messages.get(joint, f"{_joint_label_cn(joint)} 动作偏差较大，请注意调整。")
                if self.activity_key == "running":
                    if joint.startswith("hip"):
                        base_message = (
                            f"{_joint_label_cn(joint)}摆动幅度偏差较大，建议收紧核心稳定骨盆。"
                        )
                    elif joint.startswith("knee"):
                        base_message = (
                            f"{_joint_label_cn(joint)}抬举高度不一致，可适当增大抬膝幅度以保持步幅。"
                        )
                    elif joint.startswith("ankle"):
                        base_message = (
                            f"{_joint_label_cn(joint)}离地角度波动偏大，建议脚踝放松并保持抬脚节奏。"
                        )
                    else:
                        base_message = (
                            f"{_joint_label_cn(joint)}动作仍需更稳定，注意步频与摆幅协调。"
                        )
                else:
                    if joint.startswith("elbow"):
                        base_message = f"{_joint_label_cn(joint)}角度偏大，收紧肘部减少左右摆动。"
                    elif joint.startswith("shoulder"):
                        base_message = f"{_joint_label_cn(joint)}发力不稳定，肩部保持放松稳定。"
                    elif joint.startswith("knee"):
                        base_message = f"{_joint_label_cn(joint)}弯曲不足，起跳前可加大下蹲幅度。"
                    else:
                        base_message = f"{_joint_label_cn(joint)}动作偏差较大，请保持稳定节奏。"
                recs.append(base_message)
        if not recs:
            recs.append(self.default_recommendation)
        return recs

    def _compute_focus_metrics(self, user_features: list[FeatureSet]) -> dict[str, float]:
        if not self.enable_focus_metrics or not user_features:
            return {}
        if self.activity_key == "running":
            return self._compute_running_focus_metrics(user_features)
        return self._compute_shooting_focus_metrics(user_features)

    def _compute_shooting_focus_metrics(self, user_features: list[FeatureSet]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if not user_features:
            return metrics

        def _avg_peak_bend(angle_name: str, min_cycle_frames: int = 3) -> float:
            series = np.array([f.angles.get(angle_name, np.nan) for f in user_features], dtype=float)
            if series.size == 0:
                return float("nan")
            bend = 180.0 - series
            bend[(series != series) | (bend > 160.0) | (bend < 0.0)] = np.nan
            if np.all(np.isnan(bend)):
                return float("nan")

            peaks: list[float] = []
            last_peak_idx = -min_cycle_frames
            for idx in range(len(bend)):
                value = bend[idx]
                if np.isnan(value):
                    continue
                prev = bend[idx - 1] if idx > 0 else np.nan
                nxt = bend[idx + 1] if idx < len(bend) - 1 else np.nan
                cond_left = np.isnan(prev) or value >= prev
                cond_right = np.isnan(nxt) or value >= nxt
                if cond_left and cond_right and (idx - last_peak_idx) >= min_cycle_frames:
                    peaks.append(value)
                    last_peak_idx = idx

            if not peaks:
                valid = bend[~np.isnan(bend)]
                if valid.size == 0:
                    return float("nan")
                return float(np.nanmean(valid))

            return float(np.nanmean(peaks))

        def _angle_at_index(angle_name: str, idx: int) -> float:
            if idx < 0 or idx >= len(user_features):
                return float("nan")
            angle = user_features[idx].angles.get(angle_name)
            return float(angle) if angle is not None else float("nan")

        metrics["左肘最大弯曲角度"] = _avg_peak_bend("elbow_left")
        metrics["右肘最大弯曲角度"] = _avg_peak_bend("elbow_right")
        metrics["左手腕朝头部最大弯曲角度"] = _avg_peak_bend("wrist_left_head")
        metrics["右手腕朝头部最大弯曲角度"] = _avg_peak_bend("wrist_right_head")
        metrics["左膝最大弯曲角度"] = _avg_peak_bend("knee_left")
        metrics["右膝最大弯曲角度"] = _avg_peak_bend("knee_right")

        wrist_velocities = np.array(
            [f.velocities.get("right_wrist", np.nan) for f in user_features],
            dtype=float,
        )
        if np.all(np.isnan(wrist_velocities)):
            release_idx = len(user_features) - 1
        else:
            release_idx = int(np.nanargmax(wrist_velocities))
        metrics["右手腕出手前倾角度"] = _angle_at_index("wrist_right_head", release_idx)

        return metrics

    def _compute_running_focus_metrics(self, user_features: list[FeatureSet]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        keypoints_seq = [f.keypoints for f in user_features if f.keypoints.size > 0]
        if not keypoints_seq:
            return metrics

        fps = float(getattr(self.pipeline, "fps", 30.0) or 30.0)
        visibility_threshold = 0.2

        def _nanmean(values: Iterable[float]) -> float:
            arr = np.array(list(values), dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return float("nan")
            return float(np.mean(arr))

        def _cycle_peak_mean(
            series: Iterable[float],
            *,
            mode: str = "max",
            clip_min: float | None = None,
            clip_max: float | None = None,
            min_gap_seconds: float = 0.25,
        ) -> float:
            arr = np.array(list(series), dtype=float)
            if arr.size == 0:
                return float("nan")
            if clip_min is not None:
                arr[arr < clip_min] = np.nan
            if clip_max is not None:
                arr[arr > clip_max] = np.nan
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return float("nan")

            full = np.array(list(series), dtype=float)
            if clip_min is not None:
                full[full < clip_min] = np.nan
            if clip_max is not None:
                full[full > clip_max] = np.nan

            min_gap_frames = max(2, int(min_gap_seconds * fps))
            peaks: list[float] = []
            for idx in range(1, len(full) - 1):
                val = full[idx]
                if np.isnan(val):
                    continue
                prev = full[idx - 1]
                nxt = full[idx + 1]
                if mode == "min":
                    cond_left = np.isnan(prev) or val <= prev
                    cond_right = np.isnan(nxt) or val <= nxt
                else:
                    cond_left = np.isnan(prev) or val >= prev
                    cond_right = np.isnan(nxt) or val >= nxt
                if cond_left and cond_right and (
                    not peaks or idx - peaks[-1][0] >= min_gap_frames
                ):
                    peaks.append((idx, val))
            if not peaks:
                return float(np.nanmean(full[~np.isnan(full)]))
            values = [val for _, val in peaks]
            return float(np.mean(values)) if values else float("nan")

        def _limb_angle_series(start_idx: int, end_idx: int, *, signed: bool = False) -> list[float]:
            series: list[float] = []
            for kps in keypoints_seq:
                if max(start_idx, end_idx) >= kps.shape[0]:
                    series.append(float("nan"))
                    continue
                if kps.shape[1] < 3:
                    conf_start = conf_end = 1.0
                else:
                    conf_start = kps[start_idx, 2]
                    conf_end = kps[end_idx, 2]
                if conf_start < visibility_threshold or conf_end < visibility_threshold:
                    series.append(float("nan"))
                    continue
                vec = kps[end_idx, :2] - kps[start_idx, :2]
                norm = np.linalg.norm(vec)
                if norm < 1e-6:
                    series.append(float("nan"))
                    continue
                angle = float(np.degrees(np.arctan2(vec[0], -vec[1] + 1e-6)))
                if not signed:
                    angle = abs(angle)
                series.append(angle)
            return series

        def _peak_indices(series: list[float], min_gap_frames: int) -> list[int]:
            peaks: list[int] = []
            arr = np.array(series, dtype=float)
            for idx in range(1, len(arr) - 1):
                val = arr[idx]
                if np.isnan(val):
                    continue
                prev = arr[idx - 1]
                nxt = arr[idx + 1]
                cond_left = np.isnan(prev) or val >= prev
                cond_right = np.isnan(nxt) or val >= nxt
                if cond_left and cond_right and (not peaks or idx - peaks[-1] >= min_gap_frames):
                    peaks.append(idx)
            return peaks

        def _step_frequency(series: list[float]) -> float:
            if fps <= 0:
                return float("nan")
            duration = len(series) / fps if len(series) > 0 else 0.0
            if duration <= 0:
                return float("nan")
            min_gap_frames = max(2, int(0.2 * fps))
            peaks = _peak_indices(series, min_gap_frames)
            if not peaks:
                return float("nan")
            return float(len(peaks) / duration)

        upper_left = _limb_angle_series(5, 7)
        upper_right = _limb_angle_series(6, 8)
        metrics["平均大臂最大摆动幅度"] = _nanmean(
            [
                _cycle_peak_mean(upper_left, mode="max", clip_max=90.0),
                _cycle_peak_mean(upper_right, mode="max", clip_max=90.0),
            ]
        )

        thigh_left = _limb_angle_series(11, 13)
        thigh_right = _limb_angle_series(12, 14)
        metrics["平均大腿最大摆动幅度"] = _nanmean(
            [
                _cycle_peak_mean(thigh_left, mode="max", clip_max=150.0),
                _cycle_peak_mean(thigh_right, mode="max", clip_max=150.0),
            ]
        )

        def _knee_angle_series(key: str) -> list[float]:
            return [f.angles.get(key, np.nan) for f in user_features]

        metrics["小腿相对大腿平均最大弯曲幅度"] = _nanmean(
            [
                _cycle_peak_mean(
                    _knee_angle_series("knee_left"),
                    mode="min",
                    clip_min=30.0,
                    clip_max=160.0,
                ),
                _cycle_peak_mean(
                    _knee_angle_series("knee_right"),
                    mode="min",
                    clip_min=30.0,
                    clip_max=160.0,
                ),
            ]
        )

        thigh_left_signed = _limb_angle_series(11, 13, signed=True)
        thigh_right_signed = _limb_angle_series(12, 14, signed=True)
        freq_left = _step_frequency(thigh_left_signed)
        freq_right = _step_frequency(thigh_right_signed)
        metrics["步频（次/秒）"] = _nanmean([freq_left, freq_right])

        return metrics

    @staticmethod
    def _compute_focus_similarity(
        user_metrics: Optional[dict[str, float]],
        reference_metrics: Optional[dict[str, float]],
    ) -> dict[str, float]:
        if not user_metrics or not reference_metrics:
            return {}

        similarities: dict[str, float] = {}
        for key, ref_value in reference_metrics.items():
            user_value = user_metrics.get(key)
            if user_value is None or ref_value != ref_value or user_value != user_value:
                similarities[key] = float("nan")
                continue
            denom = abs(ref_value) + abs(user_value)
            if denom == 0:
                similarities[key] = 1.0
            else:
                similarities[key] = float(1 - abs(user_value - ref_value) / denom)
        return similarities


def build_reference_features_from_video(
    video_path: str | Path,
    *,
    model_path: str | Path,
    device: Optional[str] = None,
    fps: float = 30.0,
    target_fps: Optional[float] = None,
) -> list[FeatureSet]:
    pipeline = DataProcessingPipeline(
        model_path=model_path,
        device=device,
        temp_dir="artifacts/reference_frames",
        fps=fps,
        enable_tracking=True,  # 启用跟踪功能
        enable_smoothing=True,  # 启用平滑功能
        smoothing_window=5,  # 增加平滑窗口到5帧
        tracking_max_distance=0.5,  # 增加跟踪距离阈值以适应更大运动范围
    )
    sample = pipeline.process_video(video_path, target_fps=target_fps)
    return sample.features

