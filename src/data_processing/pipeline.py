from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .extract_frames import extract_video_frames
from .features import FeatureSet, compute_feature_vector
from .keypoints import KeypointResult, PoseEstimator
from .tracking import (
    SYMMETRIC_KEYPOINT_PAIRS,
    smooth_keypoints_temporal,
    track_person_across_frames,
    track_person_across_frames_simple,
)


@dataclass(slots=True)
class ProcessedSample:
    video_path: Path
    frame_paths: list[Path]
    keypoints: list[KeypointResult]
    features: list[FeatureSet]


class DataProcessingPipeline:
    """
    High level pipeline that takes a raw video and produces feature vectors.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: Optional[str] = None,
        temp_dir: str | Path = "artifacts/frames",
        fps: float = 30.0,
        enable_tracking: bool = True,
        enable_smoothing: bool = True,
        smoothing_window: int = 5,
        smoothing_mode: str = "median",
        tracking_max_distance: float = 0.5,
        tracking_mode: str = "enhanced",
    ) -> None:
        self.pose_estimator = PoseEstimator(model_path=model_path, device=device)
        self.temp_dir = Path(temp_dir)
        self.fps = fps
        self.enable_tracking = enable_tracking
        self.enable_smoothing = enable_smoothing
        self.smoothing_window = smoothing_window
        self.smoothing_mode = smoothing_mode
        self.tracking_max_distance = tracking_max_distance
        if tracking_mode not in {"enhanced", "simple"}:
            raise ValueError("tracking_mode must be 'enhanced' or 'simple'")
        self.tracking_mode = tracking_mode

    def process_video(
        self,
        video_path: str | Path,
        *,
        frame_stride: int = 1,
        target_fps: Optional[float] = None,
        cleanup: bool = False,
    ) -> ProcessedSample:
        frame_dir = self.temp_dir / Path(video_path).stem
        frame_paths = extract_video_frames(
            video_path, frame_dir, every_n_frames=frame_stride, target_fps=target_fps
        )
        keypoint_results = self.pose_estimator.predict_on_frames(frame_paths)

        # Track person across frames and apply smoothing
        if self.enable_tracking:
            if self.tracking_mode == "enhanced":
                tracked_keypoints = track_person_across_frames(
                    keypoint_results,
                    max_distance=self.tracking_max_distance,
                    symmetry_pairs=SYMMETRIC_KEYPOINT_PAIRS,
                )
            else:
                tracked_keypoints = track_person_across_frames_simple(
                    keypoint_results,
                    max_distance=self.tracking_max_distance,
                )
        else:
            # Fallback to simple selection
            tracked_keypoints = [
                result.keypoints[0] if result.keypoints.size > 0 else np.full((17, 3), np.nan)
                for result in keypoint_results
            ]
        
        # Apply temporal smoothing
        if self.enable_smoothing and len(tracked_keypoints) > 1:
            keypoints = smooth_keypoints_temporal(
                tracked_keypoints,
                window_size=self.smoothing_window,
                mode=self.smoothing_mode,
            )
        else:
            keypoints = tracked_keypoints
        
        features = compute_feature_vector(keypoints, fps=self.fps)

        sample = ProcessedSample(
            video_path=Path(video_path),
            frame_paths=frame_paths,
            keypoints=keypoint_results,
            features=features,
        )

        if cleanup:
            for frame_path in frame_paths:
                frame_path.unlink(missing_ok=True)
            if not any(frame_dir.iterdir()):
                frame_dir.rmdir()

        return sample

    def process_batch(
        self,
        videos: Iterable[str | Path],
        *,
        frame_stride: int = 1,
    ) -> list[ProcessedSample]:
        return [
            self.process_video(video, frame_stride=frame_stride)
            for video in videos
        ]

