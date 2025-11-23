from __future__ import annotations

import math
from pathlib import Path
from typing import Generator, Iterable, Optional

import cv2


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    every_n_frames: int = 1,
    target_fps: Optional[float] = None,
    overwrite: bool = False,
) -> list[Path]:
    """
    Extract frames from a video file using OpenCV.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    output_dir:
        Directory where extracted frames will be stored.
    every_n_frames:
        Extract one frame every `every_n_frames`. Ignored if `target_fps` is provided.
    target_fps:
        If provided, down-sample the video to the desired FPS. The function will compute
        the stride automatically based on the source FPS.
    overwrite:
        Whether to overwrite existing frames in the output directory.

    Returns
    -------
    list[Path]
        List of file paths for the extracted frames.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for image_path in output_dir.glob("*.jpg"):
            image_path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    stride = every_n_frames
    if target_fps is not None and target_fps > 0:
        stride = max(1, math.floor(source_fps / target_fps))

    frame_paths: list[Path] = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            frame_file = output_dir / f"{video_path.stem}_{saved_idx:06d}.jpg"
            success = cv2.imwrite(str(frame_file), frame)
            if not success:
                raise RuntimeError(f"Failed to write frame to disk: {frame_file}")
            frame_paths.append(frame_file)
            saved_idx += 1
        frame_idx += 1

    cap.release()

    if not frame_paths:
        raise RuntimeError(
            f"No frames were extracted from {video_path}. "
            f"Check the stride ({stride}) and video length ({total_frames} frames)."
        )

    return frame_paths


def iter_video_frames(
    video_path: str | Path,
    *,
    every_n_frames: int = 1,
) -> Generator[tuple[int, "cv2.Mat"], None, None]:
    """
    Yield frames from a video without writing them to disk.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    every_n_frames:
        Yield one frame every `every_n_frames`.

    Yields
    ------
    tuple[int, cv2.Mat]
        Frame index and the corresponding frame matrix.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every_n_frames == 0:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def batch_extract_frames(
    videos: Iterable[str | Path],
    output_root: str | Path,
    *,
    every_n_frames: int = 1,
    target_fps: Optional[float] = None,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """
    Extract frames for a batch of videos.

    Parameters
    ----------
    videos:
        Iterable with paths to the video files.
    output_root:
        Root directory where per-video frame folders will be created.
    every_n_frames:
        Extract one frame every `every_n_frames`.
    overwrite:
        Whether to overwrite existing frames in output directories.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, list[Path]] = {}
    for video in videos:
        video_path = Path(video)
        target_dir = output_root / video_path.stem
        summary[str(video_path)] = extract_video_frames(
            video_path=video_path,
            output_dir=target_dir,
            every_n_frames=every_n_frames,
            target_fps=target_fps,
            overwrite=overwrite,
        )
    return summary


if __name__ == "__main__":
    SAMPLE_VIDEOS = [
        Path(r"E:\data\kobe shoot video\Kobe shoot 01.mp4"),
        Path(r"E:\data\kobe shoot video\Kobe shoot 02.mp4"),
    ]
    OUTPUT_DIR = Path(r"E:\data\kobe shoot photo")

    summary = batch_extract_frames(
        videos=SAMPLE_VIDEOS,
        output_root=OUTPUT_DIR,
        target_fps=8.0,
        overwrite=False,
    )
    for video, frames in summary.items():
        print(f"{video}: {len(frames)} frames saved to {frames[0].parent if frames else 'N/A'}")

