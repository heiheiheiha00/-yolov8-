from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.data_processing.keypoints import PoseEstimator


def detect_single_pose(estimator: PoseEstimator, image_path: Path, conf_threshold: float) -> Optional[tuple[np.ndarray, np.ndarray]]:
    result = estimator.model.predict(
        source=str(image_path),
        save=False,
        stream=False,
        conf=conf_threshold,
        verbose=False,
        device=estimator.device,
        iou=estimator.iou,
    )[0]

    if not result.boxes or not result.keypoints:
        return None

    boxes = result.boxes.xywhn.cpu().numpy()  # normalized xywh
    scores = result.boxes.conf.cpu().numpy()
    keypoints = result.keypoints.xyn.cpu().numpy()

    if boxes.size == 0:
        return None

    best_idx = int(np.argmax(scores))
    return boxes[best_idx], keypoints[best_idx]


def write_yolo_pose_label(
    output_path: Path,
    class_id: int,
    bbox_xywhn: np.ndarray,
    keypoints_xyn: np.ndarray,
    visibility_threshold: float = 0.2,
) -> None:
    """
    YOLO Pose label format:
    class cx cy w h x1 y1 v1 x2 y2 v2 ... (all normalized to [0, 1])
    """
    keypoint_entries: list[str] = []
    for kp in keypoints_xyn:
        x, y, conf = kp
        visibility = 1 if conf >= visibility_threshold else 0
        keypoint_entries.extend([f"{x:.6f}", f"{y:.6f}", str(visibility)])

    line = " ".join(
        [str(class_id)]
        + [f"{bbox_xywhn[0]:.6f}", f"{bbox_xywhn[1]:.6f}", f"{bbox_xywhn[2]:.6f}", f"{bbox_xywhn[3]:.6f}"]
        + keypoint_entries
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(line + "\n", encoding="utf-8")


def auto_annotate_images(
    image_paths: Iterable[Path],
    labels_root: Path,
    *,
    model_path: Path,
    device: Optional[str],
    conf_threshold: float,
    class_id: int,
) -> tuple[int, int]:
    estimator = PoseEstimator(model_path=model_path, device=device, conf=conf_threshold)

    annotated = 0
    skipped = 0

    for image_path in tqdm(list(image_paths), desc="Annotating images"):
        label_path = labels_root / image_path.parent.name / (image_path.stem + ".txt")

        detection = detect_single_pose(estimator, image_path, conf_threshold)
        if detection is None:
            skipped += 1
            continue

        bbox_xywhn, keypoints_xyn = detection
        write_yolo_pose_label(label_path, class_id, bbox_xywhn, keypoints_xyn)
        annotated += 1

    return annotated, skipped


def gather_image_paths(directories: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for directory in directories:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(directory.glob(ext))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate pose keypoints using a YOLOv8 pose model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the YOLOv8 pose weights (.pt).")
    parser.add_argument("--images", type=Path, nargs="+", required=True, help="Directories containing images to annotate.")
    parser.add_argument("--labels", type=Path, required=True, help="Output root directory for label files.")
    parser.add_argument("--device", type=str, default=None, help="Device for inference, e.g., '0' for GPU, 'cpu' for CPU.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections.")
    parser.add_argument("--class-id", type=int, default=0, help="Class ID to use in labels.")

    args = parser.parse_args()

    image_dirs = [path for path in args.images if path.exists()]
    if not image_dirs:
        raise FileNotFoundError("No valid image directories provided.")

    image_paths = gather_image_paths(image_dirs)
    if not image_paths:
        raise FileNotFoundError("No images found in the provided directories.")

    annotated, skipped = auto_annotate_images(
        image_paths,
        labels_root=args.labels,
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        class_id=args.class_id,
    )

    print(f"Annotation complete. Annotated: {annotated}, skipped (no detection): {skipped}")


if __name__ == "__main__":
    main()

