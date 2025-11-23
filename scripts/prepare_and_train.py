from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.data_processing.extract_frames import batch_extract_frames
from src.data_processing.keypoints import PoseEstimator
from src.models import TrainingConfig, train_pose_model

# -----------------------------
# Configuration (edit as needed)
# -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass(slots=True)
class ActivityConfig:
    key: str
    label: str
    raw_videos: list[Path]
    frame_output_root: Path
    dataset_root: Path
    runs_name: str
    class_name: str = "player"
    class_id: int = 0
    target_fps: float | None = None


ACTIVITY_CONFIGS: dict[str, ActivityConfig] = {
    "shooting": ActivityConfig(
        key="shooting",
        label="投篮动作",
        raw_videos=[
            Path(r"E:\data\kobe shoot video\Kobe shoot 01.mp4"),
            Path(r"E:\data\kobe shoot video\Kobe shoot 02.mp4"),
        ],
        frame_output_root=Path(r"E:\data\kobe shoot photo"),
        dataset_root=PROJECT_ROOT / "datasets",
        runs_name="kobe_pose",
        class_name="player",
        class_id=0,
        target_fps=15.0,
    ),
    "running": ActivityConfig(
        key="running",
        label="跑步动作",
        raw_videos=[
            Path(r"E:\data\Denis running video\Denis running.mp4"),
        ],
        frame_output_root=Path(r"E:\data\running_frames"),
        dataset_root=PROJECT_ROOT / "datasets_running",
        runs_name="running_pose",
        class_name="runner",
        class_id=0,
        target_fps=15.0,
    ),
}

DEFAULT_ACTIVITY = "running"

TARGET_FPS_DEFAULT = 15.0  # 提高采样率以减少骨架偏移，建议 15-20 fps
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42
CLEAN_DATASET = True  # 删除旧的 images/labels 再重新生成

MODEL_ARCH = "yolov8n-pose.pt"
TRAIN_EPOCHS = 150
TRAIN_IMGSZ = 640
TRAIN_BATCH = 16
TRAIN_DEVICE = "0"  # default GPU
RUNS_PROJECT = str(PROJECT_ROOT / "runs" / "train")  # 使用项目根目录下的 runs/train

# 自动标注配置
AUTO_ANNOTATE = True
AUTO_MODEL_PATH = Path("yolov8n-pose.pt")
AUTO_CONF_THRESHOLD = 0.25
AUTO_VISIBILITY_THRESHOLD = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备数据并训练 YOLOv8 Pose 模型")
    parser.add_argument(
        "--activity",
        choices=ACTIVITY_CONFIGS.keys(),
        default=os.getenv("ACTIVITY", DEFAULT_ACTIVITY),
        help="选择要处理的动作类型（默认: shooting）",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        type=str,
        help="覆盖默认的原始视频路径列表",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        help="覆盖默认的抽帧 FPS",
    )
    return parser.parse_args()


def ensure_dirs(paths: Iterable[Path], *, must_be_empty: bool = False) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        if must_be_empty:
            if any(path.iterdir()):
                raise RuntimeError(f"Directory {path} is not empty. Please clean it before running.")


def sanitize_filename(path: Path) -> str:
    video_prefix = path.parent.name.replace(" ", "_")
    return f"{video_prefix}_{path.stem}{path.suffix}"


def clean_dataset_root(dataset_root: Path) -> None:
    for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
        target = dataset_root / subdir
        if target.exists():
            shutil.rmtree(target)


def copy_frames(frames: Iterable[Path], destination: Path) -> list[str]:
    copied_files: list[str] = []
    ensure_dirs([destination], must_be_empty=False)

    for frame_path in frames:
        if not frame_path.exists():
            continue
        new_name = sanitize_filename(frame_path)
        target_path = destination / new_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frame_path, target_path)
        copied_files.append(new_name)
    return copied_files


def verify_labels(dataset_root: Path, filenames: Iterable[str], split: str) -> list[str]:
    missing: list[str] = []
    label_dir = dataset_root / "labels" / split
    for filename in filenames:
        label_path = label_dir / Path(filename).with_suffix(".txt")
        if not label_path.exists():
            missing.append(str(label_path))
    return missing


def write_data_yaml(dataset_root: Path, class_name: str) -> Path:
    yaml_path = dataset_root / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    root_str = dataset_root.resolve().as_posix()
    # COCO 17 关键点的水平翻转对称索引
    # 顺序: nose, left_eye, right_eye, left_ear, right_ear,
    #       left_shoulder, right_shoulder, left_elbow, right_elbow,
    #       left_wrist, right_wrist, left_hip, right_hip,
    #       left_knee, right_knee, left_ankle, right_ankle
    flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    yaml_content = "\n".join(
        [
            f"path: {root_str}",
            "train: images/train",
            "val: images/val",
            "test: images/val",
            "names:",
            f"  0: {class_name}",
            "kpt_shape: [17, 3]",
            f"flip_idx: {flip_idx}",
        ]
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def detect_pose(estimator: PoseEstimator, image_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    result = estimator.model.predict(
        source=str(image_path),
        save=False,
        stream=False,
        conf=AUTO_CONF_THRESHOLD,
        verbose=False,
        device=estimator.device,
        iou=estimator.iou,
    )[0]

    if not result.boxes or not result.keypoints:
        return None

    boxes = result.boxes.xywhn.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    keypoints = result.keypoints.xyn.cpu().numpy()
    if boxes.size == 0 or keypoints.size == 0:
        return None

    best_idx = int(np.argmax(scores))
    return boxes[best_idx], keypoints[best_idx]


def write_pose_label(
    label_path: Path,
    bbox_xywhn: np.ndarray,
    keypoints_xyn: np.ndarray,
    *,
    class_id: int,
) -> None:
    entries = [
        str(class_id),
        f"{bbox_xywhn[0]:.6f}",
        f"{bbox_xywhn[1]:.6f}",
        f"{bbox_xywhn[2]:.6f}",
        f"{bbox_xywhn[3]:.6f}",
    ]
    for kp in keypoints_xyn:
        if len(kp) == 3:
            x, y, conf = kp
        elif len(kp) == 2:
            x, y = kp
            conf = 1.0
        else:
            raise ValueError(f"Unexpected keypoint format: {kp}")
        visibility = 1 if conf >= AUTO_VISIBILITY_THRESHOLD else 0
        entries.extend([f"{x:.6f}", f"{y:.6f}", str(visibility)])
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(" ".join(entries) + "\n", encoding="utf-8")


def auto_annotate_split(
    estimator: PoseEstimator,
    image_dir: Path,
    label_dir: Path,
    *,
    class_id: int,
) -> tuple[int, int]:
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(image_dir.glob(ext))
    annotated = 0
    skipped = 0
    for image_path in image_paths:
        label_path = label_dir / (image_path.stem + ".txt")
        pose = detect_pose(estimator, image_path)
        if pose is None:
            skipped += 1
            continue
        bbox_xywhn, keypoints_xyn = pose
        write_pose_label(label_path, bbox_xywhn, keypoints_xyn, class_id=class_id)
        annotated += 1
    return annotated, skipped


def main(
    activity_key: str,
    *,
    raw_videos: list[Path],
    target_fps: float,
) -> None:
    activity = ACTIVITY_CONFIGS[activity_key]
    if not raw_videos:
        raise RuntimeError(
            f"活动「{activity.label}」未配置原始视频，请使用 --videos 参数或在脚本中添加路径。"
        )

    frame_output_root = activity.frame_output_root
    dataset_root = activity.dataset_root
    runs_name = activity.runs_name
    class_name = activity.class_name
    class_id = activity.class_id

    print(f"[{activity.label}] Extracting frames from raw videos...")
    extraction_summary = batch_extract_frames(
        videos=raw_videos,
        output_root=frame_output_root,
        target_fps=target_fps,
        overwrite=False,
    )

    all_frames = [frame for frames in extraction_summary.values() for frame in frames]
    if not all_frames:
        raise RuntimeError("No frames were extracted. Please check video paths and permissions.")

    random.seed(RANDOM_SEED)
    random.shuffle(all_frames)

    split_index = int(len(all_frames) * TRAIN_SPLIT)
    train_frames = all_frames[:split_index]
    val_frames = all_frames[split_index:]

    if CLEAN_DATASET:
        print("Cleaning existing dataset directories...")
        clean_dataset_root(dataset_root)

    images_train_dir = dataset_root / "images" / "train"
    images_val_dir = dataset_root / "images" / "val"
    ensure_dirs([images_train_dir, images_val_dir], must_be_empty=False)
    labels_train_dir = dataset_root / "labels" / "train"
    labels_val_dir = dataset_root / "labels" / "val"
    ensure_dirs([labels_train_dir, labels_val_dir], must_be_empty=False)

    print(f"Copying {len(train_frames)} frames to {images_train_dir}")
    train_filenames = copy_frames(train_frames, images_train_dir)

    print(f"Copying {len(val_frames)} frames to {images_val_dir}")
    val_filenames = copy_frames(val_frames, images_val_dir)

    if AUTO_ANNOTATE:
        print("Running automatic pose annotation...")
        estimator = PoseEstimator(model_path=AUTO_MODEL_PATH, device=TRAIN_DEVICE)
        ann_train, skip_train = auto_annotate_split(
            estimator, images_train_dir, labels_train_dir, class_id=class_id
        )
        ann_val, skip_val = auto_annotate_split(
            estimator, images_val_dir, labels_val_dir, class_id=class_id
        )
        print(
            f"Auto-annotated train: {ann_train} (skipped {skip_train}), "
            f"val: {ann_val} (skipped {skip_val})."
        )
        if skip_train + skip_val > 0:
            print("Some frames were skipped due to missing detections; consider manual review.")

    yaml_path = write_data_yaml(dataset_root, class_name)
    print(f"data.yaml written to {yaml_path}")

    missing_train = verify_labels(dataset_root, train_filenames, "train")
    missing_val = verify_labels(dataset_root, val_filenames, "val")
    missing_labels = missing_train + missing_val

    if missing_labels:
        print("Label files are missing for the following frames:")
        for missing in missing_labels:
            print(" -", missing)
        print(
            "Please complete YOLO Pose keypoint annotations (labels/*.txt) before launching training."
        )
        return

    print(f"[{activity.label}] Launching YOLOv8 pose training...")
    config = TrainingConfig(
        data_yaml=yaml_path,
        model_architecture=MODEL_ARCH,
        epochs=TRAIN_EPOCHS,
        imgsz=TRAIN_IMGSZ,
        batch=TRAIN_BATCH,
        device=TRAIN_DEVICE,
        project=RUNS_PROJECT,
        name=runs_name,
    )

    best_weights = train_pose_model(config)
    print(f"Training completed. Best weights saved at: {best_weights}")
    model_path_dir = PROJECT_ROOT / "artifacts" / "models"
    model_path_dir.mkdir(parents=True, exist_ok=True)
    model_path_file = model_path_dir / f"{activity.key}_model_path.txt"
    model_path_file.write_text(str(best_weights), encoding="utf-8")

    env_path = PROJECT_ROOT / ".env"
    env_var = "MODEL_PATH" if activity.key == "shooting" else "RUN_MODEL_PATH"
    model_path_line = f"{env_var}={best_weights}\n"
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        updated = False
        for idx, line in enumerate(lines):
            if line.startswith(f"{env_var}="):
                lines[idx] = model_path_line.strip()
                updated = True
                break
        if not updated:
            lines.append(model_path_line.strip())
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        env_path.write_text(model_path_line, encoding="utf-8")
    print(f"{env_var} written to .env")


if __name__ == "__main__":
    args = parse_args()
    selected_activity = args.activity
    activity_config = ACTIVITY_CONFIGS[selected_activity]
    videos_override = [Path(p) for p in args.videos] if args.videos else None
    videos = videos_override or activity_config.raw_videos
    fps = args.target_fps or activity_config.target_fps or TARGET_FPS_DEFAULT
    main(selected_activity, raw_videos=videos, target_fps=fps)

