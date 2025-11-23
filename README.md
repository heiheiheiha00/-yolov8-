# 基于 YOLOv8 的篮球投篮动作捕捉与评估系统

本项目提供从数据准备、模型训练、推理评估到 Web 可视化的一整套工作流，帮助教练或运动员对投篮动作进行标准化分析与优化。

## 功能模块

1. **数据处理**：支持视频抽帧、姿态关键点提取、特征（角度/速度/发力）计算。
2. **跟踪与平滑**：内置人物跟踪算法和时序平滑，减少人物晃动时骨架脱离的问题。
3. **模型训练**：使用 `ultralytics` 训练 YOLOv8 Pose 模型，支持自定义数据集。
4. **推理比对**：对用户上传视频进行关键点提取与特征计算，并与标准动作对比。
5. **Web 接口**：基于 Flask 提供视频上传、结果展示（角度曲线图、建议文本）的 Web UI。
6. **可视化输出**：绘制关节角度曲线、在视频帧上叠加骨架、生成最终反馈。
7. **多动作扩展**：同一套流水线支持「投篮」「跑步」两套模型，可继续扩展更多项目。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

建议使用具备 NVIDIA GPU 的环境，并安装对应的 CUDA/cuDNN。

## 数据准备

### 视频抽帧

**注意**：为提高骨架识别稳定性，建议使用 **15-20 FPS** 的采样率。较低的采样率（如 8 FPS）可能导致动作周期中某些关键帧出现骨架偏移。

```python
# 单个视频，按 15 FPS 抽帧（推荐）
from src.data_processing import extract_video_frames

frames = extract_video_frames(
    video_path=r"videos/shoot.mp4",
    output_dir=r"datasets/raw_frames/shoot",
    target_fps=15,  # 建议 15-20 fps 以减少骨架偏移
)
```

批量抽帧示例（本项目中已内置在 `src/data_processing/extract_frames.py` 的主模块）：

```python
from src.data_processing import batch_extract_frames

videos = [
    r"E:\data\kobe shoot video\Kobe shoot 01.mp4",
    r"E:\data\kobe shoot video\Kobe shoot 02.mp4",
]

summary = batch_extract_frames(
    videos=videos,
    output_root=r"E:\data\kobe shoot photo",
    target_fps=15,  # 建议 15-20 fps 以减少骨架偏移
)
for video, frames in summary.items():
    print(video, len(frames))
```

### 自动整理数据集并启动训练

`scripts/prepare_and_train.py` 现已支持 **投篮 (shooting)** 与 **跑步 (running)** 两套数据流程：

1. 按配置抽帧（默认 15 FPS，可使用 `--target-fps` 覆盖）；
2. 将帧拷贝至对应数据集目录（投篮使用 `datasets/`，跑步使用 `datasets_running/`）；
3. （可选）使用预训练 YOLOv8 Pose 权重自动生成 YOLO Pose 标签；
4. 写入各自的 `data.yaml`；
5. 校验标签完备性后启动 YOLOv8 Pose 训练；
6. 自动将最优权重写入 `artifacts/models/<activity>_model_path.txt`，并更新 `.env` 中的 `MODEL_PATH`（投篮）或 `RUN_MODEL_PATH`（跑步）。

**命令示例：**

```bash
# 投篮动作（默认配置）
python scripts/prepare_and_train.py --activity shooting

# 跑步动作（需提供跑步原始视频）
python scripts/prepare_and_train.py --activity running --videos E:\data\running\run01.mp4 E:\data\running\run02.mp4

# 如需更换抽帧速率
python scripts/prepare_and_train.py --activity running --videos ... --target-fps 18
```

> 不带 `--activity` 参数运行脚本时，默认训练“跑步”模型；如果需要训练投篮模型，记得显式传 `--activity shooting`，两套模型互不覆盖。

**注意事项**

- 修改采样率或替换原始视频后，需要重新运行脚本以更新整个数据集。
- 脚本默认 `CLEAN_DATASET = True`，会清空对应数据集目录，请确认无重要文件再运行。
- 跑步数据默认写入 `runs/train/running_pose/`，其权重会写入 `.env` 的 `RUN_MODEL_PATH`，供前端/推理切换使用。

> ⚠️ YOLO Pose 训练要求每帧都有对应的 `labels/*.txt`。若自动标注跳过了部分帧，可根据脚本输出路径手动补齐。

### 自动关键点标注（可选）

若想利用预训练 YOLOv8 Pose 模型自动生成初始关键点标注（再做人工校正），可运行：

```bash
python scripts/auto_annotate_pose.py \
    --model yolov8n-pose.pt \
    --images datasets/images/train datasets/images/val \
    --labels datasets/labels \
    --device 0 \
    --conf 0.25
```

脚本会为检测到的主目标写入 `labels/<split>/xxx.txt`，格式符合 YOLO Pose 要求。若某帧未检出人体，则会跳过并在终端报告。

> `prepare_and_train.py` 已集成自动标注逻辑（可通过 `AUTO_ANNOTATE`、`AUTO_MODEL_PATH` 等配置项开关及调整）。如需完全手动标注，可关闭该开关或在执行前清空生成的标签后重新运行。

### 姿态关键点提取

```python
from src.data_processing import PoseEstimator

estimator = PoseEstimator(model_path="yolov8n-pose.pt")
results = estimator.predict_on_frames(frames, save_json="artifacts/keypoints/shoot.json")
```

### 特征计算

```python
from src.data_processing import compute_feature_vector
import numpy as np

keypoints = [result.keypoints[0] for result in results if result.keypoints.size > 0]
features = compute_feature_vector(keypoints, fps=30)
```

### 标注指导

如果需要对抽帧后的数据进行人工标注（用于训练标准投篮数据集），请遵循以下原则：

- **帧选择**：覆盖完整的投篮动作流程：持球准备 → 下蹲蓄力 → 起跳 → 出手 → 跳投顶点 → 落地。
- **关键帧**：每个关键动作阶段至少选择 3-5 帧，确保姿态变化连续。
- **标签格式**：建议使用 `ultralytics` Pose 数据格式，JSON/YAML 中指定每帧的关键点坐标（17 个/body25）。
- **命名规范**：`<球员ID>_<动作阶段>_<帧序号>.jpg`，便于脚本关联。
- **质量控制**：标注后进行复核，确保关键点位置准确（肩、肘、腕、髋、膝、踝等）。

可利用 [Label Studio](https://labelstud.io/) 或 [Roboflow](https://roboflow.com/) 的 Pose 标注插件。

### 跑步动作数据补充

- **视频准备**：保持头到脚完整入镜，最好选择干净背景和固定机位，减少人物快速偏移。
- **抽帧建议**：依旧推荐 15-20 FPS，可在脚本中使用 `--target-fps` 覆盖。
- **标注关注点**：重点关注髋、膝、踝的周期性变化，可在后续扩展为更加细致的步态参数。
- **数据目录**：跑步动作默认使用 `datasets_running/` 与 `runs/train/running_pose/`，避免与投篮数据混淆。
- **重点指标**：跑步推理会额外输出「平均大臂最大摆动幅度」「平均大腿最大摆动幅度（步幅）」「小腿相对大腿平均最大弯曲幅度」以及「步频（次/秒）」等关键指标。

## 模型训练

准备 `data.yaml`：

```yaml
path: datasets
train: images/train
val: images/val
test: images/test
names:
  0: player
kpt_shape: [17, 3]
```

启动训练：

```python
from src.models import TrainingConfig, train_pose_model

config = TrainingConfig(
    data_yaml="datasets/data.yaml",
    model_architecture="yolov8n-pose.pt",
    epochs=150,
    imgsz=640,
    batch=32,
    device="0",  # GPU ID
    project="runs/train",
    name="shooting_pose",
)

best_weights = train_pose_model(config)
print("Best weights saved at:", best_weights)
```

## 推理与动作比对

```python
from src.models.inference import PoseInferenceService, build_reference_features_from_video
from src.utils.serialization import save_features_to_file

reference_features = build_reference_features_from_video(
    "videos/reference.mp4",
    model_path="artifacts/models/best.pt",
    device="0",
)
save_features_to_file(reference_features, "artifacts/reference_features.json")

service = PoseInferenceService(
    model_path="artifacts/models/best.pt",
    reference_features=reference_features,
    device="0",
)

result = service.compare("videos/user.mp4")
print(result.similarity_scores)
print(result.recommendation)
```

### 推理测试流程

1. **准备参考特征**
   ```bash
   python - <<'PY'
   from src.models.inference import build_reference_features_from_video
   from src.utils.serialization import save_features_to_file

   features = build_reference_features_from_video(
       video_path=r"E:\data\kobe shoot video\Kobe shoot 01.mp4",
       model_path="artifacts/models/best.pt",
       device="0",
   )
   save_features_to_file(features, "artifacts/reference_features.json")
   PY
   ```
   如果已有 `artifacts/reference_features.json`，可跳过该步骤。

2. **运行离线推理对比**
   ```bash
   python - <<'PY'
   from src.models.inference import PoseInferenceService
   from src.utils.serialization import load_features_from_file

   reference = load_features_from_file("artifacts/reference_features.json")
   service = PoseInferenceService(
       model_path="artifacts/models/best.pt",
       reference_features=reference,
       device="0",
   )
   result = service.compare(r"E:\data\kobe shoot video\Kobe shoot 02.mp4")
   print("Similarity:", result.similarity_scores)
   print("Advice:", "\n".join(result.recommendation))
   print("Angle plot saved to:", result.angle_plot_path)
   PY
   ```
   完成后可在 `artifacts/results` 下查看角度曲线图、文本建议等输出。

## Web 服务

1. 准备 `.env`（示例）：

   ```
   FLASK_SECRET_KEY=change-me
   DEVICE=0
   TARGET_FPS=15.0
   # 投篮模型 & 参考特征
   MODEL_PATH=runs/train/kobe_pose/weights/best.pt
   REFERENCE_FEATURES_PATH=artifacts/reference_features_shooting.json
   REFERENCE_VIDEO_PATH=E:\data\kobe shoot video\Kobe shoot 01.mp4
   # 跑步模型 & 参考特征
   RUN_MODEL_PATH=runs/train/running_pose/weights/best.pt
   RUN_REFERENCE_FEATURES_PATH=artifacts/reference_features_running.json
   RUN_REFERENCE_VIDEO_PATH=E:\data\running\reference.mp4
   ```

2. 启动：

   ```bash
   export FLASK_APP=src.web.app:create_app  # Windows: set FLASK_APP=src.web.app:create_app
   flask run --host=0.0.0.0 --port=5000
   ```

   打开浏览器访问 `http://localhost:5000` 上传视频并查看结果。

### 前端测试步骤

1. **在 PyCharm / 命令行运行离线推理**
   ```bash
   python scripts/run_inference.py --activity shooting --user-video E:\data\kobe shoot video\Kobe shoot 02.mp4
   python scripts/run_inference.py --activity running --user-video E:\data\running\runner_test.mp4 --target-fps 15
   ```
   - `--activity`：`shooting` 或 `running`
   - `--reference-video` / `--reference-features`：可覆盖默认参考
   - 控制台会输出相似度、建议及生成资源路径

2. **确保 `.env` 与模型文件齐备**：参考前述示例，需包含 `MODEL_PATH` 与 `RUN_MODEL_PATH` 两套配置。
3. **启动 Web 服务**
   ```bash
   set FLASK_APP=src.web.app:create_app   # PowerShell: $env:FLASK_APP="src.web.app:create_app"
   flask run --host=0.0.0.0 --port=5000
   ```
4. **访问前端**
   - 浏览器打开 `http://localhost:5000`
   - 在下拉框中选择「投篮」或「跑步」，再上传对应视频
   - 等待分析完成，查看角度曲线、相似度柱状图、建议、骨架叠加视频等
5. **验证输出**
   - 结果文件存放在 `artifacts/web_results/<session_id>/`
   - 包括 `*_angles.png`、`advice.txt` 等，可下载或进一步处理

## 可视化

- `src.visualization.plots.render_angle_curves`：生成关节角度折线图。
- `src.visualization.video.annotate_video_with_pose`：在视频帧上绘制骨架并导出新视频。

## 目录结构

```
.
├── artifacts/                    # 抽帧、训练权重、推理结果等产物
│   ├── models/                   # 训练完成的 pt 权重、model_path.txt
│   ├── reference_features.json   # 基准动作特征（可选）
│   ├── uploads/                  # Web 上传的临时视频
│   └── web_results/              # Web 输出图像与建议
├── datasets/
│   ├── data.yaml                 # 自动生成的数据集配置
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── runs/                         # Ultralytics 训练日志与权重备份
├── scripts/
│   ├── prepare_and_train.py      # 抽帧→标注→训练一体化脚本
│   ├── auto_annotate_pose.py     # 独立自动标注脚本
│   └── run_inference.py          # 离线推理测试脚本
├── src/
│   ├── data_processing/
│   ├── models/
│   ├── utils/
│   ├── visualization/
│   └── web/
├── requirements.txt
└── README.md
```

## 运行方式概览

1. **一键数据准备 + 训练**
   ```bash
   python scripts/prepare_and_train.py --activity shooting
   python scripts/prepare_and_train.py --activity running --videos <跑步视频...>
   ```
   自动完成抽帧、（可选）自动标注、写入数据集并启动 YOLOv8 Pose 训练。最优权重位于 `runs/train/<name>/weights/best.pt`，并写入 `.env` 对应的 `MODEL_PATH` / `RUN_MODEL_PATH`。

2. **离线推理验证（PyCharm 或命令行）**
   ```bash
   python scripts/run_inference.py --activity shooting --user-video <待测视频>
   python scripts/run_inference.py --activity running --user-video <待测跑步视频>
   ```
   脚本会在缺少参考特征时自动根据配置的视频生成后再进行比对，并输出角度曲线、关键节点/重点关节相似度图以及建议。

3. **启动 Web 前端（内置投篮/跑步切换）**
   ```bash
   python src/web/app.py
   # 或
   set FLASK_APP=src.web.app:create_app
   flask run --host=0.0.0.0 --port=5000
   ```
   浏览器访问 `http://localhost:5000`，在页面上选择「投篮」或「跑步」并上传视频即可查看角度曲线、关键节点相似度、重点关节相似度柱状图、重点动作参数表以及骨架叠加视频。运行前确保 `.env` 中 `MODEL_PATH` / `RUN_MODEL_PATH` 及对应的参考特征路径配置正确。

## 跟踪与平滑优化

系统内置了人物跟踪和时序平滑功能，以解决人物晃动时骨架脱离的问题：

- **人物跟踪**：基于关键点中心距离和相似度，自动跟踪同一人物跨帧移动，避免在人物晃动时选择错误的目标。默认跟踪距离阈值为 0.5（归一化坐标），可适应较大的运动范围。
- **时序平滑**：使用移动平均窗口（默认 5 帧）对关键点位置进行平滑，减少检测抖动和异常值，提供更稳定的骨架跟踪。
- **自动启用**：在 `app.py` 和推理服务中，跟踪和平滑功能默认启用，无需额外配置。

**当前配置参数**：
- `enable_tracking=True`：跟踪功能已启用
- `enable_smoothing=True`：平滑功能已启用
- `smoothing_window=5`：平滑窗口为 5 帧（已从 3 帧提升）
- `tracking_max_distance=0.5`：跟踪距离阈值 0.5（已从 0.3 提升，适应更大运动范围）

如果遇到骨架脱离问题，可以进一步：
1. 提高视频采样率（15-20 FPS）以获得更多帧数据
2. 在代码中调整 `tracking_max_distance`（当前 0.5，可增加到 0.7-0.8）以适应更大的运动范围
3. 增加 `smoothing_window`（当前 5，可增加到 7 或 9）以获得更平滑的结果

## 未来工作

- 引入时序模型（LSTM/Transformer）提升动作状态识别能力。
- 增加多摄像头融合、3D 关键点估计。
- 集成更先进的跟踪算法（如 DeepSORT、ByteTrack）。
- 提供移动端/小程序上传入口。

欢迎提出 Issue 或 PR！

