# 训练脚本运行顺序指南

本文档说明项目中各个训练脚本的运行顺序和依赖关系。

## 📋 训练流程概览

```
1. 训练YOLOv8姿态估计模型（投篮）
   ↓
2. 训练YOLOv8姿态估计模型（跑步）
   ↓
3. 生成参考特征文件（可选，如果已有可跳过）
   ↓
4. 训练关键词推荐模型（XGBoost）
```

## 🔄 详细训练步骤

### 步骤 1: 训练投篮动作姿态估计模型

**脚本**: `scripts/prepare_and_train.py`

**命令**:
```bash
python scripts/prepare_and_train.py --activity shooting
```

**功能**:
- 从原始视频中提取帧（默认15 FPS）
- 自动标注关键点（如果启用）
- 准备训练数据集
- 训练YOLOv8姿态估计模型
- 保存模型到 `runs/train/kobe_pose*/weights/best.pt`
- 更新配置文件

**前置条件**:
- 确保在 `scripts/prepare_and_train.py` 中配置了投篮视频路径
- 或使用 `--videos` 参数指定视频文件

**预计时间**: 根据数据集大小和GPU性能，通常需要30分钟到数小时

---

### 步骤 2: 训练跑步动作姿态估计模型

**脚本**: `scripts/prepare_and_train.py`

**命令**:
```bash
python scripts/prepare_and_train.py --activity running --videos <视频路径1> <视频路径2> ...
```

**功能**:
- 从跑步视频中提取帧
- 自动标注关键点
- 准备跑步训练数据集
- 训练YOLOv8姿态估计模型
- 保存模型到 `runs/train/running_pose/weights/best.pt`

**前置条件**:
- 准备跑步动作的视频文件
- 使用 `--videos` 参数指定视频路径

**预计时间**: 根据数据集大小和GPU性能，通常需要30分钟到数小时

---

### 步骤 3: 生成参考特征文件（可选）

如果还没有参考特征文件，需要从参考视频生成：

**投篮参考特征**:
```python
from src.models.inference import build_reference_features_from_video
from src.utils.serialization import save_features_to_file

features = build_reference_features_from_video(
    video_path="参考视频路径.mp4",
    model_path="runs/train/kobe_pose2/weights/best.pt",
    device="0",
    target_fps=15.0,
)
save_features_to_file(features, "artifacts/reference_features.json")
```

**跑步参考特征**:
```python
features = build_reference_features_from_video(
    video_path="参考视频路径.mp4",
    model_path="runs/train/running_pose/weights/best.pt",
    device="0",
    target_fps=15.0,
)
save_features_to_file(features, "artifacts/reference_features_running.json")
```

**注意**: 如果已有参考特征文件，可以跳过此步骤。

---

### 步骤 4: 训练关键词推荐模型

**脚本**: `scripts/train_keyword_recommender.py`

**命令**:
```bash
python scripts/train_keyword_recommender.py
```

**功能**:
- 读取关键词库 (`artifacts/keyword_library/search_keywords.json`)
- 为跑步和篮球分别训练XGBoost模型
- 使用TF-IDF向量化文本特征
- 保存模型到 `artifacts/keyword_models/`

**生成文件**:
- `artifacts/keyword_models/running_keyword_model.json`
- `artifacts/keyword_models/running_vectorizer.pkl`
- `artifacts/keyword_models/running_keyword_mapping.json`
- `artifacts/keyword_models/basketball_keyword_model.json`
- `artifacts/keyword_models/basketball_vectorizer.pkl`
- `artifacts/keyword_models/basketball_keyword_mapping.json`

**前置条件**:
- 确保已安装 `xgboost` 和 `scikit-learn`
- 关键词库文件已存在

**预计时间**: 通常只需要几秒钟到几分钟

---

## 🚀 快速开始（完整流程）

如果你想一次性完成所有训练，可以按以下顺序执行：

```bash
# 1. 训练投篮模型
python scripts/prepare_and_train.py --activity shooting

# 2. 训练跑步模型（需要提供视频路径）
python scripts/prepare_and_train.py --activity running --videos <你的跑步视频路径>

# 3. 训练关键词推荐模型
python scripts/train_keyword_recommender.py
```

## ⚠️ 注意事项

1. **依赖关系**:
   - 关键词推荐模型不依赖姿态估计模型，可以独立训练
   - 但Web应用需要姿态估计模型才能正常工作

2. **GPU要求**:
   - YOLOv8模型训练需要GPU（推荐）
   - 关键词推荐模型训练可以在CPU上运行

3. **时间估算**:
   - 姿态估计模型训练：30分钟 - 数小时（取决于数据集和GPU）
   - 关键词推荐模型训练：几秒 - 几分钟

4. **数据准备**:
   - 确保视频文件路径正确
   - 确保有足够的磁盘空间存储数据集和模型

## 📝 验证训练结果

训练完成后，检查以下文件是否存在：

**姿态估计模型**:
- `runs/train/kobe_pose2/weights/best.pt` (投篮)
- `runs/train/running_pose/weights/best.pt` (跑步)

**关键词推荐模型**:
- `artifacts/keyword_models/running_keyword_model.json`
- `artifacts/keyword_models/basketball_keyword_model.json`

**参考特征**:
- `artifacts/reference_features.json` (投篮)
- `artifacts/reference_features_running.json` (跑步)

## 🔧 故障排除

如果训练过程中遇到问题：

1. **模型训练失败**: 检查GPU内存是否足够，可以减小batch size
2. **关键词模型训练失败**: 确保已安装所有依赖 `pip install -r requirements.txt`
3. **路径错误**: 检查视频文件路径是否正确，使用绝对路径更安全

