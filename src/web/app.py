from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from src.config import ActivitySpec, PROJECT_ROOT, get_activity_specs
from src.models.inference import PoseInferenceService, build_reference_features_from_video
from src.utils.serialization import load_features_from_file, save_features_to_file


load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", PROJECT_ROOT / "artifacts" / "uploads"))
RESULT_DIR = Path(os.getenv("RESULT_DIR", PROJECT_ROOT / "artifacts" / "web_results"))
TARGET_FPS = float(os.getenv("TARGET_FPS", "15.0")) if os.getenv("TARGET_FPS") else None

ACTIVITY_SPECS = get_activity_specs()
ACTIVITY_OPTIONS = list(ACTIVITY_SPECS.values())
DEFAULT_ACTIVITY_KEY = ACTIVITY_OPTIONS[0].key if ACTIVITY_OPTIONS else "shooting"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")
app.config["UPLOAD_MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

REFERENCE_CACHE: dict[str, list] = {}
INFERENCE_SERVICES: dict[str, PoseInferenceService] = {}


def _ensure_model_exists(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到模型文件: {model_path}\n"
            "请先完成对应动作的训练，或在 .env 中更新模型路径。"
        )


def _load_reference_features_for_activity(spec: ActivitySpec) -> list:
    if spec.key in REFERENCE_CACHE:
        return REFERENCE_CACHE[spec.key]

    if spec.reference_features_path and spec.reference_features_path.exists():
        features = load_features_from_file(spec.reference_features_path)
        REFERENCE_CACHE[spec.key] = features
        return features

    if spec.reference_video_path and spec.reference_video_path.exists():
        features = build_reference_features_from_video(
            spec.reference_video_path,
            model_path=spec.model_path,
            device=os.getenv("DEVICE"),
            target_fps=TARGET_FPS,
        )
        if spec.reference_features_path:
            spec.reference_features_path.parent.mkdir(parents=True, exist_ok=True)
            save_features_to_file(features, spec.reference_features_path)
        REFERENCE_CACHE[spec.key] = features
        return features

    raise RuntimeError(
        f"活动「{spec.label}」缺少参考特征，请设置 {spec.key.upper()}_REFERENCE_FEATURES_PATH 或 "
        f"{spec.key.upper()}_REFERENCE_VIDEO_PATH。"
    )


def _get_inference_service(activity_key: str) -> PoseInferenceService:
    if activity_key in INFERENCE_SERVICES:
        return INFERENCE_SERVICES[activity_key]

    if activity_key not in ACTIVITY_SPECS:
        raise KeyError(f"未知的动作类型: {activity_key}")

    spec = ACTIVITY_SPECS[activity_key]
    _ensure_model_exists(spec.model_path)
    reference_features = _load_reference_features_for_activity(spec)
    service = PoseInferenceService(
        model_path=spec.model_path,
        reference_features=reference_features,
        device=os.getenv("DEVICE"),
        activity_key=spec.key,
        key_node_angles=spec.key_node_angles,
        focus_angle_keys=spec.focus_angle_keys,
        enable_focus_metrics=spec.enable_focus_metrics,
        recommendation_thresholds=spec.recommendation_thresholds,
        recommendation_messages=spec.recommendation_messages,
        default_recommendation=spec.default_recommendation,
    )
    INFERENCE_SERVICES[activity_key] = service
    return service


@app.route("/", methods=["GET", "POST"])
def upload():
    selected_activity = request.form.get("activity", DEFAULT_ACTIVITY_KEY)

    if request.method == "POST":
        file = request.files.get("video")
        if not file or file.filename == "":
            flash("请选择一个视频文件。")
            return redirect(request.url)

        if selected_activity not in ACTIVITY_SPECS:
            flash("请选择有效的动作类型。")
            return redirect(request.url)

        session_id = uuid.uuid4().hex
        upload_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(upload_path)

        try:
            service = _get_inference_service(selected_activity)
            result = service.compare(
                upload_path, output_dir=RESULT_DIR / session_id, target_fps=TARGET_FPS
            )
        except Exception as exc:  # pragma: no cover - runtime errors
            flash(f"分析失败: {exc}")
            return redirect(request.url)

        session_dir = RESULT_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        meta_path = session_dir / "meta.json"
        meta_path.write_text(
            json.dumps({"activity": selected_activity}, ensure_ascii=False),
            encoding="utf-8",
        )

        if result.focus_metrics:
            metrics_path = session_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(result.focus_metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if result.annotated_video_path:
            target = session_dir / result.annotated_video_path.name
            if result.annotated_video_path != target:
                try:
                    target.write_bytes(result.annotated_video_path.read_bytes())
                except Exception:
                    pass

        return redirect(url_for("result", session_id=session_id))

    return render_template(
        "upload.html",
        activity_options=ACTIVITY_OPTIONS,
        selected_activity=selected_activity,
    )


@app.route("/results/<session_id>")
def result(session_id: str):
    session_dir = RESULT_DIR / session_id
    if not session_dir.exists():
        flash("未找到分析结果，请重新上传视频。")
        return redirect(url_for("upload"))

    angle_plot = next(session_dir.glob("*_angles.png"), None)
    keynode_plot = next(session_dir.glob("*_keynode_similarity.png"), None)
    focus_similarity_plot = next(session_dir.glob("*_focus_similarity.png"), None)
    advice_path = session_dir / "advice.txt"
    recommendations: list[str] = []

    if advice_path.exists():
        recommendations = advice_path.read_text(encoding="utf-8").splitlines()

    annotated_video = next(session_dir.glob("*_annotated.mp4"), None)
    metrics_path = session_dir / "metrics.json"
    focus_metrics: dict[str, float] | None = None
    if metrics_path.exists():
        focus_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    activity_label = None
    meta_path = session_dir / "meta.json"
    show_keynode_plot = False
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        act_key = meta.get("activity")
        if act_key and act_key in ACTIVITY_SPECS:
            activity_label = ACTIVITY_SPECS[act_key].label
            show_keynode_plot = act_key == "shooting"

    return render_template(
        "result.html",
        angle_plot_filename=angle_plot.name if angle_plot else None,
        keynode_plot_filename=keynode_plot.name if keynode_plot else None,
        focus_similarity_plot_filename=focus_similarity_plot.name if focus_similarity_plot else None,
        annotated_video_filename=annotated_video.name if annotated_video else None,
        focus_metrics=focus_metrics,
        recommendations=recommendations,
        session_id=session_id,
        activity_label=activity_label,
        show_keynode_plot=show_keynode_plot,
    )


@app.route("/media/<session_id>/<path:filename>")
def media(session_id: str, filename: str):
    file_path = RESULT_DIR / session_id / filename
    if not file_path.exists():
        flash("文件不存在。")
        return redirect(url_for("result", session_id=session_id))
    return send_file(file_path)


@app.route("/download/<session_id>/<filename>")
def download(session_id: str, filename: str):
    file_path = RESULT_DIR / session_id / filename
    if not file_path.exists():
        flash("文件不存在。")
        return redirect(url_for("result", session_id=session_id))
    return send_file(file_path, as_attachment=True)


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

