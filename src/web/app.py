from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, send_file, url_for, Response

from src.config import ActivitySpec, PROJECT_ROOT, get_activity_specs
from src.models.inference import PoseInferenceService, build_reference_features_from_video
from src.utils.serialization import load_features_from_file, save_features_to_file
from src.web.keyword_recommender import get_recommended_keywords
from src.web.search_service import get_search_service


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


@app.errorhandler(500)
def internal_error(error):
    """处理服务器内部错误"""
    import traceback
    error_details = traceback.format_exc()
    print(f"[服务器错误] {error}")
    print(f"[服务器错误] 详情: {error_details}")
    flash(f"服务器错误: {error}")
    return redirect(url_for("upload"))


@app.errorhandler(413)
def request_entity_too_large(error):
    """处理文件过大错误"""
    print(f"[文件过大] {error}")
    flash("上传的文件过大，请选择较小的视频文件（最大512MB）")
    return redirect(url_for("upload"))


@app.errorhandler(404)
def not_found(error):
    """处理404错误"""
    return redirect(url_for("upload"))


def _ensure_model_exists(model_path: Path) -> None:
    if not model_path.exists():
        # 尝试查找正确的路径
        expected_path = PROJECT_ROOT / "runs" / "train" / "kobe_pose2" / "weights" / "best.pt"
        alternative_path = PROJECT_ROOT / "runs" / "train" / "kobe_pose" / "weights" / "best.pt"
        
        error_msg = f"未找到模型文件: {model_path}\n"
        error_msg += f"尝试的路径（绝对）: {model_path.resolve()}\n"
        
        if expected_path.exists():
            error_msg += f"\n找到模型文件在: {expected_path.resolve()}\n"
            error_msg += "请检查环境变量 MODEL_PATH 或更新配置。"
        elif alternative_path.exists():
            error_msg += f"\n找到模型文件在: {alternative_path.resolve()}\n"
            error_msg += "请检查环境变量 MODEL_PATH 或更新配置。"
        else:
            error_msg += "\n请先完成对应动作的训练，或在 .env 中更新模型路径。"
        
        raise FileNotFoundError(error_msg)


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

    error_msg = f"活动「{spec.label}」缺少参考特征。\n"
    if spec.reference_features_path:
        error_msg += f"尝试的参考特征文件路径: {spec.reference_features_path.resolve()}\n"
        # Check for common fallback names
        fallback_path = PROJECT_ROOT / "artifacts" / "reference_features.json"
        if fallback_path.exists() and spec.key == "shooting":
            error_msg += f"找到备用文件: {fallback_path.resolve()}\n"
            error_msg += "请将文件重命名为 reference_features_shooting.json 或设置 REFERENCE_FEATURES_PATH 环境变量。\n"
    if spec.reference_video_path:
        error_msg += f"尝试的参考视频路径: {spec.reference_video_path.resolve()}\n"
    error_msg += f"\n请设置 {spec.key.upper()}_REFERENCE_FEATURES_PATH 或 {spec.key.upper()}_REFERENCE_VIDEO_PATH 环境变量。"
    raise RuntimeError(error_msg)


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
            print(f"[视频分析] 开始分析视频: {upload_path}")
            print(f"[视频分析] 活动类型: {selected_activity}")
            print(f"[视频分析] 输出目录: {RESULT_DIR / session_id}")
            
            service = _get_inference_service(selected_activity)
            print(f"[视频分析] 推理服务已获取，开始处理...")
            
            result = service.compare(
                upload_path, output_dir=RESULT_DIR / session_id, target_fps=TARGET_FPS
            )
            
            print(f"[视频分析] 处理完成")
        except Exception as exc:  # pragma: no cover - runtime errors
            import traceback
            error_details = traceback.format_exc()
            print(f"[视频分析] 分析失败: {exc}")
            print(f"[视频分析] 错误详情: {error_details}")
            
            # 清理上传的文件
            try:
                if upload_path.exists():
                    upload_path.unlink()
            except Exception:
                pass
            
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
        
        # 保存相似度分数和偏差角度，用于详细建议
        if result.similarity_scores:
            similarity_path = session_dir / "similarity_scores.json"
            similarity_path.write_text(
                json.dumps(result.similarity_scores, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        
        if result.deviation_angles:
            deviation_path = session_dir / "deviation_angles.json"
            deviation_path.write_text(
                json.dumps(result.deviation_angles, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        
        if result.focus_similarity:
            focus_similarity_path = session_dir / "focus_similarity.json"
            focus_similarity_path.write_text(
                json.dumps(result.focus_similarity, indent=2, ensure_ascii=False),
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
    activity_key = None
    meta_path = session_dir / "meta.json"
    show_keynode_plot = False
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        activity_key = meta.get("activity")
        if activity_key and activity_key in ACTIVITY_SPECS:
            activity_label = ACTIVITY_SPECS[activity_key].label
            show_keynode_plot = activity_key == "shooting"

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
        activity_key=activity_key,
        show_keynode_plot=show_keynode_plot,
    )


@app.route("/media/<session_id>/<path:filename>")
def media(session_id: str, filename: str):
    file_path = RESULT_DIR / session_id / filename
    if not file_path.exists():
        flash("文件不存在。")
        return redirect(url_for("result", session_id=session_id))
    
    # 对于视频文件，支持Range请求以实现流式播放
    if filename.endswith(('.mp4', '.webm', '.ogg', '.mov')):
        return _send_video_file(file_path, filename)
    
    return send_file(file_path)


def _send_video_file(file_path: Path, filename: str) -> Response:
    """发送视频文件，支持HTTP Range请求以实现流式播放"""
    file_size = file_path.stat().st_size
    range_header = request.headers.get('Range', None)
    
    if range_header:
        # 解析Range头
        byte_start = 0
        byte_end = file_size - 1
        
        range_match = range_header.replace('bytes=', '').split('-')
        if range_match[0]:
            byte_start = int(range_match[0])
        if len(range_match) > 1 and range_match[1]:
            byte_end = int(range_match[1])
        
        length = byte_end - byte_start + 1
        
        with open(file_path, 'rb') as f:
            f.seek(byte_start)
            data = f.read(length)
        
        response = Response(
            data,
            206,  # Partial Content
            {
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(length),
                'Content-Type': 'video/mp4',
            },
            direct_passthrough=True,
        )
        return response
    else:
        # 没有Range请求，返回整个文件（使用流式传输）
        def generate():
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    yield chunk
        
        response = Response(
            generate(),
            mimetype='video/mp4',
            headers={
                'Accept-Ranges': 'bytes',
                'Content-Length': str(file_size),
            }
        )
        return response


@app.route("/download/<session_id>/<filename>")
def download(session_id: str, filename: str):
    file_path = RESULT_DIR / session_id / filename
    if not file_path.exists():
        flash("文件不存在。")
        return redirect(url_for("result", session_id=session_id))
    return send_file(file_path, as_attachment=True)


@app.route("/search/<path:keyword>")
def search_keyword(keyword: str):
    """搜索关键词接口"""
    try:
        from urllib.parse import unquote
        keyword = unquote(keyword)  # URL解码
        
        print(f"[搜索路由] 收到搜索请求: {keyword}")
        search_service = get_search_service()
        result = search_service.search(keyword)
        print(f"[搜索路由] 搜索结果类型: {result.get('type', 'mixed')}")
        print(f"[搜索路由] 结果键: {list(result.keys())}")
        
        # 如果是重定向类型，直接跳转
        if result.get("type") == "redirect":
            return redirect(result["url"])
        
        # 如果是LLM回答类型，显示结果页面
        if result.get("type") == "llm_answer":
            return render_template(
                "search_result.html",
                keyword=keyword,
                llm_answer=result.get("answer"),
                search_url=None,
                results=None,
            )
        
        # 如果是混合类型，显示综合结果
        if "llm_answer" in result or "search_url" in result or "llm_error" in result:
            return render_template(
                "search_result.html",
                keyword=keyword,
                llm_answer=result.get("llm_answer"),
                llm_error=result.get("llm_error") or result.get("message"),
                technical_error=result.get("technical_error"),
                search_url=result.get("search_url"),
                results=result.get("results"),
            )
        
        # 如果是结果类型，显示搜索结果页面
        if result.get("type") == "results":
            return render_template(
                "search_result.html",
                keyword=keyword,
                llm_answer=None,
                search_url=None,
                results=result.get("results", []),
            )
        
        # 默认：跳转到百度搜索
        search_url = f"https://www.baidu.com/s?wd={keyword}"
        return redirect(search_url)
    except Exception as e:
        import traceback
        print(f"搜索失败: {e}")
        print(traceback.format_exc())
        # 出错时跳转到百度搜索
        search_url = f"https://www.baidu.com/s?wd={keyword}"
        return redirect(search_url)


@app.route("/results/<session_id>/detailed_advice")
def detailed_advice(session_id: str):
    """显示详细的动作建议页面"""
    session_dir = RESULT_DIR / session_id
    if not session_dir.exists():
        flash("未找到分析结果，请重新上传视频。")
        return redirect(url_for("upload"))
    
    # 读取活动类型
    meta_path = session_dir / "meta.json"
    activity_key = None
    activity_label = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        activity_key = meta.get("activity")
        if activity_key and activity_key in ACTIVITY_SPECS:
            activity_label = ACTIVITY_SPECS[activity_key].label
    
    # 只对投篮和跑步动作生成详细建议
    if activity_key not in ["shooting", "running"]:
        flash("详细建议功能目前仅支持投篮和跑步动作分析。")
        return redirect(url_for("result", session_id=session_id))
    
    # 读取数据
    metrics_path = session_dir / "metrics.json"
    similarity_path = session_dir / "similarity_scores.json"
    deviation_path = session_dir / "deviation_angles.json"
    focus_similarity_path = session_dir / "focus_similarity.json"
    
    focus_metrics = {}
    similarity_scores = {}
    deviation_angles = {}
    focus_similarity = {}
    
    if metrics_path.exists():
        focus_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if similarity_path.exists():
        similarity_scores = json.loads(similarity_path.read_text(encoding="utf-8"))
    if deviation_path.exists():
        deviation_angles = json.loads(deviation_path.read_text(encoding="utf-8"))
    if focus_similarity_path.exists():
        focus_similarity = json.loads(focus_similarity_path.read_text(encoding="utf-8"))
    
    # 生成详细建议
    detailed_advice_data = _generate_detailed_advice(
        activity_key, focus_metrics, similarity_scores, deviation_angles, focus_similarity
    )
    
    # 获取推荐关键词
    all_advice_texts = (
        detailed_advice_data.get("overall_evaluation", [])
        + detailed_advice_data.get("technical_analysis", [])
        + detailed_advice_data.get("improvement_suggestions", [])
    )
    recommended_keywords = []
    if all_advice_texts:
        try:
            recommended_keywords = get_recommended_keywords(activity_key, all_advice_texts, top_k=5)
            print(f"推荐关键词数量: {len(recommended_keywords)}")
            if recommended_keywords:
                print(f"推荐关键词: {recommended_keywords}")
            else:
                print("警告: 未获取到推荐关键词，可能是模型未训练或加载失败")
        except Exception as e:
            import traceback
            print(f"获取推荐关键词失败: {e}")
            print(f"错误详情: {traceback.format_exc()}")
    
    return render_template(
        "detailed_advice.html",
        session_id=session_id,
        activity_label=activity_label,
        recommended_keywords=recommended_keywords,
        **detailed_advice_data,
    )


def _generate_detailed_advice(
    activity_key: str,
    focus_metrics: dict,
    similarity_scores: dict,
    deviation_angles: dict,
    focus_similarity: dict,
) -> dict:
    """根据分析数据生成详细建议"""
    import random
    from pathlib import Path
    
    # 根据活动类型选择模板
    if activity_key == "running":
        template_path = PROJECT_ROOT / "artifacts" / "advice_templates" / "running_advice_template.json"
        template_key = "running_advice"
    else:
        template_path = PROJECT_ROOT / "artifacts" / "advice_templates" / "basketball_advice_template.json"
        template_key = "basketball_advice"
    
    if not template_path.exists():
        return {
            "overall_evaluation": [],
            "technical_analysis": [],
            "improvement_suggestions": [],
        }
    
    template_data = json.loads(template_path.read_text(encoding="utf-8"))
    template = template_data[template_key]
    
    # 计算综合评分
    avg_similarity = 0.0
    if similarity_scores:
        valid_scores = [v for v in similarity_scores.values() if not (v != v) and not math.isnan(v)]
        if valid_scores:
            avg_similarity = sum(valid_scores) / len(valid_scores)
    
    # 安全计算评分，处理NaN
    if math.isnan(avg_similarity) or avg_similarity < 0:
        score = 60  # 默认评分
    else:
        score = int(avg_similarity * 100)
    
    if activity_key == "running":
        # 跑步动作的变量生成
        stability = "稳定" if score >= 80 else "略不稳定" if score >= 60 else "明显漂移"
        
        def safe_float(value: Any, default: float = 0.0) -> float:
            """安全地将值转换为浮点数，处理NaN和None"""
            if value is None:
                return default
            try:
                val = float(value)
                if math.isnan(val):
                    return default
                return val
            except (ValueError, TypeError):
                return default
        
        step_frequency = safe_float(focus_metrics.get("步频（次/秒）"), 0)
        step_frequency_per_min = int(step_frequency * 60) if step_frequency > 0 else 0
        
        # 估算前倾角度（基于髋部角度）
        hip_angle = safe_float(focus_metrics.get("髋部角度"), 90)
        lean_angle = max(0, min(30, abs(hip_angle - 90)))
        
        # 估算摆臂幅度
        arm_swing = safe_float(focus_metrics.get("摆臂幅度"), 45)
        
        variables = {
            "动作流畅度评分": str(score),
            "步频": str(step_frequency_per_min),
            "稳定性评价": stability,
            "总体建议内容": "加强核心力量训练，调整摆臂方式" if score < 70 else "保持当前动作，继续优化细节",
            "检测时刻": random.choice(["起跑阶段", "中程阶段", "冲刺阶段"]),
            "前倾角度": f"{int(lean_angle) if not math.isnan(lean_angle) else 15}",
            "评价前倾角度": "适中" if 10 <= lean_angle <= 20 else "偏大" if lean_angle > 20 else "偏小",
            "摆臂幅度": f"{int(arm_swing) if not math.isnan(arm_swing) else 45}",
            "评价摆臂": "正常" if 40 <= arm_swing <= 60 else "过小" if arm_swing < 40 else "过大",
            "建议范围": "10-20",
            "落脚距离": f"{random.randint(20, 40)}",
            "评价落脚": random.choice(["偏前", "偏后", "合理"]),
            "建议距离范围": "25-35",
            "训练方法": random.choice(["核心力量训练", "间歇跑", "高抬腿训练"]),
            "待改进部分": random.choice(["摆臂幅度", "步频", "身体前倾"]),
            "目标效果": random.choice(["提升速度", "改善耐力", "降低膝盖负担"]),
            "热身方法": random.choice(["动态拉伸", "慢跑热身", "腿部摇摆"]),
        }
    else:
        # 投篮动作的变量生成
        stability = "稳定" if score >= 80 else "略不稳定" if score >= 60 else "明显失衡"
        
        # 从focus_metrics获取角度数据（使用中文键名）
        def safe_float(value: Any, default: float = 0.0) -> float:
            """安全地将值转换为浮点数，处理NaN和None"""
            if value is None:
                return default
            try:
                val = float(value)
                if math.isnan(val):
                    return default
                return val
            except (ValueError, TypeError):
                return default
        
        elbow_left_angle = safe_float(focus_metrics.get("左肘最大弯曲角度") or focus_metrics.get("elbow_left"), 150)
        elbow_right_angle = safe_float(focus_metrics.get("右肘最大弯曲角度") or focus_metrics.get("elbow_right"), 150)
        wrist_left_head = safe_float(focus_metrics.get("左手腕朝头部最大弯曲角度") or focus_metrics.get("wrist_left_head"), 65)
        wrist_right_head = safe_float(focus_metrics.get("右手腕朝头部最大弯曲角度") or focus_metrics.get("wrist_right_head"), 65)
        wrist_release = safe_float(focus_metrics.get("右手腕出手前倾角度"), 0)
        knee_left_angle = safe_float(focus_metrics.get("左膝最大弯曲角度") or focus_metrics.get("knee_left"), 120)
        knee_right_angle = safe_float(focus_metrics.get("右膝最大弯曲角度") or focus_metrics.get("knee_right"), 120)
        
        # 计算肩肘角度（平均，转换为肩肘夹角，通常肘弯曲角度越大，肩肘夹角越小）
        # 肘弯曲角度通常在120-180度，肩肘夹角通常在80-100度
        elbow_avg = (elbow_left_angle + elbow_right_angle) / 2
        shoulder_elbow_angle = max(75, min(95, 180 - elbow_avg + 20))  # 转换为肩肘夹角
        
        # 计算手腕角度（使用出手时的手腕角度，如果没有则使用平均值）
        if wrist_release > 0:
            wrist_angle = wrist_release
        else:
            wrist_angle = (wrist_left_head + wrist_right_head) / 2
        
        # 计算膝屈伸幅度（膝关节从伸直180度到最大弯曲的角度差）
        knee_avg = (knee_left_angle + knee_right_angle) / 2
        knee_flexion = 180 - knee_avg  # 屈伸幅度
        
        # 估算出手时机（基于相似度）
        release_timing_score = avg_similarity
        if release_timing_score >= 0.8:
            timing_eval = "时机适中"
            timing_advice = "保持当前出手节奏"
        elif release_timing_score >= 0.6:
            timing_eval = "时机略早"
            timing_advice = "适当延迟出手时机，在跳跃最高点出手"
        else:
            timing_eval = "过早出手"
            timing_advice = "调整起跳与出手的配合，在身体上升至最高点时出手"
        
        # 估算随手持续时间（基于动作流畅度）
        follow_through_duration = max(0.3, min(1.5, score / 100 * 1.2))
        
        variables = {
            "比赛场景": random.choice(["定点投篮训练", "绕桩后急停投篮", "比赛实战投篮"]),
            "动作流畅度评分": str(score),
            "稳定性评价": stability,
            "总体建议内容": "加强核心力量训练，调整出手角度" if score < 70 else "保持当前动作，继续优化细节",
            "肩肘角度": f"{int(shoulder_elbow_angle) if not math.isnan(shoulder_elbow_angle) else 90}",
            "角度评价": "适中" if 85 <= shoulder_elbow_angle <= 95 else "过大" if shoulder_elbow_angle > 95 else "过小",
            "建议肩肘角范围": "85~90",
            "预估提升百分比": f"{int((1 - avg_similarity) * 15) if not math.isnan(avg_similarity) else 10}",
            "手腕角度": f"{int(wrist_angle) if not math.isnan(wrist_angle) else 65}",
            "手腕角评价": "合适" if 55 <= wrist_angle <= 75 else "未充分后仰" if wrist_angle < 55 else "过大",
            "建议手腕角范围": "60~70",
            "膝屈伸幅度": f"{int(knee_flexion) if not math.isnan(knee_flexion) else 30}",
            "膝屈伸评价": "适中" if 25 <= knee_flexion <= 45 else "偏小" if knee_flexion < 25 else "偏大",
            "建议膝屈伸范围": "30~40",
            "出手时机评价": timing_eval,
            "时机建议": timing_advice,
            "随手持续时间": f"{follow_through_duration:.1f}",
            "随手评价": "良好" if follow_through_duration >= 0.8 else "太短" if follow_through_duration < 0.5 else "太长影响平衡",
            "建议随手时间": "0.8~1.2",
            "训练方法": random.choice(["中距离投篮训练", "三分稳定性训练", "体能投篮结合练习"]),
            "待改进部分": random.choice(["出手高度", "手腕发力", "膝关节配合"]),
            "目标效果": random.choice(["提升命中率", "减少投篮偏差", "增加出手速度"]),
            "专项训练": random.choice(["急停跳投训练", "绕桩后投篮训练"]),
            "投篮数量": str(random.choice([50, 80, 100, 120])),
        }
    
    # 填充模板
    def fill_template(text: str) -> str:
        for key, value in variables.items():
            text = text.replace(f"${{{key}}}", str(value))
        return text
    
    overall_evaluation = [fill_template(t) for t in template["overall_evaluation"]]
    technical_analysis = [fill_template(t) for t in template["technical_analysis"]]
    improvement_suggestions = [fill_template(t) for t in template["improvement_suggestions"]]
    
    return {
        "overall_evaluation": overall_evaluation,
        "technical_analysis": technical_analysis,
        "improvement_suggestions": improvement_suggestions,
    }


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

