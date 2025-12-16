#!/usr/bin/env python3
"""
训练关键词推荐模型
将关键词库处理为XGBoost训练集，训练一个用于匹配关键词相似度的模型
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 获取项目根目录（脚本在scripts/目录下，需要向上两级到项目根目录）
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
KEYWORD_LIB_PATH = PROJECT_ROOT / "artifacts" / "keyword_library" / "search_keywords.json"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "keyword_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_keyword_library() -> dict[str, Any]:
    """加载关键词库"""
    with open(KEYWORD_LIB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_issues_from_advice(advice_text: str, activity_type: str) -> list[str]:
    """从建议文本中提取可能的问题关键词"""
    # 定义问题模式
    issue_patterns = {
        "running": [
            r"前倾.*?(不足|过大|偏[小大])",
            r"摆臂.*?(过小|过大|不足|过度)",
            r"步频.*?(过低|过高|不足|过快)",
            r"落脚.*?(偏前|偏后)",
            r"核心.*?(不足|缺乏)",
            r"呼吸.*?(不协调|节奏)",
            r"髋.*?(不足|灵活性)",
            r"耐力.*?(不足|缺乏)",
            r"速度.*?(不足|缺乏)",
            r"膝盖.*?(疼痛|损伤)",
            r"肌肉.*?(酸痛|疲劳)",
        ],
        "basketball": [
            r"肩肘.*?(过大|过小|不足)",
            r"手腕.*?(不足|过度|发力)",
            r"膝关节.*?(不足|过度|屈伸)",
            r"出手.*?(过早|过晚|时机)",
            r"随手.*?(太短|过长|不足)",
            r"弧度.*?(过低|过高)",
            r"稳定性.*?(不足|缺乏)",
            r"专注.*?(不足|缺乏)",
            r"力量.*?(不均|分配)",
        ],
    }
    
    patterns = issue_patterns.get(activity_type, [])
    issues = []
    for pattern in patterns:
        matches = re.findall(pattern, advice_text)
        issues.extend(matches)
    
    return issues


def create_training_data(keyword_lib: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """创建训练数据"""
    X_texts = []  # 建议文本
    y_labels = []  # 对应的关键词索引
    
    for activity_type, issues_list in keyword_lib["search_keywords"].items():
        for issue_idx, issue_data in enumerate(issues_list):
            issue = issue_data["issue"]
            keywords = issue_data["keywords"]
            
            # 为每个关键词创建训练样本
            for keyword in keywords:
                # 组合问题和关键词作为输入文本
                combined_text = f"{issue} {keyword}"
                X_texts.append(combined_text)
                y_labels.append(issue_idx)  # 使用问题索引作为标签
    
    return np.array(X_texts), np.array(y_labels)


def train_xgboost_model(X_texts: np.ndarray, y_labels: np.ndarray) -> tuple[Any, Any]:
    """训练XGBoost模型"""
    # 使用TF-IDF向量化文本
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X_vectorized = vectorizer.fit_transform(X_texts)
    
    # 转换为DMatrix格式
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y_labels, test_size=0.2, random_state=42
    )
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 训练参数
    params = {
        "objective": "multi:softprob",
        "num_class": len(np.unique(y_labels)),
        "max_depth": 6,
        "eta": 0.3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "random_state": 42,
    }
    
    # 训练模型
    evals = [(dtrain, "train"), (dtest, "test")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    
    return model, vectorizer


def save_model(model: Any, vectorizer: Any, activity_type: str) -> None:
    """保存模型和向量化器"""
    model_path = MODEL_DIR / f"{activity_type}_keyword_model.json"
    vectorizer_path = MODEL_DIR / f"{activity_type}_vectorizer.pkl"
    
    model.save_model(str(model_path))
    
    with open(vectorizer_path, "wb") as f:  # type: ignore[assignment]
        pickle.dump(vectorizer, f)
    
    print(f"模型已保存: {model_path}")
    print(f"向量化器已保存: {vectorizer_path}")


def main() -> None:
    """主函数"""
    print("加载关键词库...")
    keyword_lib = load_keyword_library()
    
    for activity_type in ["running", "basketball"]:
        print(f"\n处理 {activity_type} 数据...")
        
        # 创建该活动类型的训练数据
        activity_data = keyword_lib["search_keywords"][activity_type]
        X_texts = []
        y_labels = []
        
        for issue_idx, issue_data in enumerate(activity_data):
            issue = issue_data["issue"]
            keywords = issue_data["keywords"]
            
            for keyword in keywords:
                combined_text = f"{issue} {keyword}"
                X_texts.append(combined_text)
                y_labels.append(issue_idx)
        
        if not X_texts:
            print(f"警告: {activity_type} 没有数据，跳过")
            continue
        
        X_texts = np.array(X_texts)
        y_labels = np.array(y_labels)
        
        print(f"训练样本数: {len(X_texts)}")
        print(f"问题类别数: {len(np.unique(y_labels))}")
        
        # 训练模型
        print("训练XGBoost模型...")
        model, vectorizer = train_xgboost_model(X_texts, y_labels)
        
        # 保存模型
        save_model(model, vectorizer, activity_type)
        
        # 保存关键词映射
        keyword_mapping = {
            issue_idx: {
                "issue": issue_data["issue"],
                "keywords": issue_data["keywords"],
            }
            for issue_idx, issue_data in enumerate(activity_data)
        }
        mapping_path = MODEL_DIR / f"{activity_type}_keyword_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(keyword_mapping, f, ensure_ascii=False, indent=2)
        print(f"关键词映射已保存: {mapping_path}")
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()

