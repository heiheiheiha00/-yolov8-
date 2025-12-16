"""关键词推荐模块"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "keyword_models"


class KeywordRecommender:
    """关键词推荐器"""
    
    def __init__(self, activity_type: str) -> None:
        self.activity_type = activity_type
        self.model = None
        self.vectorizer = None
        self.keyword_mapping = None
        self._load_model()
    
    def _get_model_name(self) -> str:
        """将活动类型转换为模型文件名"""
        # 映射活动类型到模型文件名
        mapping = {
            "shooting": "basketball",  # 投篮动作使用basketball模型
            "running": "running",      # 跑步动作使用running模型
        }
        return mapping.get(self.activity_type, self.activity_type)
    
    def _load_model(self) -> None:
        """加载模型和向量化器"""
        model_name = self._get_model_name()
        model_path = MODEL_DIR / f"{model_name}_keyword_model.json"
        vectorizer_path = MODEL_DIR / f"{model_name}_vectorizer.pkl"
        mapping_path = MODEL_DIR / f"{model_name}_keyword_mapping.json"
        
        print(f"[关键词推荐] 活动类型: {self.activity_type} -> 模型名称: {model_name}")
        print(f"[关键词推荐] 尝试加载模型: {model_path}")
        print(f"[关键词推荐] 模型文件存在: {model_path.exists()}")
        print(f"[关键词推荐] 向量化器文件存在: {vectorizer_path.exists()}")
        print(f"[关键词推荐] 映射文件存在: {mapping_path.exists()}")
        
        if not model_path.exists() or not vectorizer_path.exists():
            print(f"[关键词推荐] 模型文件不存在，无法加载")
            return
        
        try:
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.keyword_mapping = json.load(f)
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.model = None
            self.vectorizer = None
            self.keyword_mapping = None
    
    def recommend_keywords(self, advice_texts: list[str], top_k: int = 5) -> list[str]:
        """根据建议文本推荐关键词"""
        if not self.model or not self.vectorizer or not self.keyword_mapping:
            print(f"模型未加载: model={self.model is not None}, vectorizer={self.vectorizer is not None}, mapping={self.keyword_mapping is not None}")
            # 如果模型未加载，返回一些默认关键词
            return self._get_fallback_keywords(top_k)
        
        # 合并所有建议文本
        combined_text = " ".join(advice_texts)
        if not combined_text.strip():
            print("建议文本为空")
            return self._get_fallback_keywords(top_k)
        
        # 向量化
        try:
            X = self.vectorizer.transform([combined_text])
            dmatrix = xgb.DMatrix(X)
            
            # 预测
            predictions = self.model.predict(dmatrix)[0]
            
            # 获取top-k问题类别（选择概率最高的）
            top_issue_indices = np.argsort(predictions)[-top_k:][::-1]
            
            print(f"预测结果: top_issue_indices={top_issue_indices}, predictions={predictions[top_issue_indices]}")
            
            # 收集关键词
            recommended_keywords = []
            seen_keywords = set()
            
            for issue_idx in top_issue_indices:
                # 尝试字符串键和整数键
                issue_idx_str = str(issue_idx)
                issue_data = None
                
                if issue_idx_str in self.keyword_mapping:
                    issue_data = self.keyword_mapping[issue_idx_str]
                elif issue_idx in self.keyword_mapping:
                    issue_data = self.keyword_mapping[issue_idx]
                
                if issue_data:
                    keywords = issue_data.get("keywords", [])
                    print(f"问题索引 {issue_idx} 找到 {len(keywords)} 个关键词")
                    for keyword in keywords:
                        if keyword not in seen_keywords:
                            recommended_keywords.append(keyword)
                            seen_keywords.add(keyword)
                            if len(recommended_keywords) >= top_k:
                                break
                
                if len(recommended_keywords) >= top_k:
                    break
            
            print(f"最终推荐关键词: {recommended_keywords}")
            return recommended_keywords[:top_k] if recommended_keywords else self._get_fallback_keywords(top_k)
        except Exception as e:
            import traceback
            print(f"推荐关键词失败: {e}")
            print(f"错误详情: {traceback.format_exc()}")
            return self._get_fallback_keywords(top_k)
    
    def _get_fallback_keywords(self, top_k: int) -> list[str]:
        """获取备用关键词（当模型不可用时）"""
        if not self.keyword_mapping:
            return []
        
        # 从所有问题中随机选择一些关键词
        all_keywords = []
        for issue_data in self.keyword_mapping.values():
            if isinstance(issue_data, dict):
                keywords = issue_data.get("keywords", [])
                all_keywords.extend(keywords)
        
        # 去重并返回前top_k个
        unique_keywords = list(dict.fromkeys(all_keywords))  # 保持顺序的去重
        return unique_keywords[:top_k]


def get_recommended_keywords(activity_type: str, advice_texts: list[str], top_k: int = 5) -> list[str]:
    """获取推荐关键词的便捷函数"""
    recommender = KeywordRecommender(activity_type)
    return recommender.recommend_keywords(advice_texts, top_k)

