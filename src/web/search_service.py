"""搜索服务模块 - 支持搜索引擎和大模型API"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import requests


class SearchService:
    """搜索服务基类"""
    
    def search(self, keyword: str) -> dict[str, Any]:
        """执行搜索，返回搜索结果"""
        raise NotImplementedError


class BaiduSearchService(SearchService):
    """百度搜索服务（使用百度搜索API）"""
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("BAIDU_SEARCH_API_KEY")
        self.secret_key = secret_key or os.getenv("BAIDU_SEARCH_SECRET_KEY")
        self.base_url = "https://www.baidu.com/s"
    
    def search(self, keyword: str) -> dict[str, Any]:
        """使用百度搜索"""
        # 直接返回百度搜索URL
        search_url = f"{self.base_url}?wd={keyword}"
        return {
            "type": "redirect",
            "url": search_url,
            "keyword": keyword,
        }


class GoogleSearchService(SearchService):
    """Google搜索服务（使用Google Custom Search API）"""
    
    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, keyword: str) -> dict[str, Any]:
        """使用Google Custom Search API"""
        if not self.api_key or not self.search_engine_id:
            # 如果没有API密钥，返回Google搜索URL
            search_url = f"https://www.google.com/search?q={keyword}"
            return {
                "type": "redirect",
                "url": search_url,
                "keyword": keyword,
            }
        
        try:
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": keyword,
            }
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # 提取搜索结果
            items = data.get("items", [])[:5]  # 取前5个结果
            results = [
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
                for item in items
            ]
            
            return {
                "type": "results",
                "keyword": keyword,
                "results": results,
                "total": data.get("searchInformation", {}).get("totalResults", "0"),
            }
        except Exception as e:
            # 如果API调用失败，返回Google搜索URL
            search_url = f"https://www.google.com/search?q={keyword}"
            return {
                "type": "redirect",
                "url": search_url,
                "keyword": keyword,
                "error": str(e),
            }


class LLMSearchService(SearchService):
    """大模型搜索服务（使用OpenAI、DeepSeek或其他LLM API）"""
    
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, model: Optional[str] = None) -> None:
        # 优先使用DeepSeek配置
        deepseek_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if deepseek_key:
            # 使用DeepSeek API
            self.api_key = deepseek_key
            self.api_base = api_base or os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
            self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        else:
            # 如果没有DeepSeek密钥，才使用OpenAI或其他
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            self.api_base = api_base or os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
            self.model = model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    def search(self, keyword: str) -> dict[str, Any]:
        """使用大模型生成搜索建议或回答"""
        print(f"[DeepSeek] 开始处理关键词: {keyword}")
        print(f"[DeepSeek] API密钥: {self.api_key[:10] if self.api_key else 'None'}...")
        print(f"[DeepSeek] API端点: {self.api_base}")
        print(f"[DeepSeek] 模型: {self.model}")
        
        if not self.api_key:
            error_msg = "未配置大模型API密钥（请设置DEEPSEEK_API_KEY或OPENAI_API_KEY）"
            print(f"[DeepSeek] 错误: {error_msg}")
            return {
                "type": "error",
                "message": error_msg,
            }
        
        try:
            try:
                from openai import OpenAI
            except ImportError:
                error_msg = "未安装openai库。注意：DeepSeek使用OpenAI兼容的API格式，所以需要安装openai库（虽然调用的是DeepSeek API）。请运行: pip install openai"
                print(f"[DeepSeek] 错误: {error_msg}")
                return {
                    "type": "error",
                    "message": error_msg,
                }
            
            print(f"[DeepSeek] 正在调用DeepSeek API（使用OpenAI兼容格式）...")
            client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            
            prompt = f"""你是一个运动训练专家。用户搜索了关键词："{keyword}"。

请提供：
1. 这个训练方法的简要说明（2-3句话）
2. 3-5个相关的训练要点或注意事项
3. 建议的训练频率和强度

请用中文回答，格式清晰。"""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的运动训练指导专家，擅长篮球和跑步训练。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            
            answer = response.choices[0].message.content
            print(f"[DeepSeek] API调用成功，回答长度: {len(answer)} 字符")
            
            return {
                "type": "llm_answer",
                "keyword": keyword,
                "answer": answer,
            }
        except ImportError:
            error_msg = "未安装openai库（DeepSeek使用OpenAI兼容的API格式，需要openai库）。请运行: pip install openai"
            print(f"[DeepSeek] 错误: {error_msg}")
            return {
                "type": "error",
                "message": error_msg,
            }
        except Exception as e:
            import traceback
            error_str = str(e)
            
            # 检查是否是余额不足错误
            if "402" in error_str or "Insufficient Balance" in error_str or "余额" in error_str:
                error_msg = "DeepSeek API 账户余额不足。请前往 DeepSeek 官网充值后重试。"
                user_friendly_msg = "💳 DeepSeek API 账户余额不足\n\n请前往 https://www.deepseek.com/ 充值账户余额后重试。"
            elif "401" in error_str or "Unauthorized" in error_str or "Invalid API key" in error_str:
                error_msg = "DeepSeek API 密钥无效或已过期。请检查 .env 文件中的 DEEPSEEK_API_KEY 是否正确。"
                user_friendly_msg = "🔑 DeepSeek API 密钥无效\n\n请检查 .env 文件中的 DEEPSEEK_API_KEY 是否正确，或前往 DeepSeek 官网重新获取 API 密钥。"
            elif "429" in error_str or "Rate limit" in error_str or "限流" in error_str:
                error_msg = "DeepSeek API 请求频率过高，请稍后再试。"
                user_friendly_msg = "⏱️ 请求频率过高\n\n请稍等片刻后重试。"
            else:
                error_msg = f"DeepSeek API 调用失败: {error_str}"
                user_friendly_msg = f"❌ API 调用失败\n\n{error_str}\n\n请检查网络连接或联系技术支持。"
            
            print(f"[DeepSeek] 错误: {error_msg}")
            print(f"[DeepSeek] 错误详情: {traceback.format_exc()}")
            return {
                "type": "error",
                "message": user_friendly_msg,
                "technical_error": error_msg,
            }


class HybridSearchService(SearchService):
    """混合搜索服务（结合搜索引擎和大模型）"""
    
    def __init__(self, search_service: Optional[SearchService] = None, llm_service: Optional[LLMSearchService] = None) -> None:
        self.search_service = search_service or BaiduSearchService()
        self.llm_service = llm_service
    
    def search(self, keyword: str) -> dict[str, Any]:
        """混合搜索：先获取大模型回答，再提供搜索链接"""
        print(f"[混合搜索] 开始处理关键词: {keyword}")
        result = {
            "keyword": keyword,
            "search_url": None,
            "llm_answer": None,
        }
        
        # 获取搜索引擎URL
        print(f"[混合搜索] 获取搜索引擎URL...")
        search_result = self.search_service.search(keyword)
        if search_result.get("type") == "redirect":
            result["search_url"] = search_result.get("url")
            print(f"[混合搜索] 搜索URL: {result['search_url']}")
        
        # 如果配置了大模型，获取智能回答
        if self.llm_service:
            print(f"[混合搜索] 调用LLM服务...")
            llm_result = self.llm_service.search(keyword)
            print(f"[混合搜索] LLM结果类型: {llm_result.get('type')}")
            if llm_result.get("type") == "llm_answer":
                result["llm_answer"] = llm_result.get("answer")
                print(f"[混合搜索] 成功获取AI回答，长度: {len(result['llm_answer'])} 字符")
            elif llm_result.get("type") == "error":
                print(f"[混合搜索] LLM调用失败: {llm_result.get('message')}")
                result["llm_error"] = llm_result.get("message")
        else:
            print(f"[混合搜索] 警告: LLM服务未配置")
        
        return result


def get_search_service() -> SearchService:
    """获取搜索服务实例"""
    search_type = os.getenv("SEARCH_SERVICE_TYPE", "baidu").lower()
    print(f"[搜索服务] 服务类型: {search_type}")
    
    if search_type == "google":
        return GoogleSearchService()
    elif search_type == "llm" or search_type == "openai" or search_type == "deepseek":
        llm_service = LLMSearchService()
        print(f"[搜索服务] 创建LLM服务，API密钥: {os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY') or 'None'}")
        return llm_service
    elif search_type == "hybrid":
        search_svc = BaiduSearchService()
        # 优先检查DeepSeek API密钥
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        llm_key = os.getenv("LLM_API_KEY")
        has_llm_key = bool(deepseek_key or openai_key or llm_key)
        
        print(f"[搜索服务] 混合模式 - DeepSeek密钥: {'已设置' if deepseek_key else '未设置'}")
        print(f"[搜索服务] 混合模式 - OpenAI密钥: {'已设置' if openai_key else '未设置'}")
        print(f"[搜索服务] 混合模式 - LLM密钥: {'已设置' if llm_key else '未设置'}")
        print(f"[搜索服务] 混合模式 - 是否有LLM密钥: {has_llm_key}")
        
        llm_svc = LLMSearchService() if has_llm_key else None
        if llm_svc:
            print(f"[搜索服务] LLM服务已创建")
        else:
            print(f"[搜索服务] 警告: 未找到LLM API密钥，LLM服务未创建")
        
        return HybridSearchService(search_service=search_svc, llm_service=llm_svc)
    else:
        # 默认使用百度搜索
        print(f"[搜索服务] 使用默认百度搜索")
        return BaiduSearchService()

