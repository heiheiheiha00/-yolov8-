# DeepSeek API 配置说明

## 快速配置

在项目根目录创建或编辑 `.env` 文件，添加以下配置：

```env
# 使用混合模式（推荐）：同时显示AI回答和搜索链接
SEARCH_SERVICE_TYPE=hybrid
DEEPSEEK_API_KEY=sk-fabfee39f2724e2692d2872aefafadfc
```

或者仅使用 LLM 模式（只显示AI回答，不显示搜索链接）：

```env
SEARCH_SERVICE_TYPE=llm
DEEPSEEK_API_KEY=sk-fabfee39f2724e2692d2872aefafadfc
```

## 配置说明

- `SEARCH_SERVICE_TYPE=hybrid`：混合模式，会同时显示：
  - DeepSeek AI 生成的智能回答
  - 百度搜索链接
  
- `SEARCH_SERVICE_TYPE=llm`：仅 LLM 模式，只显示 AI 回答

- `DEEPSEEK_API_KEY`：你的 DeepSeek API 密钥（已提供）

## 可选配置

如果需要自定义 API 端点或模型，可以添加：

```env
LLM_API_BASE=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

## 使用

配置完成后，重启 Flask 应用，点击详细建议页面中的关键词标签即可使用 DeepSeek 生成智能回答。

## 注意事项

1. `.env` 文件不会被提交到 Git（已在 .gitignore 中）
2. 确保已安装 `openai` 库：`pip install openai`
3. DeepSeek 兼容 OpenAI API 格式，使用相同的库

