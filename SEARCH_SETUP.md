# 搜索功能配置指南

本项目已集成搜索引擎和大模型API，支持联网搜索功能。

## 功能说明

点击详细建议页面中的关键词标签，可以：
1. **搜索引擎搜索**：跳转到百度/Google搜索相关训练内容
2. **大模型回答**：使用AI生成专业的训练建议和说明
3. **混合模式**：同时提供AI回答和搜索链接

## 配置方式

### 方式一：使用默认百度搜索（无需配置）

默认使用百度搜索，无需任何配置即可使用。点击关键词会直接跳转到百度搜索结果页面。

### 方式二：使用Google搜索API

1. 获取Google Custom Search API密钥：
   - 访问 [Google Cloud Console](https://console.cloud.google.com/)
   - 创建项目并启用 Custom Search API
   - 创建API密钥和搜索引擎ID

2. 在 `.env` 文件中添加：
   ```env
   SEARCH_SERVICE_TYPE=google
   GOOGLE_SEARCH_API_KEY=你的API密钥
   GOOGLE_SEARCH_ENGINE_ID=你的搜索引擎ID
   ```

### 方式三：使用大模型API（DeepSeek/OpenAI等）

#### 使用DeepSeek API（推荐，性价比高）

1. 获取DeepSeek API密钥：
   - 访问 [DeepSeek 官网](https://www.deepseek.com/)
   - 注册账号并获取API密钥

2. 在 `.env` 文件中添加：
   ```env
   SEARCH_SERVICE_TYPE=llm
   DEEPSEEK_API_KEY=你的DeepSeek API密钥
   # 可选配置
   LLM_API_BASE=https://api.deepseek.com/v1  # 默认值，可不设置
   LLM_MODEL=deepseek-chat  # 默认值，可不设置
   ```

3. 安装OpenAI库（DeepSeek兼容OpenAI API格式）：
   ```bash
   pip install openai
   ```

#### 使用OpenAI API

1. 获取OpenAI API密钥

2. 在 `.env` 文件中添加：
   ```env
   SEARCH_SERVICE_TYPE=llm
   OPENAI_API_KEY=你的OpenAI API密钥
   LLM_API_BASE=https://api.openai.com/v1  # 可选，默认值
   LLM_MODEL=gpt-3.5-turbo  # 可选，默认值
   ```

3. 安装OpenAI库：
   ```bash
   pip install openai
   ```

### 方式四：混合模式（推荐）

同时使用搜索引擎和大模型，提供最完整的搜索体验：

```env
SEARCH_SERVICE_TYPE=hybrid
OPENAI_API_KEY=你的API密钥
```

## 环境变量说明

| 变量名 | 说明 | 默认值 | 必需 |
|--------|------|--------|------|
| `SEARCH_SERVICE_TYPE` | 搜索服务类型：`baidu`、`google`、`llm`、`hybrid`、`deepseek` | `baidu` | 否 |
| `BAIDU_SEARCH_API_KEY` | 百度搜索API密钥（如需要） | - | 否 |
| `GOOGLE_SEARCH_API_KEY` | Google搜索API密钥 | - | 否（使用Google时） |
| `GOOGLE_SEARCH_ENGINE_ID` | Google搜索引擎ID | - | 否（使用Google时） |
| `DEEPSEEK_API_KEY` | DeepSeek API密钥（推荐） | - | 否（使用DeepSeek时） |
| `OPENAI_API_KEY` | OpenAI API密钥 | - | 否（使用OpenAI时） |
| `LLM_API_KEY` | 通用LLM API密钥（DeepSeek/OpenAI API密钥的别名） | - | 否 |
| `LLM_API_BASE` | LLM API基础URL | `https://api.deepseek.com/v1`（有DEEPSEEK_API_KEY时）<br>`https://api.openai.com/v1`（有OPENAI_API_KEY时） | 否 |
| `LLM_MODEL` | LLM模型名称 | `deepseek-chat`（DeepSeek）<br>`gpt-3.5-turbo`（OpenAI） | 否 |

## 使用示例

### 示例1：仅使用百度搜索（默认）

无需配置，直接使用。

### 示例2：使用DeepSeek生成智能回答

`.env` 文件：
```env
SEARCH_SERVICE_TYPE=llm
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx
```

点击关键词后，会显示AI生成的训练建议和说明。

### 示例2b：使用OpenAI生成智能回答

`.env` 文件：
```env
SEARCH_SERVICE_TYPE=llm
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
```

### 示例3：混合模式（使用DeepSeek）

`.env` 文件：
```env
SEARCH_SERVICE_TYPE=hybrid
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx
```

点击关键词后，会同时显示：
- AI生成的智能回答
- 搜索引擎链接

## 支持的LLM服务

理论上支持任何兼容OpenAI API格式的LLM服务，包括：
- **DeepSeek**（推荐，性价比高，中文支持好）
- OpenAI GPT系列
- Azure OpenAI
- 其他兼容OpenAI API的服务

只需设置 `LLM_API_BASE` 指向相应的API端点即可。

### DeepSeek配置示例

```env
SEARCH_SERVICE_TYPE=llm
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx
LLM_API_BASE=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

## 注意事项

1. **API密钥安全**：请勿将API密钥提交到代码仓库，使用 `.env` 文件管理
2. **API费用**：使用大模型API可能产生费用，请注意使用量
3. **网络访问**：确保服务器可以访问相应的API服务
4. **依赖安装**：使用LLM功能需要安装 `openai` 库：`pip install openai`

## 故障排除

### 问题1：点击关键词没有反应

- 检查浏览器控制台是否有错误
- 确认Flask应用正常运行
- 检查路由是否正确配置

### 问题2：大模型API调用失败

- 检查API密钥是否正确
- 确认网络可以访问API服务
- 查看Flask控制台的错误信息
- 确认已安装 `openai` 库

### 问题3：搜索结果为空

- 检查搜索引擎API配置
- 确认API密钥有效
- 查看Flask控制台的错误信息

