# RAG 知识库问答系统

基于检索增强生成(Retrieval-Augmented Generation, RAG)的本地知识库问答系统，支持自定义知识库的智能问答。

## 系统架构

系统主要由以下几个核心模块组成：

1. 知识库管理(kb.py)
   - 支持文本知识库的创建和管理
   - 文本分块和预处理
   - 向量化存储

2. 检索引擎(index.py)
   - 基于向量相似度的知识检索
   - 支持Top-K相关片段获取

3. API服务(api.py)
   - RESTful API接口
   - 支持知识库问答

4. RAG实现(rag.py)
   - 检索增强的问答生成
   - 上下文相关性优化

## 快速开始

### 环境配置

1. 克隆代码库
```bash
git clone [repository_url]
cd rag
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置文件
- 复制配置模板
```bash
cp config_template.json config.json
```
- 修改config.json，配置必要的API密钥和参数

### 使用示例

1. 初始化知识库
```python
from kb import KnowledgeBase

# 创建知识库实例
kb = KnowledgeBase()

# 添加文档
kb.add_document("私人知识库.txt")
```

2. 启动API服务
```bash
python api.py
```

3. 发送问答请求
```bash
curl -X POST "http://localhost:8000/qa" \
     -H "Content-Type: application/json" \
     -d '{"question":"张小刚的联系方式是什么？"}'
```

## 核心功能

### 1. 知识库管理
- 支持txt、pdf等多种格式文档导入
- 自动文本分块，保持语义完整性
- 向量化存储，支持高效检索

### 2. 智能检索
- 基于向量相似度的语义检索
- 支持上下文相关性排序
- 可配置的Top-K检索数量

### 3. 问答生成
- 基于检索结果的上下文感知回答
- 支持多轮对话
- 答案准确性和相关性优化

## 配置说明

config.json主要配置项：

```json
{
    "api_key": "your-api-key",
    "embedding_model": "text-embedding-ada-002",
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_k": 3
}
```

- api_key: OpenAI API密钥
- embedding_model: 向量化模型选择
- max_tokens: 生成答案的最大token数
- temperature: 生成答案的随机性
- top_k: 检索的相关文档数量

## 日志管理

系统日志存储在logs目录下，包含：
- 运行日志
- 错误日志
- API访问日志

## 注意事项

1. 请确保config.json中配置了有效的API密钥
2. 大型文档建议适当调整分块大小
3. 检索参数可根据实际需求调整

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进系统。

## 许可证

MIT License