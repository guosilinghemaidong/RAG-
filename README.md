# 基于RAG的企业知识库智能问答系统

面向企业内部文档检索场景，实现多格式文档的自动解析、语义检索与智能问答生成。

## ✨ 核心特性

- 🔍 混合检索：向量语义检索 + BM25关键词检索，互补提升召回率
- 🔄 重排序优化：BGE-Reranker精排，降低幻觉率
- 📎 来源溯源：回答附带引用文档片段，可追溯验证
- 📄 多格式支持：PDF / TXT / Markdown 文档自动解析
- 🌐 Web界面：Streamlit交互式问答界面

## 🏗️ 技术架构

- 检索层：LangChain + Chroma + BM25
- 排序层：BGE-Reranker-v2-m3
- 生成层：DeepSeek / 通义千问 API
- 前端层：Streamlit

## 🚀 快速开始

```bash
# 1. 克隆项目
git clone https://gitee.com/你的用户名/rag-knowledge-qa.git
cd rag-knowledge-qa

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置API密钥
cp .env.example .env
# 编辑 .env 文件，填入你的API Key

# 4. 启动应用
streamlit run app.py
```

## 📊 性能指标

| 策略 | 准确率 | 召回率 | 幻觉率 |
|------|--------|--------|--------|
| 纯向量检索 | ~72% | ~65% | ~25% |
| 混合检索 | ~85% | ~78% | ~15% |
| 混合检索+重排序 | ~91% | ~87% | ~8% |

## 📁 项目结构

```
rag-knowledge-qa/
├── README.md              # 项目说明
├── .gitignore             # Git忽略规则
├── .env.example           # API密钥模板（不含真实密钥）
├── requirements.txt       # Python依赖清单
├── app.py                 # Streamlit主界面
├── rag_pipeline.py        # RAG核心流程
├── document_loader.py     # 文档加载与分块
├── retriever.py           # 混合检索+重排序
├── evaluator.py           # 评估模块
└── data/                  # 示例文档数据
```

## 📝 开发日志

- Day 1: 搭建Naive RAG骨架（文档加载→分块→向量化→检索→生成）
- Day 2: Streamlit Web界面 + 多格式文档支持
- Day 3: 混合检索（向量+BM25）+ BGE-Reranker重排序
- Day 4: 评估体系 + 量化优化
- Day 5: 部署上线 + 简历撰写

## 📜 License

MIT
