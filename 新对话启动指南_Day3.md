# RAG项目 - 新对话启动指南

## 已完成（Day 1-2）

### Day 1 ✅
- RAG核心流程：文档加载 → 分块 → 向量化(BGE-M3) → Chroma存储 → 相似度检索 → DeepSeek生成
- LangChain LCEL管道式链路
- Git/GitHub环境搭建（SSH配置、仓库创建、push流程）
- chunk_size参数对比实验（500最优，20信息丢失）
- 模拟企业文档：`data/knowledge.txt`（星云科技员工手册）

### Day 2 ✅
- Streamlit Web界面：文档上传 + 交互问答 + 引用溯源展示
- session_state + @st.cache_resource 缓存策略
- Day 2代码就是当前 `app.py` 和 `rag_pipeline.py`

### 项目当前文件结构
```
C:\Users\41434\Desktop\RAG学习\
├── .venv/               # Python虚拟环境
├── chroma_db/           # Chroma向量库（已有数据）
├── data/
│   └── knowledge.txt    # 模拟企业文档
├── .env                 # API密钥配置
├── .gitignore
├── app.py               # Day 2 Streamlit界面 ← 当前版本
├── rag_pipeline.py      # Day 2 RAG核心流程 ← 当前版本
├── requirements.txt
└── README.md
```

### 当前技术栈
LangChain LCEL + BGE-M3 Embedding + Chroma + DeepSeek + Streamlit

---

## Day 3 任务（待做）

### 目标
升级检索策略：**混合检索（向量+BM25）+ 重排序**

### 具体内容

1. **BM25关键词检索**
   - 用 `rank-bm25` 包 或 `langchain_community.retrievers.BM25Retriever`
   - 向量检索擅长语义，BM25擅长精确关键词（编号、专有名词）
   - 需要把分块结果保存到 `data/chunks.pkl`，供BM25建索引

2. **混合检索（手动合并）**
   - 不要用 `EnsembleRetriever`（这个类在你的langchain版本里已经不存在了）
   - 手动合并：向量检索结果 + BM25结果 → 去重 → 加权排序 → 取Top-K
   - 两种来源打标签区分，便于展示

3. **重排序**
   - 方案A（已放弃）：本地 CrossEncoder → scipy DLL问题，Windows不兼容
   - **方案B（推荐）**：Jina Reranker API → 免费、企业级、免部署
   - Jina API Key获取：https://jina.ai 注册
   - API地址：`https://api.jina.ai/v1/rerank`
   - 模型：`jina-reranker-v2-base-multilingual`

4. **UI升级**
   - 侧边栏加"检索模式"radio：向量检索 / 混合检索 / 混合检索+重排序
   - 混合模式加"向量权重"slider
   - 重排序模式加重排序分数表格展示

### 依赖安装
```bash
pip install rank-bm25 requests
```

### ⚠️ 已知问题
`sentence_transformers` 和 `scipy` 版本冲突导致 DLL 错误。**不要在 rag_pipeline.py 顶部导入 CrossEncoder**，重排序用 Jina API 即可避免此问题。

---

## Day 4（待做）
RAGAS评估框架：量化 Faithfulness / Answer Relevancy / Context Recall / Context Precision

## Day 5（待做）
Streamlit Cloud部署 + 简历终版

---

## 新对话启动指令

```
我正在做一个RAG项目的Day 3，项目在 C:\Users\41434\Desktop\RAG学习

Day 1-2已完成：
- RAG核心流程（文档加载→分块→向量化→检索→LLM生成）
- Streamlit Web界面（文档上传、交互问答、引用溯源）

Day 3要做：
升级检索策略，加BM25关键词检索 + Jina Reranker API重排序。
代码要保持手动合并（不用EnsembleRetriever，因为langchain版本里不存在了）。
重排序用Jina API（不用本地CrossEncoder，因为Windows有scipy DLL问题）。

技术栈：LangChain LCEL + BGE-M3 + Chroma + DeepSeek + Streamlit
```
