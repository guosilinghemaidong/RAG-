"""
RAG核心流程 - Day 1: Naive RAG骨架
=========================================

完整流程：文档加载 → 分块 → 向量化 → 存储 → 检索 → LLM生成

这个文件是RAG项目的心脏，面试时你要能逐行讲清楚每一步在做什么。
"""

import os
from dotenv import load_dotenv

# ========== 第一步：加载环境变量（API密钥）==========
load_dotenv()  # 从 .env 文件加载配置
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ========== 第二步：文档加载与分块 ==========

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(file_path):
    """
    加载文档文件
    支持 .txt 和 .pdf 格式

    面试考点：为什么需要文档加载？
    → 因为原始文档是非结构化的，需要先读取内容才能后续处理
    """
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

    docs = loader.load()
    print(f"✅ 加载文档成功，共 {len(docs)} 页/段")
    return docs


def split_documents(docs, chunk_size=500, chunk_overlap=50):
    """
    文档分块（Chunking）

    面试考点（必背！）：
    - chunk_size: 每块的字符数，太大包含噪声，太小丢失上下文
    - chunk_overlap: 相邻块的重叠字符数，防止关键信息被切断
    - separators: 按优先级尝试分割，先按段落(\n\n)，再按句子(\n)，最后按字符

    为什么分块？
    → LLM有上下文窗口限制，不能一次性读入整篇文档
    → 小块检索精度更高，能准确定位到相关段落
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ",", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ 分块完成，共 {len(chunks)} 个文本块")
    print(f"   块大小: {chunk_size} 字符, 重叠: {chunk_overlap} 字符")

    # 打印一个示例块看看效果
    if chunks:
        print(f"\n📝 示例分块（第1块）：")
        print(f"   内容: {chunks[0].page_content[:100]}...")
        print(f"   来源: {chunks[0].metadata.get('source', '未知')}")

    return chunks


# ========== 第三步：向量化与存储 ==========

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def create_vectorstore(chunks, persist_directory="./chroma_db"):
    """
    创建向量数据库

    面试考点（必背！）：
    - Embedding是什么？把文本变成高维向量（数字数组），语义相似的文本向量距离近
    - 为什么用BGE-M3？中文效果最好的开源Embedding模型，在MTEB榜单排名靠前
    - Chroma是什么？轻量级本地向量数据库，适合学习和小项目；生产环境用Milvus
    - persist_directory: 向量数据持久化保存的路径，下次不用重新向量化

    数据流程：
    文本块 → Embedding模型 → 向量(768维数字数组) → 存入Chroma → 持久化保存
    """
    print("⏳ 正在加载Embedding模型（首次运行需下载，约2GB，请耐心等待）...")

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},  # CPU运行，不需要显卡
        encode_kwargs={"normalize_embeddings": True},  # 归一化，方便余弦相似度计算
    )

    print("✅ Embedding模型加载完成")
    print("⏳ 正在向量化文档块并存储到Chroma...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory,
    )

    # 新版Chroma自动持久化，不需要手动调用persist()

    print(f"✅ 向量数据库创建完成，保存在 {persist_directory}")
    print(f"   共存储 {len(chunks)} 个文档块的向量")

    return vectorstore


def load_vectorstore(persist_directory="./chroma_db"):
    """
    加载已有的向量数据库（不需要重新向量化）

    生产环境中，文档入库是一次性的，查询时只需要加载已有的向量库
    """
    print(f"⏳ 正在加载向量数据库: {persist_directory}")

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
    )

    print("✅ 向量数据库加载完成")
    return vectorstore


# ========== 第四步：检索 + LLM生成 ==========

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def create_llm():
    """
    创建LLM实例
    使用DeepSeek API（兼容OpenAI接口格式）

    面试考点：
    - 为什么用DeepSeek？便宜（约1元/百万token）、中文能力强、API兼容OpenAI格式
    - temperature=0：让回答更稳定、更准确，减少幻觉（RAG场景很重要）
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE,
        temperature=0,  # RAG场景用低temperature，让回答更忠实于检索到的文档
    )
    return llm


def format_docs(docs):
    """把检索到的文档列表格式化为一段文本"""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vectorstore, k=5):
    """
    创建RAG问答链（使用LCEL - LangChain Expression Language）

    面试考点（最核心！）：
    - LCEL是什么？LangChain的新式链路写法，用 | 管道符串联各步骤，比旧版更灵活
    - k=5：检索Top-5最相关的文档块，作为上下文送给LLM

    链路流程（用管道符 | 串联）：
    {"question": 问题, "context": retriever | format_docs}
        → Prompt模板 → LLM生成 → 输出解析

    对比旧版RetrievalQA：
    - 旧版：RetrievalQA.from_chain_type(llm, retriever)  → 已废弃
    - 新版：retriever | format_docs | prompt | llm  → 推荐，更灵活
    """
    llm = create_llm()

    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 相似度搜索
        search_kwargs={"k": k},    # 返回Top-K个结果
    )

    # 自定义Prompt模板
    prompt_template = """你是一个企业知识库问答助手。请严格基于以下参考文档来回答用户的问题。
如果参考文档中没有相关信息，请直接回答"根据现有知识库，我无法回答这个问题"，不要编造内容。

参考文档：
{context}

用户问题：{question}

请给出准确、完整的回答，并在回答末尾标注参考来源："""

    QA_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    # 用LCEL管道串联各步骤
    rag_chain = (
        # 第一步：组装输入 - question原样传递，context从检索器获取后格式化
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        # 第二步：填入Prompt模板
        | QA_PROMPT
        # 第三步：送入LLM生成回答
        | llm
        # 第四步：解析输出为纯文本
        | StrOutputParser()
    )

    print("✅ RAG问答链创建完成（LCEL方式）")
    return rag_chain, retriever


def ask_question(rag_chain, retriever, question):
    """
    执行问答

    面试考点：
    - rag_chain.invoke(question)：LCEL链路的执行方法
    - retriever.invoke(question)：单独获取来源文档（用于引用溯源）
    """
    print(f"\n🔍 问题: {question}")
    print("⏳ 正在检索和生成回答...")

    # 获取回答
    answer = rag_chain.invoke(question)

    # 获取来源文档（引用溯源，RAG核心价值）
    source_docs = retriever.invoke(question)

    # 打印回答
    print(f"\n💡 回答:\n{answer}")

    # 打印来源文档
    print(f"\n📎 参考来源:")
    for i, doc in enumerate(source_docs, 1):
        source = doc.metadata.get('source', '未知')
        content_preview = doc.page_content[:80].replace('\n', ' ')
        print(f"   [{i}] {source} → {content_preview}...")

    return {"answer": answer, "source_documents": source_docs}

# ... existing code ...

# ========== 第五步：BM25关键词检索 ==========

import pickle
from rank_bm25 import BM25Okapi


def save_chunks_for_bm25(chunks, save_path="./data/chunks.pkl"):
    """
    把分块结果保存到本地文件，供BM25建索引用

    为什么要保存？
    → BM25需要原始文本块来建索引，但分块只在文档处理时产生
    → 保存到pkl文件，下次直接用，不用重新分块
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅ 分块已保存到 {save_path}，共 {len(chunks)} 个块")


def load_chunks_for_bm25(load_path="./data/chunks.pkl"):
    """
    加载之前保存的分块数据
    如果pkl文件不存在，自动从源文档重新分块生成
    """
    if os.path.exists(load_path):
        with open(load_path, "rb") as f:
            chunks = pickle.load(f)
        print(f"✅ 从 {load_path} 加载了 {len(chunks)} 个分块")
        return chunks
    else:
        print(f"⚠️ {load_path} 不存在，自动从源文档重新分块...")
        # 尝试从默认文档路径重新分块
        default_doc = "./data/knowledge.txt"
        if os.path.exists(default_doc):
            docs = load_documents(default_doc)
            chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
            save_chunks_for_bm25(chunks, load_path)
            return chunks
        else:
            raise FileNotFoundError(
                f"找不到 {load_path}，也没有找到默认文档 {default_doc}。\n"
                "请在Streamlit中重新上传文档，或手动将文档放到 data/ 目录下。"
            )


def create_bm25_retriever(chunks, k=5):
    """
    创建BM25检索器

    面试考点（必背！）：
    - BM25是什么？经典的关键词检索算法，基于词频(TF)和逆文档频率(IDF)
    - 和向量检索的区别？
      → 向量检索：擅长语义理解（"退款"能匹配"返还费用"）
      → BM25：擅长精确匹配（"503错误码"、"工号A123"这类编号/专有名词）
    - 为什么要混合？互补！语义相近的用向量，精确关键词的用BM25

    参数说明：
    - corpus: 分块的文本列表，BM25在此基础上建倒排索引
    - k1: 控制词频饱和度（越高，高频词权重越大）
    - b: 控制文档长度归一化（0=不归一化，1=完全归一化）
    """
    corpus = [chunk.page_content for chunk in chunks]

    # 中文分词：简单按字符分割（生产环境用jieba分词）
    tokenized_corpus = [list(doc) for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

    def bm25_search(query, top_k=k):
        """BM25检索函数，返回(top_k个文档, 对应分数)"""
        tokenized_query = list(query)
        scores = bm25.get_scores(tokenized_query)
        # 取Top-K最高分的索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        result_scores = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有分数的结果
                results.append(chunks[idx])
                result_scores.append(scores[idx])
        return results, result_scores

    return bm25_search


# ========== 第六步：混合检索（手动合并） ==========

def hybrid_retrieval(query, vector_retriever, bm25_search, chunks,
                     vector_weight=0.6, bm25_weight=0.4, k=5):
    """
    混合检索：合并向量检索和BM25的结果

    面试考点（核心！）：
    - 为什么要手动合并而不用EnsembleRetriever？
      → 因为langchain新版本移除了这个类，手动合并更灵活可控
    - 合并策略：RRF（Reciprocal Rank Fusion）倒数排名融合
      → 公式：score = Σ 1/(rank + constant)，rank越小（排名越靠前）得分越高
      → 优点：不需要归一化不同检索器的分数，直接按排名融合

    参数说明：
    - vector_weight / bm25_weight: 两种检索的权重比例
    - k: 最终返回的Top-K个结果
    """
    # 1. 分别检索，各取Top-K*2（多取一些，合并后再筛选）
    vector_docs = vector_retriever.invoke(query)
    bm25_docs, bm25_scores = bm25_search(query, top_k=k * 2)

    # 2. 给每个文档打分并合并
    # 用文档内容作为key去重，避免同一个块被两种检索都选到
    doc_scores = {}  # {文档内容: {"doc": 文档对象, "score": 综合得分, "sources": 来源标记}}

    # 向量检索结果打分（按排名，排名越靠前分越高）
    for rank, doc in enumerate(vector_docs[:k * 2]):
        content = doc.page_content
        # RRF公式：1 / (rank + 60)，60是平滑常数
        rrf_score = vector_weight / (rank + 1)
        if content in doc_scores:
            doc_scores[content]["score"] += rrf_score
            doc_scores[content]["sources"].append("向量")
        else:
            doc_scores[content] = {
                "doc": doc,
                "score": rrf_score,
                "sources": ["向量"],
            }

    # BM25检索结果打分
    for rank, (doc, bm25_score) in enumerate(zip(bm25_docs, bm25_scores)):
        content = doc.page_content
        rrf_score = bm25_weight / (rank + 1)
        if content in doc_scores:
            doc_scores[content]["score"] += rrf_score
            if "BM25" not in doc_scores[content]["sources"]:
                doc_scores[content]["sources"].append("BM25")
        else:
            doc_scores[content] = {
                "doc": doc,
                "score": rrf_score,
                "sources": ["BM25"],
            }

    # 3. 按综合得分排序，取Top-K
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    top_results = sorted_docs[:k]

    # 4. 分离文档和分数
    result_docs = [item["doc"] for item in top_results]
    result_scores = [item["score"] for item in top_results]
    result_sources = [item["sources"] for item in top_results]

    return result_docs, result_scores, result_sources



# ========== 第七步：SiliconFlow Reranker API 重排序 ==========

import requests


def siliconflow_rerank(query, docs, api_key=None, top_k=3):
    """
    使用SiliconFlow API对检索结果进行重排序

    面试考点（必背！）：
    - 什么是重排序（Reranking）？
      → 初步检索（向量/BM25）是"双塔模型"，query和doc分别编码再算相似度
      → 重排序是"交叉编码器"，把query和doc拼在一起送入模型，理解更深
      → 类比：初筛是HR看简历关键词，重排序是面试官仔细看简历
    - 为什么用SiliconFlow API而不是本地CrossEncoder？
      → 本地CrossEncoder在Windows上有scipy DLL兼容问题
      → SiliconFlow国内可用、免费、无需GPU、企业级质量
    - 模型：BAAI/bge-reranker-v2-m3，和Embedding用的BGE-M3同系列

    参数说明：
    - query: 用户问题
    - docs: 初步检索返回的文档列表
    - top_k: 重排序后保留的Top-K个
    """
    if api_key is None:
        api_key = os.getenv("SILICONFLOW_API_KEY", "")

    if not api_key:
        print("⚠️ 未配置SILICONFLOW_API_KEY，跳过重排序")
        return docs[:top_k], [1.0] * min(top_k, len(docs))

    # 调用SiliconFlow Reranker API
    url = "https://api.siliconflow.cn/v1/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 构造请求体
    documents_text = [doc.page_content for doc in docs]
    payload = {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": query,
        "documents": documents_text,
        "top_n": top_k,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        # 解析返回结果
        reranked_docs = []
        reranked_scores = []
        for item in data["results"]:
            idx = item["index"]
            reranked_docs.append(docs[idx])
            reranked_scores.append(item["relevance_score"])

        print(f"✅ SiliconFlow重排序完成，从 {len(docs)} 个文档中选出 Top-{top_k}")
        return reranked_docs, reranked_scores

    except requests.exceptions.RequestException as e:
        print(f"⚠️ SiliconFlow API调用失败: {e}，返回原始排序")
        return docs[:top_k], [1.0] * min(top_k, len(docs))




# ========== 主函数：一键跑通全流程 ==========

def main():
    """
    Day 1 主流程：从文档加载到问答生成的完整链路

    两种模式：
    1. 首次运行：需要加载文档 → 分块 → 向量化 → 存储（较慢，约3-5分钟）
    2. 后续运行：直接加载已有的向量库（很快，几秒钟）
    """
    PERSIST_DIR = "./chroma_db"
    DOC_PATH = "./data/knowledge.txt"  # 示例文档

    # 判断是否已有向量库
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("=" * 50)
        print("🚀 检测到已有向量数据库，直接加载")
        print("=" * 50)
        vectorstore = load_vectorstore(PERSIST_DIR)
    else:
        print("=" * 50)
        print("🚀 首次运行：完整RAG流程")
        print("=" * 50)

        # Step 1: 加载文档
        docs = load_documents(DOC_PATH)

        # Step 2: 分块
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)

        # Step 3: 向量化 + 存储
        vectorstore = create_vectorstore(chunks, PERSIST_DIR)

    # Step 4: 创建问答链
    rag_chain, retriever = create_rag_chain(vectorstore, k=5)

    # Step 5: 测试问答！
    print("\n" + "=" * 50)
    print("🎯 开始测试问答")
    print("=" * 50)

    # 测试几个典型问题
    test_questions = [
        "公司的退款流程是什么？",
        "退款代码503是什么意思？",
        "年假有多少天？",
        "公司用什么向量数据库？",
    ]

    for q in test_questions:
        ask_question(rag_chain, retriever, q)
        print("-" * 40)


if __name__ == "__main__":
    main()