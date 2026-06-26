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
