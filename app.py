"""
Streamlit Web界面 - Day 2
=========================================

把Day 1的RAG流程包装成可交互的网页应用，支持：
1. 上传文档（PDF/TXT/Markdown）
2. 输入问题，实时获取回答
3. 显示引用来源（溯源，RAG核心价值）
4. 显示检索到的原始文档片段

运行方式：streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# 导入Day 1的核心模块
from rag_pipeline import (
    load_documents,
    split_documents,
    create_vectorstore,
    load_vectorstore,
    create_llm,
    format_docs,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PERSIST_DIR = "./chroma_db"

# ========== Streamlit页面配置 ==========

st.set_page_config(
    page_title="企业知识库问答系统",
    page_icon="📚",
    layout="wide",  # 宽布局，更美观
)

# ========== 页面标题和说明 ==========

st.title("📚 企业知识库智能问答系统")
st.caption("基于RAG（检索增强生成）架构 · 支持PDF/TXT文档上传 · 回答可溯源验证")

# ========== 侧边栏：文档管理 ==========

with st.sidebar:
    st.header("📁 文档管理")

    # 上传文档
    uploaded_files = st.file_uploader(
        "上传知识库文档",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="支持PDF、TXT、Markdown格式，可同时上传多个文件",
    )

    # 处理上传的文档
    if uploaded_files and st.button("🔄 处理文档", type="primary"):
        # 保存上传的文件到临时目录
        save_dir = "./data"
        os.makedirs(save_dir, exist_ok=True)

        all_chunks = []
        progress = st.progress(0, text="正在处理文档...")

        for i, uploaded_file in enumerate(uploaded_files):
            # 保存文件
            file_path = os.path.join(save_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 加载+分块
            try:
                docs = load_documents(file_path)
                chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
                all_chunks.extend(chunks)
            except Exception as e:
                st.error(f"处理 {uploaded_file.name} 失败: {e}")
                continue

            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"已处理 {i + 1}/{len(uploaded_files)} 个文件",
            )

        # 向量化+存储
        if all_chunks:
            # 删除旧的向量库，重新创建
            import shutil
            if os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR)

            progress.progress(0.8, text="正在向量化并存储到数据库...")

            embedding = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=embedding,
                persist_directory=PERSIST_DIR,
            )

            progress.progress(1.0, text=f"✅ 完成！共处理 {len(all_chunks)} 个文本块")

            st.success(f"✅ 文档处理完成！共 {len(uploaded_files)} 个文件，{len(all_chunks)} 个文本块")

            # 保存vectorstore到session_state，方便后续检索
            st.session_state["vectorstore"] = vectorstore
        else:
            st.warning("没有成功处理任何文档")

    # 显示当前知识库状态
    st.divider()
    st.subheader("📊 知识库状态")
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        # 加载已有向量库获取信息
        try:
            existing_vs = load_vectorstore(PERSIST_DIR)
            collection = existing_vs._collection
            chunk_count = collection.count()
            st.info(f"✅ 知识库已就绪\n- 文档块数量: {chunk_count}\n- 向量数据库: Chroma\n- Embedding模型: BGE-M3")
        except Exception:
            st.info("✅ 知识库已就绪")
    else:
        st.warning("⚠️ 知识库为空，请先上传文档")

    # 参数设置
    st.divider()
    st.subheader("⚙️ 检索参数")
    k_value = st.slider("检索文档块数量 (Top-K)", min_value=1, max_value=10, value=5,
                         help="检索最相关的K个文档块，K越大上下文越多但可能引入噪声")

# ========== 主界面：问答区域 ==========

st.header("💬 智能问答")

# 获取或加载向量数据库
def get_vectorstore():
    """获取向量数据库实例"""
    if "vectorstore" in st.session_state:
        return st.session_state["vectorstore"]
    elif os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return load_vectorstore(PERSIST_DIR)
    else:
        return None

vectorstore = get_vectorstore()

if vectorstore is None:
    st.warning("⚠️ 请先在左侧上传文档并点击[处理文档]，才能开始问答")
    st.stop()

# 创建RAG链路
@st.cache_resource
def create_rag_components(_vectorstore, k):
    """
    创建RAG链路（缓存，避免每次提问都重建）
    
    注意：_vectorstore前面加_是因为Streamlit缓存机制，
    不加_的参数变化时缓存会失效，加_的参数不参与缓存判断
    """
    llm = create_llm()
    retriever = _vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    prompt_template = """你是一个企业知识库问答助手。请严格基于以下参考文档来回答用户的问题。
如果参考文档中没有相关信息，请直接回答"根据现有知识库，我无法回答这个问题"，不要编造内容。
回答时请尽量完整、详细，并在回答末尾注明参考来源。

参考文档：
{context}

用户问题：{question}

请给出准确、完整的回答："""

    QA_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


rag_chain, retriever = create_rag_components(vectorstore, k_value)

# 问答输入框
question = st.text_input(
    "请输入你的问题：",
    placeholder="例如：公司的退款流程是什么？",
)

# 提问按钮
if question and st.button("🔍 提问", type="primary"):
    with st.spinner("正在检索和生成回答..."):
        # 获取回答
        answer = rag_chain.invoke(question)

        # 获取来源文档
        source_docs = retriever.invoke(question)

    # 显示回答
    st.subheader("💡 回答")
    st.markdown(answer)

    # 显示引用来源
    st.subheader("📎 参考来源")
    for i, doc in enumerate(source_docs, 1):
        source = doc.metadata.get("source", "未知来源")
        # 只显示文件名，不显示完整路径
        source_name = os.path.basename(source) if source != "未知来源" else source

        with st.expander(f"来源 [{i}] - {source_name}"):
            st.text(doc.page_content)

# ========== 底部信息 ==========

st.divider()
st.caption(
    "技术架构：LangChain LCEL + BGE-M3 Embedding + Chroma + DeepSeek LLM | "
    "项目地址：[GitHub](https://github.com/guosilinghemaidong/RAG-)"
)
