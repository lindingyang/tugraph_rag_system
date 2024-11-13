import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import json
from llm.api import get_gpt_response
from tqdm import tqdm
from rag.rag_without_langchain_bge_embedding import initialize_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


# 读取 web_paths
web_path_directory = "../dataset/web_paths_split_full_best.json"
with open(web_path_directory, 'r') as file:
    data = json.load(file)
    web_paths = data["web_paths"]

# 加载模型
model_name = "../embedding_model/bge-large-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章:"
)

print("成功加载模型")

# 加载文档并进行分块
loader = WebBaseLoader(web_paths=tuple(web_paths))
docs = loader.load()

chunk_size = 512
chunk_overlap = 256
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

splits = text_splitter.split_documents(docs)

num_questions = 1  # 基于每对 chunks 生成 1 个问题


# 生成问题并实时保存
with open("../dataset/sft_data/generated_tugraphdb_questions_gpt4omini_each_single_chunk.jsonl", "a") as f:
    for chunk in tqdm(splits, desc="Generating Questions"):
        context_text = chunk.page_content  # 提取切割后的文本

        # 创建 prompt，包含 chunk 的文本
        prompt = (
            "你是一个TuGraph-DB问答任务的助手。"
            f"请基于提供的相关信息，生成{num_questions}个TuGraphDB相关的问题。"
            "请确保你生成的问题简明扼要。"
            "\n\n"
            f"相关信息: {context_text}\n"
            f"生成的问题:"
        )

        print(prompt)  # 可选：调试时打印 prompt
        question = get_gpt_response(prompt)  # 直接返回的是字符串

        print(f"Question: {question}")  # 打印生成的问题

        # 实时保存生成的问题
        question_entry = {"question": question}
        f.write(json.dumps(question_entry, ensure_ascii=False) + "\n")