import random
import time
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import json
from llm.api import get_response
from tqdm import tqdm

# 读取 web_paths
web_path_directory = "../dataset/web_paths.json"
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

chunk_size = 1024
chunk_overlap = 512
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

splits = text_splitter.split_documents(docs)

num_questions = 1  # 基于每对 chunks 生成 1 个问题
num_chunks = 2  # 每次随机选择 2 个 chunk

# 生成问题并实时保存
with open("../dataset/sft_data/generated_tugraphdb_questions_gpt4omini_crossover.jsonl", "a") as f:
    for _ in tqdm(range(len(splits)), desc="Generating Questions"):
        # 随机选择两个不同的 chunks
        selected_chunks = random.sample(splits, num_chunks)

        # 分别获取两个 chunk 的文本
        chunk1_text = selected_chunks[0].page_content
        chunk2_text = selected_chunks[1].page_content

        # 创建 prompt，分别包含两个 chunk 的文本
        prompt = (
            "你是一个TuGraph-DB问答任务的助手。"
            f"请基于提供的两个不同的相关信息，生成{num_questions}个TuGraphDB相关的问题。"
            "请确保你生成的问题简明扼要。"
            "\n\n"
            f"相关信息1: {chunk1_text}\n"
            f"相关信息2: {chunk2_text}\n"
            f"生成的问题:"
        )

        print(prompt)  # 可选：调试时打印 prompt，调试完成后可以去掉
        questions = get_response(prompt)

        print(f"Questions: {questions}")  # 打印生成的问题

        # 实时保存生成的问题
        question_entry = {"question": questions}
        f.write(json.dumps(question_entry, ensure_ascii=False) + "\n")
        time.sleep(1)  # 防止 API 请求速率过高
