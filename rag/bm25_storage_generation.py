import os
import json
import jieba
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

web_path_directory = "../dataset/web_paths_split_full_manual_refine_table2text_final.json"

# 从 JSON 文件中读取 web_paths
with open(web_path_directory, 'r') as file:
    data = json.load(file)
    web_paths = data["web_paths"]

# 加载文档
loader = WebBaseLoader(web_paths=tuple(web_paths))

docs = loader.load()

# 分词
def cut_words(text):
    return jieba.lcut(text)

# 分块处理
chunk_size = 1024
chunk_overlap = 512
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
split_docs = text_splitter.split_documents(docs)

# 保存分块和分词结果到文件
processed_chunks = [
    {
        'source': doc.metadata['source'],
        'document': doc.page_content,
        'chunks': [
            {
                'content': chunk.page_content,
            }
            for chunk in split_docs if chunk.metadata['source'] == doc.metadata['source']
        ]
    }
    for doc in docs
]

# 创建保存目录（如果不存在）
output_directory = '../bm25_storage_split_full_manual_refine_1024_512_table2text_final/'
os.makedirs(output_directory, exist_ok=True)

# 保存到文件
with open(os.path.join(output_directory, 'processed_chunks.json'), 'w', encoding='utf-8') as f:
    json.dump(processed_chunks, f, ensure_ascii=False, indent=4)

print("分块结果已保存到 ../bm25_storage_split_full_manual_refine_1024_512_table2text_final/processed_chunks.json")
