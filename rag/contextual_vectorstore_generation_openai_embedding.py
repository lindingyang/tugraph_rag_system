import json
import os
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


def load_and_situate_context(file_path: str, vector_store_output_path: str):
    model_name = "text-embedding-3-large"
    embedding = OpenAIEmbeddings(model=model_name)
    print(f"成功加载{model_name}模型")

    if not os.path.exists(vector_store_output_path):
        os.makedirs(vector_store_output_path)

    with open(file_path, 'r') as file:
        embedding_chunks = []
        data = json.load(file)
        for item in tqdm(data, desc="Processing documents", unit="doc"):
            source = item.get("source", "")
            chunks = item.get("chunks", [])
            for chunk in chunks:
                text_to_embed = chunk.get("content", "")
                embedding_chunks.append(Document(metadata={"source": source}, page_content=text_to_embed))

        # 保存contextual_vector_store向量数据库
        db = FAISS.from_documents(embedding_chunks, embedding=embedding)
        db.save_local(vector_store_output_path)
        print("contextual向量数据库已成功保存。")


if __name__ == '__main__':
    load_and_situate_context(
        file_path="../contextual_bm25_storage_split_full_manual_refine_1024_512_table2text_final/processed_chunks.json",
        vector_store_output_path="../contextual_vectorstore_1024_512_split_full_manual_refine_openai_embedding_table2text_final"
    )
