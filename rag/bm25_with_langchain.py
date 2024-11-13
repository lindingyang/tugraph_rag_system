import jieba
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
import json
from langchain_core.documents import Document


def cut_words(text):
    return jieba.lcut(text)


def create_bm25_retriever(file_path, k=3):
    # 从文件读取分块结果
    with open(file_path, 'r', encoding='utf-8') as f:
        processed_chunks = json.load(f)

    # 创建文档列表，将字典转换为 Document 对象
    documents = [
        Document(metadata={'source': doc['source']}, page_content=chunk['content'])
        for doc in processed_chunks for chunk in doc['chunks']
    ]

    # 创建 BM25Retriever，使用已分块和分词的文本
    retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=cut_words,  # 直接使用分词函数
        k=k
    )

    print("BM25Retriever 已创建，准备进行检索。")
    return retriever


if __name__ == '__main__':
    bm25_retriever = create_bm25_retriever('../bm25_storage_1024_512/processed_chunks.json', k=3)


    qa_sys_prompt = """你是一个TuGraph-DB问答任务的助手。 "
    对于下面的问题，请基于提供的相关信息进行回答。 "
    最多只用三句话回答，确保回答简明扼要。"
    \n\n"
    文档内容：{context}
    问题：{question}
    """


    qa_prompt = ChatPromptTemplate.from_template(qa_sys_prompt)


    llm = ChatOpenAI(model="gpt-4o-mini",
                     api_key=os.getenv("OPENAI_API_KEY"),
                     base_url=os.getenv("OPENAI_API_BASE"))


    # return_source_documents参数表示是否需要输出检索到的文本块内容
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=bm25_retriever, return_source_documents=True,
                                           chain_type_kwargs={"prompt": qa_prompt})

    question = '我们今天介绍的图，是图像的图（Image）吗？'
    print(qa_chain.invoke(question))