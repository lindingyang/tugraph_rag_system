import re
from llm.hf import LLMModel
from FlagEmbedding import FlagReranker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS



model_name = "../embedding_model/bge-large-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction="为这个句子生成表示以用于检索相关文章:"
            )

reranker = FlagReranker('../embedding_model/bge-reranker-v2-m3', use_fp16=True)  # Setting use_fp16 to True speeds up


def initialize_retriever(model_path_embedding, vectorstore_path, device='cpu', k=3):
    # 初始化 Embedding 模型
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_path_embedding,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},
        query_instruction="为这个句子生成表示以用于检索相关文章:"
    )

    # 从本地加载 FAISS 索引
    db = FAISS.load_local(
        vectorstore_path,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

    # 创建检索器
    retriever = db.as_retriever(search_kwargs={"k": k})

    return retriever


def initialize_reranker(model_path_reranker):
    # 初始化 Reranker 模型
    reranker = FlagReranker(model_path_reranker, use_fp16=True)
    return reranker


def get_knowledge_scope(question, llm):
    prompt = (
        f"如果想回答问题：{question}，需要哪些知识，给出最需要的3条，注意以列表的形式返回，具体格式如下："
        "[<需要的知识1>, <需要的知识2>, <需要的知识3>]"
    )
    response = llm.generate_response(prompt)
    # return list(map(lambda x: x.strip(),response[1:-1].split(",")))
    return list(map(lambda x: re.sub(r'[<>]', '', x.strip()), response[1:-1].split(',')))


def generate_answer(context_list, question, llm):
    context = "\n".join(context_list)
    prompt3_zh = (
        "你是一个TuGraph-DB问答任务的助手。 "
        "对于下面的问题，请基于提供的相关信息进行回答。 "
        "最多只用三句话回答，确保回答简明扼要。"
        "\n\n"
        f"问题: {question}\n"
        f"相关信息: {context}\n"
        f"回答:"
    )
    return llm.generate_response(prompt3_zh)


def run_rag(question, retriever):
    # 检索相关文档
    docs = retriever.invoke(question)
    contexts_list = [doc.page_content for doc in docs]
    query_passages = [[question, passage] for passage in contexts_list]
    return contexts_list, query_passages


def merge_and_deduplicate(question, contexts_list, bm25_contexts_list):
    # Convert lists to sets of tuples and then merge
    merged_contexts = set(map(tuple, contexts_list)) | set(map(tuple, bm25_contexts_list))

    # Convert the set of tuples back to a list of strings
    merged_contexts_list = ["".join(context) for context in merged_contexts]

    # Generate the merged query_passages
    merged_query_passages = [[question, context] for context in merged_contexts_list]

    return merged_contexts_list, merged_query_passages



def merge_and_deduplicate_RRP(contexts_list1, scores_list1, contexts_list2, scores_list2):
    # 创建一个字典来存储去重后的上下文及其得分
    contexts_with_scores = {}

    # 将第一个上下文列表及其得分添加到字典中
    for context, score in zip(contexts_list1, scores_list1):
        contexts_with_scores[context] = max(contexts_with_scores.get(context, float('-inf')), score)

    # 将第二个上下文列表及其得分添加到字典中
    for context, score in zip(contexts_list2, scores_list2):
        contexts_with_scores[context] = max(contexts_with_scores.get(context, float('-inf')), score)

    # 提取去重后的上下文和得分
    final_contexts_list = list(contexts_with_scores.keys())
    # 根据 final_contexts_list 生成 final_scores_list
    final_scores_list = [contexts_with_scores[context] for context in final_contexts_list]

    return final_contexts_list, final_scores_list




# def run_rerank(query_passages, contexts_list, reranker):
#     scores = reranker.compute_score(query_passages, normalize=True)
#     # 根据得分对段落进行排序（从高到低）
#     sorted_passages = sorted(zip(scores, contexts_list), key=lambda x: x[0], reverse=True)
#
#     # 提取rerank后的段落
#     sorted_contexts = "\n".join([passage for score, passage in sorted_passages])
#     return sorted_contexts


def run_rerank(query_passages, contexts_list, reranker):
    scores = reranker.compute_score(query_passages, normalize=True)

    # 根据得分对段落及其得分进行排序（从高到低）
    sorted_passages_with_scores = sorted(zip(scores, contexts_list), key=lambda x: x[0], reverse=True)

    # 分离得分和上下文
    sorted_scores, sorted_contexts = zip(*sorted_passages_with_scores)

    # 返回排序后的上下文字符串和得分列表
    return sorted_contexts, sorted_scores


if __name__ == '__main__':
    llm = LLMModel(model_name="zhipu")
    llm_rag = LLMModel(model_name="zhipu")
    # 示例问题
    question = "在使用LIMIT子句时，如果查询数据库中前两个人的名字，返回的名字是什么？"
    knowledge = get_knowledge_scope(question, llm)
    print(knowledge)
    print("*"*150)
    retriever = initialize_retriever(
        model_path_embedding="../embedding_model/bge-large-zh-v1.5",
        vectorstore_path="../vectorstore_1024_512_by_doc_with_summary_split_bge_embedding/summary",
        device="cpu",
        k=3
    )

    # 获取相关的文档id，每个知识检索出3个相关的文档
    doc_indices = [
        item.metadata["tugraph-db-source"]
        for kw in knowledge
        for item in retriever.invoke(kw)
    ]

    # 去重
    doc_indices = set(doc_indices)

    print(doc_indices)
    print("*"*150)

    context = []
    for doc in doc_indices:
        safe_doc = doc.replace('/', '_')
        retriever = initialize_retriever(
            model_path_embedding="../embedding_model/bge-large-zh-v1.5",
            vectorstore_path=f"../vectorstore_1024_512_by_doc_with_summary_split_bge_embedding/{safe_doc}",
            device="cpu",
            k=3
        )
        context_list, _ = run_rag(question, retriever)
        context.extend(context_list)

    print(context)
    print("*"*150)
    response = generate_answer(context, question, llm_rag)
    print(response)