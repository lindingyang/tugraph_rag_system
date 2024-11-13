from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from llm.api import get_gpt_response
from FlagEmbedding import FlagReranker
from rag.rag_without_langchain_bge_embedding import run_rag



#对每个子问题进行rag，检索相应文档，并得到各自的回答。
# 将所有子问题检索到的文档合并，并进行重排序，作为最终rag检索到的文档。
# 将对子问题的回答汇总，结合特定的提示词，指导大模型根据这些子问题的答案，给出原问题的更准确的答案。


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



# 从本地加载 FAISS 索引
db = FAISS.load_local(
    '../vectorstore_1024_512_bge_embedding',
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})


def extract_sub_questions(text):
    # Split the text into lines and filter out empty lines
    lines = list(filter(None, text.split('\n')))

    # Initialize variables to hold the sub-questions
    sub1, sub2, sub3 = "", "", ""

    # Extract sub-questions based on identifiers "1.", "2.", "3."
    for line in lines:
        if line.startswith("1."):
            sub1 = line[3:].strip()  # Remove the identifier and any leading/trailing whitespace
        elif line.startswith("2."):
            sub2 = line[3:].strip()
        elif line.startswith("3."):
            sub3 = line[3:].strip()

    # Return the sub-questions
    return sub1, sub2, sub3


def generate_sub_question(question, model_type="openai"):
    # prompt = (
    #     "你是一名乐于助人的助手，你的任务是将输入问题分解为一组循序渐进的子问题，每个子问题的回答都能够帮助最终解决原始问题。"
    #     "例子：\n"
    #     f"请根据以下问题生成多个有逻辑顺序的子问题：Cython是如何导入与Olap相关的模块和图数据库模块的？"
    #     "输出（3个子问题）：\n"
    #     "1.Cython 是什么?\n"
    #     "2.Cython是如何导入Olap相关模块的？\n"
    #     "3.Cython是如何导入图数据库模块的？\n"
    #     f"请根据以下问题生成多个有逻辑顺序的子问题：{question}"
    #     "输出（3个子问题）：\n"
    #     "1.\n"
    #     "2.\n"
    #     "3."
    # )
    prompt = (
        "你是一名乐于助人的助手，负责生成与输入问题相关的多个子问题。"
        "你的任务是将输入问题分解为一组可以单独回答的子问题。"
        "例子：\n"
        f"请生成与以下问题相关的多个子问题：Cython是如何导入与Olap相关的模块和图数据库模块的？"
        "输出（3个子问题）：\n"
        "1.Cython 是什么?\n"
        "2.Cython是如何导入Olap相关模块的？\n"
        "3.Cython是如何导入图数据库模块的？\n"
        f"请生成与以下问题相关的多个子问题：{question}"
        "输出（3个子问题）：\n"
        "1.\n"
        "2.\n"
        "3."
    )
    if model_type == "openai":
        return get_gpt_response(prompt)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_answer(context, question, model_type="openai"):
    prompt3_zh = (
        "你是一个TuGraph-DB问答任务的助手。 "
        "对于下面的问题，请基于提供的相关信息进行回答。 "
        "最多只用三句话回答，确保回答简明扼要。"
        "\n\n"
        f"问题: {question}\n"
        f"相关信息: {context}\n"
        f"回答:"
    )
    if model_type == "openai":
        return get_gpt_response(prompt3_zh)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_final_answer(context, question, sub1, sub2, sub3, answer1, answer2, answer3, model_type="openai"):
    prompt3_zh = (
        "你是一个TuGraph-DB问答任务的助手。 "
        "为了确保回答的准确性，我将问题拆分成三个子问题。 "
        "对于下面的问题，请基于提供的相关信息与子问题问答对进行回答。 "
        "你的最终回答最多只用三句话，确保最终回答简明扼要。"
        "\n\n"
        f"问题: {question}\n"
        f"相关信息: {context}\n"
        f"子问题1: {sub1}\n"
        f"回答1: {answer1}\n"
        f"子问题2: {sub2}\n"
        f"回答2: {answer2}\n"
        f"子问题3: {sub3}\n"
        f"回答3: {answer3}\n"
        f"最终回答:"
    )
    if model_type == "openai":
        return get_gpt_response(prompt3_zh)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 封装 RAG 过程的函数
def run_sub_rag(sub_question, model="openai"):
    # 检索相关文档
    docs = retriever.invoke(sub_question)
    contexts_list = [doc.page_content for doc in docs]

    query_passages = [[sub_question, passage] for passage in contexts_list]
    scores = reranker.compute_score(query_passages, normalize=True)
    # 根据得分对段落进行排序（从高到低）
    sorted_passages = sorted(zip(scores, contexts_list), key=lambda x: x[0], reverse=True)

    # 提取rerank后的段落
    sorted_contexts = "\n".join([passage for score, passage in sorted_passages])

    # 生成回答
    answer = generate_answer(sorted_contexts, sub_question, model)

    return contexts_list, answer


def run_final_rag(sub1, sub2, sub3, question, model="openai"):
    contexts_list1, answer1 = run_sub_rag(sub1)
    contexts_list2, answer2 = run_sub_rag(sub2)
    contexts_list3, answer3 = run_sub_rag(sub3)

    contexts_list = contexts_list1 + contexts_list2 + contexts_list3

    query_passages = [[question, passage] for passage in contexts_list]
    scores = reranker.compute_score(query_passages, normalize=True)
    # 根据得分对段落进行排序（从高到低）并只保留前三名
    sorted_passages = sorted(zip(scores, contexts_list), key=lambda x: x[0], reverse=True)[:3]

    # 提取rerank后的前三名段落
    sorted_contexts = "\n".join([passage for score, passage in sorted_passages])

    # 生成回答
    answer = generate_final_answer(sorted_contexts, question, sub1, sub2, sub3, answer1, answer2, answer3, model)

    return sorted_contexts, answer


if __name__ == '__main__':
    # 示例问题
    question = "当中止一个正在执行的任务时，应该使用哪种HTTP请求方法？"
    print("question:", question)

    sub_question = generate_sub_question(question, "zhipu")  # openai或zhipu
    sub1, sub2, sub3 = extract_sub_questions(sub_question)
    print("sub1:", sub1)
    print("sub2:", sub2)
    print("sub3:", sub3)

    context_raw, answer_raw = run_rag(question, "zhipu")  # openai或zhipu

    context_final, answer_final = run_final_rag(sub1, sub2, sub3, question, "zhipu")
    print("answer_raw:", answer_raw)
    print("context_raw:", context_raw)

    print("answer_final:", answer_final)
    print("context_final:", context_final)
