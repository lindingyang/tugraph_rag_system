from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from FlagEmbedding import FlagReranker
from tqdm import tqdm
import json
from rag.bm25_with_langchain import create_bm25_retriever
from llm.hf import LLMModel


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


def generate_answer(context_list, question, answer, llm):
    context = "\n".join(context_list)
    optimize_rag_answer_prompt = (
        "你是一个TuGraph-DB问答任务的助手。"
        "对于下面的问题，请根据检索到的相关信息，验证当前回答是否准确且符合问题意图。"
        "如果当前回答不符合检索到的内容或不够准确，请参考相关信息优化该回答，确保其简洁且准确地回答问题。\n"
        "最多只用三句话回答，确保优化后的回答简明扼要。"
        "\n\n"
        f"用户问题：{question}\n\n"
        f"相关信息: {context}\n\n"
        f"当前回答：{answer}\n\n"
        "优化后的回答:"
    )
    return llm.generate_response(optimize_rag_answer_prompt)


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


def run_rerank(query_passages, contexts_list, reranker):
    scores = reranker.compute_score(query_passages, normalize=True)

    # 根据得分对段落及其得分进行排序（从高到低）
    sorted_passages_with_scores = sorted(zip(scores, contexts_list), key=lambda x: x[0], reverse=True)

    # 分离得分和上下文
    sorted_scores, sorted_contexts = zip(*sorted_passages_with_scores)

    # 返回排序后的上下文字符串和得分列表
    return sorted_contexts, sorted_scores


def optimize_questions(input_question_file, input_answer_file, output_file, model="openai"):
    num_same = 0
    llm = LLMModel(model_name=model)
    retriever = initialize_retriever(
        model_path_embedding="../embedding_model/bge-large-zh-v1.5",
        vectorstore_path="../vectorstore_1024_512_split_full_bge_embedding",
        device="cpu",
        k=3
    )
    bm25_retriever = create_bm25_retriever(file_path='../bm25_storage_split_full_1024_512/processed_chunks.json', k=3)
    reranker = initialize_reranker(model_path_reranker="../embedding_model/bge-reranker-v2-m3")
    # 计算输入文件中的总行数，以便显示进度条
    total_lines = sum(1 for _ in open(input_question_file, 'r', encoding='utf-8'))

    with open(input_question_file, 'r', encoding='utf-8') as qfile, \
         open(input_answer_file, 'r', encoding='utf-8') as afile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        # 使用 zip 同时迭代两个文件
        for qline, aline in tqdm(zip(qfile, afile), total=total_lines, desc="Processing JSONL"):
            qdata = json.loads(qline.strip())  # 去除末尾换行符并解析 JSON
            adata = json.loads(aline.strip())  # 去除末尾换行符并解析 JSON
            question = qdata['input_field']
            answer = adata['output_field']

            # 获取检索内容
            contexts_list, query_passages = run_rag(question, retriever)
            bm25_contexts_list, bm25_query_passages = run_rag(question, bm25_retriever)
            merged_contexts_list, merged_query_passages = merge_and_deduplicate(question, contexts_list,
                                                                                bm25_contexts_list)

            rerank_contexts, rerank_scores = run_rerank(merged_query_passages, merged_contexts_list, reranker)
            # 取前k个
            top_k_contexts = rerank_contexts[:3]

            # 获取输出
            opt_answer = generate_answer(top_k_contexts, question, answer, llm)
            print("question: ", question)
            print("original_answer: ", answer)
            print("opt_answer: ", opt_answer)
            print("-"*100)

            if opt_answer == answer:
                num_same = num_same + 1
            print("优化后答案与原始答案相同数量: ", num_same)
            print("-"*100)
            # 创建新的字典以保存优化后的问题
            optimized_data = {
                'id': adata['id'],
                'output_field': opt_answer
            }
            # 写入到输出文件
            outfile.write(json.dumps(optimized_data, ensure_ascii=False) + '\n')
        print("优化后答案与原始答案相同数量: ", num_same)


if __name__ == '__main__':
    input_question_path = '../dataset/test1.jsonl'
    input_answer_path = '../dataset/answer_best.jsonl'
    output_file = '../dataset/optimized_answer_q.jsonl'
    optimize_questions(input_question_path, input_answer_path, output_file, model="zhipu")
