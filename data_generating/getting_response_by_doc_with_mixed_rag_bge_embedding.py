import json
from tqdm import tqdm
from rag.rag_by_doc import run_rag, initialize_retriever, initialize_reranker, \
    generate_answer, merge_and_deduplicate, run_rerank, get_knowledge_scope, merge_and_deduplicate_RRP
from rag.bm25_with_langchain import create_bm25_retriever
from llm.hf import LLMModel
from utils.RRP import reciprocal_rank_fusion

#
# # 先分别用rerank模型，然后汇总去重后，进行RRP
# def process_jsonl(input_file, output_file, model):
#     llm = LLMModel(model_name=model)
#     retriever = initialize_retriever(
#         model_path_embedding="../embedding_model/bge-large-zh-v1.5",
#         vectorstore_path="../vectorstore_1024_512_bge_embedding",
#         device="cpu",
#         k=3
#     )
#     bm25_retriever = create_bm25_retriever(file_path='../bm25_storage_1024_512/processed_chunks.json', k=3)
#     reranker = initialize_reranker(model_path_reranker="../embedding_model/bge-reranker-v2-m3")
#
#     # 计算输入文件中的总行数，以便显示进度条
#     total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
#
#     with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'a', encoding='utf-8') as outfile:
#         # 使用 tqdm 包装 enumerate 以显示进度
#         for line_number, line in tqdm(enumerate(infile), total=total_lines, desc="Processing JSONL"):
#             data = json.loads(line)
#             question = data.get("input_field")
#             number = data.get("id")
#             # class_type = data.get("class")
#
#             # 获取检索内容
#             contexts_list, query_passages = run_rag(question, retriever)
#             bm25_contexts_list, bm25_query_passages = run_rag(question, bm25_retriever)
#
#             rerank_contexts, rerank_scores = run_rerank(query_passages, contexts_list, reranker)
#             rerank_bm25_contexts, rerank_bm25_scores = run_rerank(bm25_query_passages, bm25_contexts_list, reranker)
#
#             merged_contexts_list, merged_scores_list = merge_and_deduplicate_RRP(rerank_contexts,
#                                                                                  rerank_scores,
#                                                                                  rerank_bm25_contexts,
#                                                                                  rerank_bm25_scores)
#
#             final_sorted_contexts, sorted_scores = reciprocal_rank_fusion(merged_contexts_list, merged_scores_list)
#             # 取前k个
#             top_k_contexts = final_sorted_contexts[:3]
#             # 获取输出
#             response = generate_answer(top_k_contexts, question, llm)
#
#             print("*" * 150)
#             # print(f"类别：{class_type}")
#             print(f"问题：{question}")
#             print(f"回复：{response}")
#
#             # 创建新字典
#             output_data = {
#                 "id": number,
#                 # "class": class_type,
#                 # "question": question,
#                 "output_field": response
#             }
#
#             # 实时写入新的JSONL文件
#             outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")
#             outfile.flush()  # 确保数据被写入文件


# 直接汇总去重，然后最后用rerank模型
def process_jsonl(input_file, output_file, model):
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
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'a', encoding='utf-8') as outfile:
        # 使用 tqdm 包装 enumerate 以显示进度
        for line_number, line in tqdm(enumerate(infile), total=total_lines, desc="Processing JSONL"):
            data = json.loads(line)
            # question = data.get("question")

            # class_type = data.get("class")

            question = data.get("input_field")
            number = data.get("id")

            knowledge = get_knowledge_scope(question, llm)
            # 获取相关的文档id，每个知识检索出3个相关的文档
            doc_indices = [
                item.metadata["tugraph-db-source"]
                for kw in knowledge
                for item in retriever.invoke(kw)
            ]
            # 去重
            doc_indices = set(doc_indices)

            # 获取检索内容
            contexts_list = []
            for doc in doc_indices:
                safe_doc = doc.replace('/', '_')
                retriever = initialize_retriever(
                    model_path_embedding="../embedding_model/bge-large-zh-v1.5",
                    vectorstore_path=f"../vectorstore_1024_512_by_doc_split_full_bge_embedding/{safe_doc}",
                    device="cpu",
                    k=3
                )
                context_kw, _ = run_rag(question, retriever)
                contexts_list.extend(context_kw)

            # contexts_list, query_passages = run_rag(question, retriever)
            bm25_contexts_list, bm25_query_passages = run_rag(question, bm25_retriever)
            merged_contexts_list, merged_query_passages = merge_and_deduplicate(question, contexts_list,
                                                                                bm25_contexts_list)

            rerank_contexts, rerank_scores = run_rerank(merged_query_passages, merged_contexts_list, reranker)
            # 取前k个
            top_k_contexts = rerank_contexts[:5]
            # print(top_k_contexts[0])
            # print("*"*150)
            # print(top_k_contexts[1])
            # print("*"*150)
            # print(top_k_contexts[2])
            # print("*"*150)
            # 获取输出
            response = generate_answer(top_k_contexts, question, llm)

            print("*" * 150)
            # print(f"类别：{class_type}")
            print(f"问题：{question}")
            print(f"回复：{response}")

            # 创建新字典
            output_data = {
                "id": number,
                # "class": class_type,
                # "question": question,
                # "response": response,
                "output_field": response
            }

            # 实时写入新的JSONL文件
            outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            outfile.flush()  # 确保数据被写入文件


if __name__ == '__main__':
    model = "zhipu"  # openai或zhipu
    input_jsonl_file = '../dataset/sft_data/1.jsonl'  # 输入文件名
    output_jsonl_file = f'../dataset/sft_data/2.jsonl'  # 输出文件名
    process_jsonl(input_jsonl_file, output_jsonl_file, model)
