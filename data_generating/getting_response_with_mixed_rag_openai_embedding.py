from rag.rag_without_langchain_openai_embedding import run_rag
import json
from tqdm import tqdm
from rag.rag_without_langchain_openai_embedding import run_rag, initialize_retriever, initialize_reranker, \
    generate_answer, merge_and_deduplicate, run_rerank, merge_and_deduplicate_RRP
from rag.bm25_with_langchain import create_bm25_retriever
from llm.hf import LLMModel
from utils.RRP import reciprocal_rank_fusion, rrp


# 直接汇总去重，然后最后用rerank模型
def process_jsonl(input_file, output_file, model):
    llm = LLMModel(model_name=model)
    retriever = initialize_retriever(
        vectorstore_path="../contextual_vectorstore_1024_512_split_full_manual_refine_openai_embedding_table2text_final",
        k=3
    )
    bm25_retriever = create_bm25_retriever(
        file_path='../contextual_bm25_storage_split_full_manual_refine_1024_512_table2text_final/processed_chunks.json', k=3)
    reranker = initialize_reranker(model_path_reranker="../embedding_model/bge-reranker-v2-m3")

    # 计算输入文件中的总行数，以便显示进度条
    # 计算输入文件中的总行数
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'a', encoding='utf-8') as outfile:
        # 使用 tqdm 包装 enumerate 以显示进度
        for line_number, line in tqdm(enumerate(infile), total=total_lines, desc="Processing JSONL"):
            data = json.loads(line)
            question = data.get("input_field")
            number = data.get("id")
            # class_type = data.get("class")
            # question = data["messages"][0]["content"]

            # 获取检索内容
            contexts_list, query_passages = run_rag(question, retriever)
            bm25_contexts_list, bm25_query_passages = run_rag(question, bm25_retriever)
            merged_contexts_list, merged_query_passages = merge_and_deduplicate(question, contexts_list,
                                                                                bm25_contexts_list)

            rerank_contexts, rerank_scores = run_rerank(merged_query_passages, merged_contexts_list, reranker)
            # 取前k个
            top_k_contexts = rerank_contexts[:3]
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
                "output_field": response
            }
            # output_data = {
            #     "messages": [
            #         {
            #             "role": "user",
            #             "content": question  # 将 question 填充到 user 的 content 中
            #         },
            #         {
            #             "role": "assistant",
            #             "content": response  # 将 response 填充到 assistant 的 content 中
            #         }
            #     ]
            # }
            # 实时写入新的JSONL文件
            outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            outfile.flush()  # 确保数据被写入文件



# # RRP+用rerank模型
# def process_jsonl(input_file, output_file, model):
#     llm = LLMModel(model_name=model)
#     retriever = initialize_retriever(
#         vectorstore_path="../contextual_vectorstore_1024_512_split_full_manual_refine_openai_embedding",
#         k=6
#     )
#     bm25_retriever = create_bm25_retriever(file_path='../contextual_bm25_storage_split_full_manual_refine_1024_512/processed_chunks.json', k=6)
#     reranker = initialize_reranker(model_path_reranker="../embedding_model/bge-reranker-v2-m3")
#     # 计算输入文件中的总行数，以便显示进度条
#     total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
#
#     with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'a', encoding='utf-8') as outfile:
#         # 使用 tqdm 包装 enumerate 以显示进度
#         for line_number, line in tqdm(enumerate(infile), total=total_lines, desc="Processing JSONL"):
#             data = json.loads(line)
#             # question = data.get("question")
#
#             question = data.get("input_field")
#             number = data.get("id")
#
#             # class_type = data.get("class")
#
#             # 获取检索内容
#             contexts_list, query_passages = run_rag(question, retriever)
#             bm25_contexts_list, bm25_query_passages = run_rag(question, bm25_retriever)
#             merged_contexts_list, merged_query_passages = merge_and_deduplicate(question, contexts_list, bm25_contexts_list)
#             # 调用rrp方法获取前 6 个得分最高的文档
#             rrp_query_passages, rrp_content_list, rrp_score_list = rrp(question, merged_contexts_list, contexts_list, bm25_contexts_list, k=6, semantic_weight=0.8, bm25_weight=0.2)
#             # rerank
#             rerank_contexts, rerank_scores = run_rerank(rrp_query_passages, rrp_content_list, reranker)
#             # 取前k个
#             top_k_contexts = rerank_contexts[:3]
#             top_k_scores = rerank_scores[:3]
#
#             # print("rrp_content_list:\n", rrp_content_list)
#             # print("*"*150)
#             # print("rrp_score_list:", rrp_score_list)
#             # print("*"*150)
#             # print("rerank_contexts:\n", rerank_contexts)
#             # print("*"*150)
#             # print("rerank_scores:", rerank_scores)
#             # print("*"*150)
#             # print("top_3_contexts:\n", top_k_contexts)
#             # print("*"*150)
#             # print("top_3_scores:", top_k_scores)
#             # print("*"*150)
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
#                 # "response": response,
#                 "output_field": response
#             }
#
#             # 实时写入新的JSONL文件
#             outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")
#             outfile.flush()  # 确保数据被写入文件


if __name__ == '__main__':
    model = "zhipu"  # openai或zhipu
    input_jsonl_file = '../dataset/sft_data/api_glm4_flash_sft_data.jsonl'  # 输入文件名
    output_jsonl_file = f'../dataset/api_glm4_flash_sft_data_{model}_best.jsonl'  # 输出文件名
    process_jsonl(input_jsonl_file, output_jsonl_file, model)
