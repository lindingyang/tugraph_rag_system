from data_generating.getting_response_with_mixed_rag_openai_embedding import process_jsonl


if __name__ == '__main__':
    model = "openai"  # openai或zhipu或qwen
    input_jsonl_file = '../dataset/test1.jsonl'  # 输入文件名
    output_jsonl_file = f'../dataset/answer_zh3_new_1024_512_3_bge-reranker-v2-m3_gpt-4o-mini-2024-07-18:personal:rag:ASeifONX_contextual_mixed_retriever_split_full_manual_refined_openai_embedding_table2text_final.jsonl'  # 输出文件名
    process_jsonl(input_jsonl_file, output_jsonl_file, model)
