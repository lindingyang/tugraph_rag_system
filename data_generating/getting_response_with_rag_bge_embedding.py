import json
from tqdm import tqdm
from rag.rag_without_langchain_bge_embedding import run_rag, initialize_retriever, initialize_reranker, generate_answer, run_rerank
from llm.hf import LLMModel


def process_jsonl(input_file, output_file, model):
    llm = LLMModel(model_name=model)
    retriever = initialize_retriever(
        model_path_embedding="../embedding_model/bge-large-zh-v1.5",
        vectorstore_path="../vectorstore_1024_512_bge_embedding",
        device="cpu",
        k=3
    )
    reranker = initialize_reranker(model_path_reranker="../embedding_model/bge-reranker-v2-m3")

    # 计算输入文件中的总行数，以便显示进度条
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'a', encoding='utf-8') as outfile:
        # 使用 tqdm 包装 enumerate 以显示进度
        for line_number, line in tqdm(enumerate(infile), total=total_lines, desc="Processing JSONL"):
            data = json.loads(line)
            question = data.get("input_field")
            number = data.get("id")
            # class_type = data.get("class")

            # 获取检索内容
            contexts_list, query_passages = run_rag(question, retriever)
            sorted_contexts, _ = run_rerank(query_passages, contexts_list, reranker, k=3)
            # 获取输出
            response = generate_answer(sorted_contexts, question, llm)

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

            # 实时写入新的JSONL文件
            outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            outfile.flush()  # 确保数据被写入文件


if __name__ == '__main__':
    model = "zhipu"  # openai或zhipu
    input_jsonl_file = '../filtered_question_not_similar.jsonl'  # 输入文件名
    output_jsonl_file = f'../filtered_data_with_response_{model}_rag.jsonl'  # 输出文件名
    process_jsonl(input_jsonl_file, output_jsonl_file, model)
