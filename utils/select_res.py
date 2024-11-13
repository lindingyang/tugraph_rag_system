import json
from tqdm import tqdm
from llm.api import get_gpt_response


def select_best_response(question, res1, res2, res3):
    prompt = (
        "请分析给定的问题和三个不同的回复。请按照以下步骤进行分析，从中选择最准确、最符合问题意图的最佳回复:\n\n"
        "1、判断问题质量：检查问题是否提供了足够的信息来回答（例如，是否缺少必要的信息或具体细节）。\n"
        "2、选择最佳回复：如果问题不完整或信息不足，选择表示无法回答或说明问题缺失的回复。如果问题完整，选择最准确、最符合问题意图的回复。\n\n"
        f"问题：{question}\n"
        f"回复1：{res1}\n"
        f"回复2：{res2}\n"
        f"回复3：{res3}\n\n"
        "请根据问题分析选择最佳回复，只输出最佳回复的内容，不要包含“回复1”,“回复2”或“回复3”标签。"
        "请按照以下格式输出最终选择的回复：\n"
        "**分析过程**：\n"
        "**最佳回复**：(选择的回复内容)"
    )

    # 获取 GPT 的响应
    res = get_gpt_response(prompt)
    return res


def process_questions(question_file, res_file_1, res_file_2, res_file_3, output_file):
    # 读取问题文件
    with open(question_file, 'r', encoding='utf-8') as q_file:
        questions = [json.loads(line.strip()) for line in q_file]

    # 读取回答文件
    with open(res_file_1, 'r', encoding='utf-8') as r1_file:
        res1_list = [json.loads(line.strip()) for line in r1_file]

    with open(res_file_2, 'r', encoding='utf-8') as r2_file:
        res2_list = [json.loads(line.strip()) for line in r2_file]

    with open(res_file_3, 'r', encoding='utf-8') as r3_file:
        res3_list = [json.loads(line.strip()) for line in r3_file]

    # 初始化计数器
    same_res1_count = 0
    same_res2_count = 0
    same_res3_count = 0

    # 打开输出文件准备写入
    with open(output_file, 'w', encoding='utf-8') as output_f:
        # 假设问题和回答的顺序一致，并使用 tqdm 来显示进度条
        for i, question_data in tqdm(enumerate(questions), total=len(questions), desc="Processing questions"):
            number = question_data.get("id")
            question = question_data.get("input_field")
            res1 = res1_list[i].get("output_field")
            res2 = res2_list[i].get("output_field")
            res3 = res3_list[i].get("output_field")

            # 选择最佳回复
            result = select_best_response(question, res1, res2, res3)
            best_reply_start = result.find("**最佳回复**：") + len("**最佳回复**：")
            best_reply = result[best_reply_start:].strip()

            # 判断与 res1 和 res2 的相同内容
            if best_reply == res1:
                same_res1_count += 1
            elif best_reply == res2:
                same_res2_count += 1
            elif best_reply == res3:
                same_res3_count += 1

            # 将结果写入到输出文件
            output_data = {"id": number, "output_field": best_reply}
            output_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

            # 打印结果（可选）
            print(f"问题: {question}")
            print(f"回复1: {res1}")
            print(f"回复2: {res2}")
            print(f"回复3: {res3}")
            print(f"最佳回复: {best_reply}")
            print("-" * 40)
            print(f"与回复1完全相同的最佳回复数: {same_res1_count}")
            print(f"与回复2完全相同的最佳回复数: {same_res2_count}")
            print(f"与回复3完全相同的最佳回复数: {same_res3_count}")

    # 打印统计信息
    print(f"与回复1完全相同的最佳回复数: {same_res1_count}")
    print(f"与回复2完全相同的最佳回复数: {same_res2_count}")
    print(f"与回复3完全相同的最佳回复数: {same_res3_count}")



if __name__ == '__main__':
    # 设置文件路径
    question_file = '../dataset/test1.jsonl'
    res_file_1 = '../dataset/answer_best.jsonl'
    res_file_2 = '../dataset/answer_zh3_1024_512_3_bge-reranker-v2-m3_glm-4-flash:499254306::auoms6xs_by_doc_mixed_retriever_split_full.jsonl'
    res_file_3 = '../dataset/answer_zh3_1024_512_3_bge-reranker-v2-m3_glm-4-flash:499254306::auoms6xs_by_doc_with_summary_mixed_retriever_split_full.jsonl'
    output_file = '../dataset/best_replies.jsonl'  # 输出的新 JSONL 文件

    # 处理所有问题并保存最佳回复
    process_questions(question_file, res_file_1, res_file_2, res_file_3, output_file)
