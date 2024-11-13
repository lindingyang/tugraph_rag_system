from llm.api import get_gpt_response
import json
from tqdm import tqdm


def optimize_questions(input_file, output_file, model="openai"):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()  # 读取所有行，以便获得总行数
        for line in tqdm(lines, desc="优化问题", unit="问题"):
            data = json.loads(line)
            question = data['input_field']
            # 构建优化提示词
            full_fill_prompt = (
                f"原始问题：{question}\n"
                "任务：根据原始问题，将原始问题优化为关于 TuGraph-DB 的完整问题。保持问题简洁，明确 TuGraph-DB 的相关性。\n"
                "优化后的问题:"
            )
            # full_fill_prompt = (
            #         f"原始问题：{question}\n"
            #         "相关背景信息：{background_info}\n"
            #         "任务：根据原始问题和相关背景信息，将原始问题优化为关于 TuGraph-DB 的完整问题。保持问题简洁，明确 TuGraph-DB 的相关性。\n"
            #         "优化后的问题:"
            # )
            # 调用模型获取优化后的问题
            if model == "openai":
                optimized_question = get_gpt_response(full_fill_prompt)
            else:
                raise ValueError(f"Unknown model type: {model}")
            final_optimized_question = optimized_question.strip()  # 去掉多余空白
            print("final_optimized_question:", final_optimized_question)
            # 创建新的字典以保存优化后的问题
            optimized_data = {
                'id': data['id'],
                'input_field': final_optimized_question
            }
            # 写入到输出文件
            outfile.write(json.dumps(optimized_data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    input_path = '../dataset/test1.jsonl'
    # 使用函数进行优化
    optimize_questions(input_path, '../dataset/optimized_question_openai.jsonl', model="openai")
    optimize_questions(input_path, '../dataset/optimized_question_zhipu.jsonl', model="zhipu")
    optimize_questions(input_path, '../dataset/optimized_question_sft_zhipu.jsonl', model="sft_zhipu")
