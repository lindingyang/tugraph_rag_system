from llm.api import get_gpt_response
import json
from tqdm import tqdm


prompt_1 = "You are an assistant for TuGraph-DB question-generating tasks. Generate a unique TuGraph-DB question from the following category.\n" \
         "基于单个文档或代码段的问题: A question that tests basic retrieval and understanding from a single tugraph-db-source. Example: \"如何在TuGraph-DB中创建一个新的图实例？\"\n" \
         "Generate one unique question that aligns with these requirements in Chinese. Ensure the question is distinct from any previously generated question.\n" \
         "你的输出需要遵循以下格式：\n" \
         "问题："


prompt_2 = "You are an assistant for TuGraph-DB question-generating tasks. Generate a unique TuGraph-DB question from the following category.\n" \
         "需要综合多个文档或代码段的问题: A question that requires information synthesis and reasoning across multiple sources. Example: \"TuGraph-DB在处理大规模图数据时有哪些优化策略？\"\n" \
         "Generate one unique question that aligns with these requirements in Chinese. Ensure the question is distinct from any previously generated question.\n" \
         "你的输出需要遵循以下格式：\n" \
         "问题："


prompt_3 = "You are an assistant for TuGraph-DB question-generating tasks. Generate a unique TuGraph-DB question from the following category.\n" \
         "无法回答的问题: A question that tests the system's ability to recognize unsupported features or undocumented content in TuGraph-DB. The system should respond with appropriate feedback, such as \"暂不支持该功能。\"\n" \
         "Generate one unique question that aligns with these requirements in Chinese. Ensure the question is distinct from any previously generated question.\n" \
         "你的输出需要遵循以下格式：\n" \
         "问题："


prompt_4 = "You are an assistant for TuGraph-DB question-generating tasks. Generate a unique TuGraph-DB question from the following category.\n" \
         "代码理解类任务: A question that tests understanding of TuGraph-DB-specific syntax or GQL (Graph Query Language). Example: \"请解释以下GQL查询的功能：[具体GQL查询].\"\n" \
         "Generate one unique question that aligns with these requirements in Chinese. Ensure the question is distinct from any previously generated question.\n" \
         "你的输出需要遵循以下格式：\n" \
         "问题："


prompt_5 = "You are an assistant for TuGraph-DB question-generating tasks. Generate a unique TuGraph-DB question.\n" \
         "Here's several examples: \n" \
         "机器性能指标中的“memory”是什么？\n" \
         "构造FieldSpec时需要哪些参数？\n" \
         "ANTLR4支持生成哪些目标语言的解析器？\n" \
         "边索引支持查询加速么？\n" \
         "`FieldData` 类中的函数 `IsReal()` 是用来查询什么类型的数据？\n" \
         "你的输出需要遵循以下格式：\n" \
         "问题："


# 定义提示词的列表
prompts = [
    prompt_1,
    prompt_2,
    prompt_3,
    prompt_4,
    prompt_5
]


# 生成问题的函数
def generate_questions(prompt, category_id):
    response = get_gpt_response(prompt)
    question = response[3:response.find("\n", 3)].strip()
    return {
        "class": category_id,
        "question": question
    }


# 保存生成的问题到JSONL文件
with open('../generated_questions.jsonl', 'a', encoding='utf-8') as file:
    for i in tqdm(range(600), desc="Generating questions"):  # 加上tqdm进度条
        for idx, prompt in enumerate(prompts, start=1):
            question_data = generate_questions(prompt, idx)

            # 创建json数据结构，确保id的唯一性
            json_data = {
                "id": i * 5 + idx + 2000,
                **question_data  # 展开字典
            }
            file.write(json.dumps(json_data, ensure_ascii=False) + '\n')

            # 输出生成的问题
            print("*" * 150)
            print(f"类别：{idx}")
            print(f"问题：{question_data['question']}")
