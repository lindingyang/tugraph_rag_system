import json


task_type = "你是一个TuGraph-DB问答任务的助手。\n" \
            "你需要对用户的TuGraph-DB问题进行详细的分析与思考。最后，你需要提供一个准确清晰的答案。"


qwen_conversation = "<|im_start|>" \
                    "\n" \
                    "user: {USER_PROMPT}" \
                    "\n" \
                    "<|im_end|>" \


output_format = "请给出你对该TuGraph-DB问题的回复。\n" \
                "如果该问题包含TuGraph-DB中不支持的特性，你应该给出适当的反馈，例如\"暂不支持该功能。\""




sft_format = f"""{task_type}\n\n{qwen_conversation}\n\n{output_format}"""

print(sft_format)

#
#
#
# def jsonl_to_jsonl(input_jsonl_file_path, output_jsonl_file_path):
#     # 打开输入JSONL文件并读取每一行
#     with open(input_jsonl_file_path, 'r', encoding='utf-8') as input_jsonl_file:
#         lines = input_jsonl_file.readlines()
#
#     # 打开输出JSONL文件
#     with open(output_jsonl_file_path, 'w', encoding='utf-8') as output_jsonl_file:
#         for line in lines:
#             # 解析每一行为JSON对象
#             json_object = json.loads(line.strip())
#             category = json_object['class']
#             # 获取并处理需要的字段
#             query = json_object['question']
#             sft_prompt = sft_format.format(USER_PROMPT=query)
#             response = json_object['response']
#
#             # 构建新的JSON对象
#             new_json_object = {
#                 "category": category,
#                 "question": sft_prompt,
#                 "response": response
#             }
#
#             # 写入到新的JSONL文件
#             output_jsonl_file.write(json.dumps(new_json_object, ensure_ascii=False) + '\n')
#
#
# # 使用方法示例
# jsonl_file_path = '../filtered_data_with_response_zhipu.jsonl'
# output_jsonl_file_path = '../qwen_sft_data_zhipu.jsonl'
# jsonl_to_jsonl(jsonl_file_path, output_jsonl_file_path)
#
