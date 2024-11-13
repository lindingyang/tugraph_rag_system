import json

# 读取原始的 JSONL 文件并进行转换
input_file = "../dataset/sft_data/filtered_data_not_familiar_with_response_zhipu_rag.jsonl"
output_file = "../dataset/sft_data/api_glm4_flash_sft_data_not_familiar.jsonl"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        data = json.loads(line)  # 读取每行的 JSON 数据
        question = data.get("question", "")  # 获取 question 字段
        response = data.get("response", "")  # 获取 response 字段

        # 转换成所需的格式
        new_format = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ]
        }

        # 将转换后的数据写入新的 JSONL 文件
        outfile.write(json.dumps(new_format, ensure_ascii=False) + "\n")

print("转换完成！")
