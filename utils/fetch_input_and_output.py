import json
from tqdm import tqdm


def process_jsonl(input_file_1, input_file_2, output_file):
    # 读取 output_file 内容并构建查询（query）到 response 的映射
    existing_data = {}

    # 读取 output_file，提取查询（query）并存储对应的响应（response）
    with open(output_file, 'r', encoding='utf-8') as outfile:
        for line in outfile:
            try:
                data = json.loads(line)
                query = data['messages'][0]['content']
                response = data['messages'][1]['content']
                existing_data[query] = response  # 保存 output_file 中查询到的响应
            except json.JSONDecodeError:
                continue

    # 读取 input_file_1 和 input_file_2，处理并替换 output_file 中的内容
    with open(input_file_1, 'r', encoding='utf-8') as infile1, \
            open(input_file_2, 'r', encoding='utf-8') as infile2:

        # 读取 output_file 的内容并保持原有内容
        outfile_lines = []
        with open(output_file, 'r', encoding='utf-8') as existing_file:
            outfile_lines = existing_file.readlines()

        # 遍历 input_file_1 和 input_file_2，只处理 input_file_1 中的查询
        for line1, line2 in tqdm(zip(infile1, infile2), desc="Processing", total=250, unit="line"):
            # 解析 input_file_1 和 input_file_2 的内容
            input_data = json.loads(line1)
            output_data = json.loads(line2)

            query = input_data.get('input_field', '')
            new_response = output_data.get('output_field', '')

            # 检查 query 是否在 output_file 中已有，如果有，替换 response
            if query in existing_data:
                # 在 output_file 中找到该 query 并替换响应
                for i, existing_line in enumerate(outfile_lines):
                    try:
                        data = json.loads(existing_line)
                        if data['messages'][0]['content'] == query:
                            # 替换 response
                            data['messages'][1]['content'] = new_response
                            outfile_lines[i] = json.dumps(data, ensure_ascii=False) + '\n'
                            break  # 替换后跳出循环
                    except json.JSONDecodeError:
                        continue

        # 将修改后的内容写回 output_file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.writelines(outfile_lines)


if __name__ == '__main__':
    input_file_1_path = '../dataset/test1.jsonl'
    input_file_2_path = '../dataset/manual_answer3（精简manual_answer1的全部回答）.jsonl'
    output_file_path = '../dataset/sft_data/api_glm4_flash_sft_data_zhipu_best(73分).jsonl'
    # 调用处理函数
    process_jsonl(input_file_1_path, input_file_2_path, output_file_path)

    print(f"处理完成，输出结果已写入{output_file_path}")


