import json

# 输入jsonl文件路径
input_file = '../dataset/val.jsonl'
# 输出jsonl文件路径
output_file = '../dataset/val.jsonl'

# 读取并处理jsonl文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        # 仅保留question和response字段
        new_data = {
            'question': data['input_field'],
            'response': data['output_field']
        }
        # 写入处理后的内容
        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

print("处理完成，已保存为output.jsonl文件。")
