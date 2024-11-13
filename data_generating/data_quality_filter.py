import pandas as pd
import json
from llm.api import get_gpt_response
from tqdm import tqdm


# 读取 JSONL 文件
data = []
with open('../dataset/sft_data/filtered_question_not_similar.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 处理每个问题
questions_to_keep = []

# 使用 tqdm 显示进度条
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="处理问题"):
    question = row['question']
    prompt = f"请评估以下问题的质量。判断它是否是一个完整、清晰且易于回答的问题。请只回答“是”或“否”。\n\n问题：{question}"

    # 使用定义好的函数获取模型评估
    answer = get_gpt_response(prompt).strip()
    print("Answer: ", answer)

    if answer == "是":
        questions_to_keep.append(row)  # 保留该问题

# 创建新的 DataFrame 并保存回 JSONL 文件
df_filtered = pd.DataFrame(questions_to_keep)
with open('../dataset/sft_data/filtered_question_not_similar_high_quality.jsonl', 'w', encoding='utf-8') as file:
    for _, row in df_filtered.iterrows():
        json.dump(row.to_dict(), file, ensure_ascii=False)
        file.write('\n')

print(f"筛选完成！保留的问题数量: {len(df_filtered)}")