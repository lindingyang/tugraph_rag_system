import pandas as pd
import json

# 读取JSONL文件
data = []
with open('../dataset/sft_data/filtered_data_not_familiar_with_response_zhipu_rag.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 对 'question' 字段进行去重操作
df_unique = df.drop_duplicates(subset=['question'])

# 将去重后的数据保存回JSONL文件
with open('../dataset/sft_data/filtered_data_not_familiar_with_response_zhipu_rag1.jsonl', 'w', encoding='utf-8') as file:
    for _, row in df_unique.iterrows():
        json.dump(row.to_dict(), file, ensure_ascii=False)
        file.write('\n')

print(f"去重完成！原始行数: {len(df)}，去重后行数: {len(df_unique)}")
