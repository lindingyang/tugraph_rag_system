from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# # 加载模型
# model_name = "../embedding_model/bge-large-zh-v1.5"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': True}
# embedding = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs,
#     query_instruction="为这个句子生成表示以用于检索相关文章:"
# )

model = SentenceTransformer('../embedding_model/bge-large-zh-v1.5')

# 读取JSONL文件
data = []
with open('../dataset/sft_data/filtered_question.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 提取问题列表
sentences = df['question'].tolist()

# 计算每个问题的向量表示
embeddings = model.encode(sentences, normalize_embeddings=True)  # 使用模型进行嵌入

# 计算余弦相似度矩阵
similarity_matrix = cosine_similarity(embeddings)

# 去重
to_remove = set()
for i in range(len(similarity_matrix)):
    if i not in to_remove:  # 如果该行未被标记为要移除
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > 0.9:  # 相似度高于0.9
                to_remove.add(j)

# 保留未被标记的行
df_unique = df.drop(index=list(to_remove))

# 将去重后的数据保存回JSONL文件
with open('../dataset/sft_data/filtered_question_not_similar.jsonl', 'w', encoding='utf-8') as file:
    for _, row in df_unique.iterrows():
        json.dump(row.to_dict(), file, ensure_ascii=False)
        file.write('\n')

print(f"去重完成！原始行数: {len(df)}，去重后行数: {len(df_unique)}")
