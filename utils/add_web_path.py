import os
import json

# 文件夹路径
folder_path = "../dataset/md_files"

# 基础 URL
base_url = "https://raw.githubusercontent.com/theshi-1128/rag_dataset/refs/heads/master/"

# 获取文件夹中所有文件的文件名
file_names = os.listdir(folder_path)

# 过滤出文件名（排除文件夹）
file_names = [f for f in file_names if os.path.isfile(os.path.join(folder_path, f))]

# 生成 web_paths 列表
web_paths = [base_url + file_name for file_name in file_names]

# 创建字典
data = {"web_paths": web_paths}

# 保存到 web.json 文件
with open("web.json", "w") as json_file:
    json.dump(data, json_file, indent=2)

print("web.json 已生成。")

