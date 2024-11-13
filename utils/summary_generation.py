import json
import requests
from tqdm import tqdm
from llm.api import get_gpt_response

# 读取 web_paths
with open("../dataset/web_paths_split_full_best.json", "r", encoding="utf-8") as f:
    data = json.load(f)

web_paths = data["web_paths"]

# 打开输出文件，准备以 JSONL 格式写入摘要
with open("../dataset/summary.jsonl", "a", encoding="utf-8") as f:
    for i, url in tqdm(enumerate(web_paths), total=len(web_paths)):
        # 获取网页内容
        text = requests.get(url).text

        # 构建生成摘要的提示词
        prompt = (
            "请简明扼要地概述下面这段文本的主要内容，确保摘要包括文档的核心主题、目的或关键观点。"
            "注意！你的总结应该准确反映文档的核心信息，避免遗漏重要细节或加入无关内容。"
            "请确保摘要既全面又简洁，避免冗长。"
            "Text:"
            f"{text}"
            "Output:"
        )

        # 获取 GPT 的摘要响应
        response = get_gpt_response(prompt)

        # 打印摘要（可选）
        print(f"摘要 {i + 1}: ", response)

        # 构建要保存的 JSON 数据
        output_data = {"summary": response.strip()}  # 去掉多余的空格

        # 将摘要写入 JSONL 文件
        f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
        f.flush()
