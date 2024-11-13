import re
import requests
import os
import json
from tqdm import tqdm  # 导入tqdm库
from llm.api import get_gpt_response


TABLE_PROMPT = """
请将以下表格转化为简洁清晰的文字描述。描述中包括：
1. 描述表格的列名。
2. 逐行描述表格中的数据内容。
3. 表格整体内容的概要总结。

输入表格：
{table_content}

请按照以下格式输出表格的文字描述:
表格内容描述:（表格内容的文字描述）
"""


def extract_tables_from_markdown(content):
    """
    从Markdown内容中提取表格并返回表格内容以及表格所在的行号
    """
    table_pattern = r'(\|.+\|[\r\n]+)(\|[-\|]+\|[\r\n]+)?((\|.+\|[\r\n]+)+)'  # 匹配Markdown表格的结构

    # 查找所有匹配的表格
    tables = re.finditer(table_pattern, content)

    extracted_tables = []
    table_positions = []

    # 处理每个找到的表格
    for match in tables:
        # 提取表格内容
        table_content = match.group(0).strip()

        # 提取表格的起始位置（行号）
        start_pos = match.start()
        start_line = content.count('\n', 0, start_pos) + 1  # 计算表格的起始行号

        extracted_tables.append(table_content)
        table_positions.append(start_line)

    return extracted_tables, table_positions


def fetch_and_extract_tables_from_github(web_paths, output_dir):
    """
    从GitHub文件提取表格，生成文字描述，并添加到表格后面
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用tqdm显示进度条
    for i, url in tqdm(enumerate(web_paths), total=len(web_paths), desc="Processing GitHub files"):
        if url:
            try:
                # 获取GitHub raw文件内容
                response = requests.get(url)
                response.raise_for_status()  # 检查请求是否成功

                # 获取文件内容并提取表格
                content = response.text
                tables, positions = extract_tables_from_markdown(content)

                # 如果提取到表格，生成描述并插入到表格后面
                if tables:
                    updated_content = content
                    for table, position in zip(tables, positions):
                        # 调用get_response方法生成描述
                        description = get_gpt_response(TABLE_PROMPT.format(table_content=table))
                        print(f"description: {description}")
                        print("-"*150)
                        # 找到表格的结束位置（基于行号）
                        table_end_pos = updated_content.find(table) + len(table)

                        # 将描述插入到表格后面
                        updated_content = updated_content[:table_end_pos] + '\n' + description + updated_content[
                                                                                                 table_end_pos:]

                    # 输出修改后的文件
                    output_file = os.path.join(output_dir, f"updated_file_{i + 1}.md")
                    with open(output_file, 'w', encoding='utf-8') as file:
                        file.write(updated_content)

                    # 输出当前文件包含的表格数量和描述
                    print(f"File {url} contains {len(tables)} table(s) with descriptions added. Saved to {output_file}")
                else:
                    # 如果没有表格，保存原始文件内容
                    output_file = os.path.join(output_dir, f"updated_file_{i + 1}.md")
                    with open(output_file, 'w', encoding='utf-8') as file:
                        file.write(content)

                    print(f"No tables found in {url}. Saved original content to {output_file}")

            except requests.RequestException as e:
                print(f"Error fetching {url}: {e}")


def load_web_paths_from_json(json_file):
    # 从JSON文件加载web_paths
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data.get('web_paths', [])
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []


if __name__ == '__main__':
    json_file = '../dataset/web_paths_split_full_manual_refine_final.json'  # 存放web_paths的JSON文件路径
    output_dir = '../extracted_tables'  # 提取的表格将保存到这个目录

    # 从JSON文件加载web_paths
    web_paths = load_web_paths_from_json(json_file)

    # 提取GitHub文件中的表格并保存
    fetch_and_extract_tables_from_github(web_paths, output_dir)
