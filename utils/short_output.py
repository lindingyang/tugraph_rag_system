from llm.api import get_gpt_response
import json
from tqdm import tqdm


PROMPT = """
仔细阅读以下的Query和Response，并对Response进行简化。请确保简化后的Response的内容完整且准确回答Query。你可以删去冗余信息，只保留直接相关且有助于回答Query的内容。可以对主语进行省略或删减。

<示例1>
**Query:**
启动参数中cleanup_dir指定的目录用于执行什么操作？

**Response:**
cleanup_dir指定的目录用于在执行完成之后被清理的操作。

**Cleared Response:**
用于在执行完成之后被清理的操作。
</示例1>

<示例2>
**Query:**
安装部署TuGraph外存配置的最低和建议分别是多少？

**Response:**
最低配置：CPU 4核，内存4GB，外存100GB；建议配置：CPU 64核，内存512GB，外存2TB NVMe SSD。

**Cleared Response:**
最低配置为100GB，建议配置为2TB NVMe SSD。
</示例2>

<示例3>
**Query:**
SetFields函数的第一个版本中，field_value_strings参数的数据类型是什么？

**Response:**
SetFields函数的第一个版本中，field_value_strings参数的数据类型是std::vector。

**Cleared Response:**
数据类型是std::vector。
</示例3>

**Query:**
{QUERY}

**Response:**
{RESPONSE}

请提供一个简化的版本，将答案精简为核心内容。请按照以下格式输出:
Cleared Response: (简化后的Response)
"""


def process_jsonl(input_file_1, input_file_2, output_file):
    with open(input_file_1, 'r', encoding='utf-8') as infile1, \
            open(input_file_2, 'r', encoding='utf-8') as infile2, \
            open(output_file, 'w', encoding='utf-8') as outfile:
        for index, (line1, line2) in enumerate(tqdm(zip(infile1, infile2), desc="Processing", total=250, unit="line"), start=1):
            # 解析输入文件的内容
            input_data = json.loads(line1)
            output_data = json.loads(line2)

            query = input_data.get('input_field', '')
            response = output_data.get('output_field', '')

            # 根据查询和响应生成prompt
            prompt = PROMPT.format(QUERY=query, RESPONSE=response)

            # 获取GPT的响应
            gpt_response = get_gpt_response(prompt)
            clear_response = gpt_response.split("Cleared Response:")[1].strip()
            print(clear_response)
            print('*' * 150)

            # 准备输出数据
            result_data = {
                "id": f"TEST1-{index}",
                "output_field": clear_response
            }

            # 写入新的jsonl文件
            outfile.write(json.dumps(result_data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    input_file_1_path = '../dataset/test1.jsonl'
    input_file_2_path = '../dataset/answer_zh3_new_1024_512_3_bge-reranker-v2-m3_gpt-4o-mini-2024-07-18:personal:rag:ASeifONX_contextual_mixed_retriever_split_full_manual_refined_openai_embedding_table2text_final.jsonl'
    output_file_path = '../dataset/answer_zh3_new_1024_512_3_bge-reranker-v2-m3_gpt-4o-mini-2024-07-18:personal:rag:ASeifONX_contextual_mixed_retriever_split_full_manual_refined_openai_embedding_table2text_final(精简gpt4o).jsonl'
    # 调用处理函数
    process_jsonl(input_file_1_path, input_file_2_path, output_file_path)

    print(f"处理完成，输出结果已写入{output_file_path}")


