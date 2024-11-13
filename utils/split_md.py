import os


def split_markdown(input_file, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用 \n# 来识别一级标题（以 # 开头的行）
    sections = content.split('\n# ')

    file_count = 0  # 初始化计数器

    for section in sections:
        # 确保每个部分都以一级标题开头
        if section.strip():
            # 获取一级标题
            title, *body = section.split('\n', 1)
            # 构建子文件名
            title = title.strip().replace('#', '').strip()  # 去掉标题符号和空格
            title = title.replace(' ', '_')  # 用下划线替换空格
            title = title[:50]  # 限制文件名长度
            output_file = os.path.join(output_dir, f"{title}.md")

            # 写入子文件
            with open(output_file, 'w', encoding='utf-8') as out_f:
                out_f.write('# ' + section)  # 写入一级标题和内容

            print(f"已创建子文件: {output_file}")
            file_count += 1  # 每创建一个子文件，计数器加一

    print(f"总共创建了 {file_count} 个子文件。")  # 输出总数


if __name__ == '__main__':
    input_markdown_file = '../md/README.md'  # 替换为您的输入文件路径
    output_directory = '../md'   # 替换为您希望输出的目录
    split_markdown(input_markdown_file, output_directory)
