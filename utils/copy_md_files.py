import os
import shutil


def copy_markdown_files(src_folder, dest_folder):
    # 如果目标文件夹不存在，创建它
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 计数器初始化
    md_file_count = 0

    # 遍历源文件夹及其子文件夹
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.md'):  # 只处理 Markdown 文件
                # 获取文件的完整路径
                src_file = os.path.join(root, file)
                # 目标文件的路径
                dest_file = os.path.join(dest_folder, file)

                # 如果目标文件已存在，修改目标文件名以避免覆盖
                if os.path.exists(dest_file):
                    base, ext = os.path.splitext(file)
                    i = 1
                    # 创建一个新的文件名，直到没有同名文件为止
                    while os.path.exists(os.path.join(dest_folder, f"{base}_{i}{ext}")):
                        i += 1
                    dest_file = os.path.join(dest_folder, f"{base}_{i}{ext}")

                # 复制文件到目标文件夹
                shutil.copy(src_file, dest_file)
                print(f'文件 {file} 已复制到 {dest_folder}')

                # 计数器加一
                md_file_count += 1

    print(f'总共复制了 {md_file_count} 个 Markdown 文件')





if __name__ == '__main__':
    src_folder = '../source'  # 替换为源文件夹路径
    dest_folder = '../dataset/md_files'  # 替换为目标文件夹路径
    copy_markdown_files(src_folder, dest_folder)
