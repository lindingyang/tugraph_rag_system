import pdfplumber
from markdownify import markdownify as md


def pdf_to_markdown(pdf_path, md_output_path):
    # 打开 PDF 文件
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""

        # 遍历每一页，提取文本
        for page in pdf.pages:
            # 提取每一页的文本
            page_text = page.extract_text()
            if page_text:
                # 保留换行符 '\n'，避免去除自然换行
                all_text += page_text + "\n\n"

        # 使用 markdownify 转换为 markdown
        # 如果不需要复杂的 HTML 转换，直接保持文本也可以
        markdown_content = md(all_text, strip=["a", "img"])  # 去掉不需要的标签

        # 保存为 markdown 文件
        with open(md_output_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_content)

    print(f"Markdown file saved at: {md_output_path}")




if __name__ == '__main__':
    pdf_path = "../dataset/document.pdf"  # PDF 文件路径
    md_output_path = "../dataset/document.md"  # 输出的 Markdown 文件路径
    pdf_to_markdown(pdf_path, md_output_path)
