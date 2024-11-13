# CCF_BDCI_2024_RAG_System

2024 CCF BDCI 赛题一 「TuGraph for AI」RAG在智能问答场景中的落地 A榜第15名思路分享。

RAG知识库数据集详见: [here](https://github.com/theshi-1128/tugraph_rag_dataset)

## 方案总结

- Embedding Model: text-embedding-3-large (openai)
- Reranker: bge-reranker-v2-m3
- 基座模型: GPT-4o-mini(使用1w条领域数据集进行SFT)
- SFT超参数设置: epoch=1, lr=1e-5, bs=16
- 知识库数据来源: Tugraph-db官方使用文档, Tugraph-Analytics官方使用文档, Tugraph官方博客
- 知识库优化策略: 文档拆分 + 格式转化(标准markdown格式) + 添加中文注释(代码和表格) + 图片转文字 + 表格转文字
- 检索策略: 混合检索（embedding + bm25 + rank_fusion）+ contextual检索(为每个chunk引入上下文信息)
- 生成策略: 设计提示词, 使输出内容简洁准确，减少冗余信息
- RAG超参数设置: chunk_size=1024, overlap=512, top_k=3

