from typing import Dict, List, Tuple


def reciprocal_rank_fusion(merged_contexts: List[str], merged_scores: List[float], k: int = 5) -> Tuple[List[str], List[float]]:
    """
    Perform Reciprocal Rank Fusion (RRF) on the provided search results.

    Args:
        merged_contexts (List[str]): A list of document contents.
        merged_scores (List[float]): A list of corresponding scores for the documents.
        k (int): A constant added to the rank for score adjustment (default is 60).

    Returns:
        Tuple[Dict[str, float], List[str], List[float]]: A tuple containing:
            - A dictionary of document contents with their fused scores, sorted by score.
            - A list of documents (sorted).
            - A list of corresponding fused scores (sorted).
    """
    fused_scores = {}

    # 创建一个字典，将上下文和得分配对
    contexts_with_scores = {context: score for context, score in zip(merged_contexts, merged_scores)}

    for rank, (doc, score) in enumerate(sorted(contexts_with_scores.items(), key=lambda x: x[1], reverse=True)):
        if doc not in fused_scores:
            fused_scores[doc] = 0
        # 更新融合得分
        fused_scores[doc] += 1 / (rank + k)

    # 按照融合得分降序排序
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # 提取排序后的文档和得分
    sorted_contexts = [doc for doc, score in sorted_results]
    sorted_scores = [score for doc, score in sorted_results]

    return sorted_contexts, sorted_scores


def rrp(question, merged_contexts_list, contexts_list, bm25_contexts_list, k: int,
                                 semantic_weight: float = 0.8, bm25_weight: float = 0.2):
    """
    该方法接收合并后的文档列表（merged_contexts_list），计算每个文档的得分，并返回前 k 个得分最高的文档内容。

    参数：
        merged_contexts_list (list): 合并后的文档内容列表，包括语义搜索和 BM25 搜索的结果。
        contexts_list (list): 语义搜索结果的文档内容列表。
        bm25_contexts_list (list): BM25 搜索结果的文档内容列表。
        k (int): 返回的文档数量。
        semantic_weight (float): 语义搜索的加权系数，默认为 0.8。
        bm25_weight (float): BM25 搜索的加权系数，默认为 0.2。

    返回：
        list: 返回前 k 个得分最高的文档内容。
    """
    # 用于存储文档的得分
    content_to_score = []

    # 遍历 merged_contexts_list 来计算每个文档的得分
    for content in merged_contexts_list:
        score = 0

        # 如果当前文档来自语义搜索，则加权得分
        if content in contexts_list:
            score += semantic_weight * (1 / (contexts_list.index(content) + 1))  # 基于语义搜索中的排名

        # 如果当前文档来自 BM25 搜索，则加权得分
        if content in bm25_contexts_list:
            score += bm25_weight * (1 / (bm25_contexts_list.index(content) + 1))  # 基于 BM25 搜索中的排名

        # 将文档内容及其得分存储在列表中
        content_to_score.append({'content': content, 'score': score})

    # 按照得分对文档进行排序，得分高的排在前面
    sorted_results = sorted(content_to_score, key=lambda x: x['score'], reverse=True)

    # 返回前 k 个得分最高的文档内容
    final_results = sorted_results[:k]
    final_content_list = [result['content'] for result in final_results]
    final_score_list = [result['score'] for result in final_results]

    final_query_passages = [[question, passage] for passage in final_content_list]

    return final_query_passages, final_content_list, final_score_list



if __name__ == '__main__':
    final_sorted_contexts = ["你啊后", "哈哈哈", "cnm", "dsafwfa"]

    final_sorted_scores = [0.42, 0.11, 0.93, 0.78]

    sorted_contexts, sorted_scores = reciprocal_rank_fusion(final_sorted_contexts, final_sorted_scores, k=5)
    print(sorted_contexts)
    print(sorted_scores)
