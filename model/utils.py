import torch

def precision_recall_at_k_torch(pred_scores, ground_truth_ids, k):
    """
    pred_scores: 1D torch tensor of predicted scores for all items (e.g. [6498])
    ground_truth_ids: list or set of ground truth item indices (e.g. [3, 20, 401])
    k: top-k value
    """

    # 상위 k개 인덱스 추출 (예측 점수 기준)
    topk_scores, topk_indices = torch.topk(pred_scores, k)
    
    topk_set = set(topk_indices.tolist())
    ground_truth_set = set(ground_truth_ids)

    num_relevant_in_top_k = len(topk_set & ground_truth_set)

    precision = num_relevant_in_top_k / k
    recall = num_relevant_in_top_k / len(ground_truth_set) if ground_truth_set else 0.0

    return precision, recall

def evaluate_precision_recall_k(model, test_loader, edges_index, edges_type, edges_weights, num_items, k=5):
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    edges_index = edges_index.to(device)
    edges_weights = edges_weights.to(device)
    edges_type = edges_type.to(device).long()

    precision_list = []
    recall_list = []

    with torch.no_grad():
        for user, pos, _ in test_loader:
            user = user.to(device)
            pos = pos.to(device)

            # 전체 아이템에 대한 점수 예측
            all_items = torch.arange(num_items).to(device)
            for u in user:
                u_repeat = u.repeat(num_items)
                scores = model(u_repeat, all_items, edges_index, edges_type, edges_weights)

                # 정답 아이템 ID 목록
                gt_items = pos[user == u].tolist()

                # precision@k, recall@k 계산
                prec, rec = precision_recall_at_k_torch(scores, gt_items, k)
                precision_list.append(prec)
                recall_list.append(rec)

    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
