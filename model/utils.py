import numpy as np
import torch

def smoothed_scaled_score(score, pivot=-0.2, alpha=0.1):
    centered = score - pivot
    return torch.tanh(alpha * centered)

def sigmoid_centered(score, pivot=-2, scale=1.5):
    return 2 / (1 + np.exp(-scale * (score - pivot))) - 1

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


def evaluate_precision_recall_k_multi(model, test_loader,num_items, ks=[10, 20, 50]):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_items = torch.arange(num_items, device=device)

    precision_dict = {k: [] for k in ks}
    recall_dict = {k: [] for k in ks}

    with torch.no_grad():
        for users, pos_items, _ in test_loader:
            users = users.to(device)
            pos_items = pos_items.to(device)

            for u in users.unique():
                u_idx = (users == u)
                gt_items = pos_items[u_idx].tolist()

                if not gt_items:
                    continue

                u_repeat = u.repeat(num_items)
                scores = model(u_repeat, all_items)

                for k in ks:
                    prec, rec = precision_recall_at_k_torch(scores, gt_items, k)
                    precision_dict[k].append(prec)
                    recall_dict[k].append(rec)

    for k in ks:
        avg_prec = float(torch.tensor(precision_dict[k]).mean()) if precision_dict[k] else 0.0
        avg_rec = float(torch.tensor(recall_dict[k]).mean()) if recall_dict[k] else 0.0
        print(f"Precision@{k}: {avg_prec:.4f}")
        print(f"Recall@{k}: {avg_rec:.4f}")