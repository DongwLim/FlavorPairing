import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
import numpy as np
from sklearn.decomposition import PCA
import random
from models import NeuralCF
from dataset import map_graph_nodes
from utils import smoothed_scaled_score

def plot_score_distribution(pos_score, neg_score, title="Score Distribution"):
    """
    Plot the distribution of positive and negative scores.

    Parameters:
    - pos_score: List or array of positive scores.
    - neg_score: List or array of negative scores.
    - title: Title of the plot.
    """
    true_labels = np.array([1] * len(pos_score) + [0] * len(neg_score))
    all_scores = np.array(pos_score + neg_score)
    
    sklearn_auc = roc_auc_score(true_labels, all_scores)
    print(f"ROC AUC Score: {sklearn_auc:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(pos_score, bins=50, alpha=0.5, label='Positive Score', color='blue')
    plt.hist(neg_score, bins=50, alpha=0.5, label='Negative Score', color='red')
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.savefig("./figure/gcn_plot_1.png")
    
def test_visualization(model, test_loader, edges_index, edges_weights, edges_type):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    edges_index = edges_index.to(device)
    edges_weights = edges_weights.to(device)
    
    model.eval()
    pos_scores = []
    neg_scores = []
    
    with torch.no_grad():
        for user, pos, neg in test_loader:
            user = user.long()
            pos = pos.long()
            neg = neg.long()

            user, pos, neg = user.to(device), pos.to(device), neg.to(device)

            pos_output = model(user, pos, edges_index, edges_type, edges_weights)
            neg_output = model(user, neg, edges_index, edges_type, edges_weights)
            
            pos_scores.extend(pos_output.cpu().numpy())
            neg_scores.extend(neg_output.cpu().numpy())
    
    plot_score_distribution(pos_scores, neg_scores, title="Score Distribution")
            
def all_score_visualization(edges_index, edges_weights, edges_type):
    all_scores = []
    
    model = NeuralCF(num_users=155, num_items=6498, emb_size=128)
    model.load_state_dict(torch.load("./model/checkpoint/best_model.pth"))
    model.eval()

    mapping = map_graph_nodes()
    
    lid_to_idx = mapping['liquor']
    iid_to_idx = mapping['ingredient']

    lid = []
    iid = []

    for i in range(7101):
        if i in lid_to_idx:
            lid.append(lid_to_idx[i])
        elif i in iid_to_idx:
            iid.append(iid_to_idx[i])
        else:
            continue

    sample_iid = random.sample(iid, 500)  
    sample_lid = random.sample(lid, 10)

    with torch.no_grad():
        for i in tqdm(sample_iid, desc="Processing ingredients"):
            for l in sample_lid:
                preds = model(l, i, edges_index, edges_type, edges_weights)
                all_scores.append(preds.item())
            
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=50, alpha=0.7, color='green')
    plt.title("All Scores Distribution")
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("./figure/all_score_visualization_output.png")
    
def visualize_embeddings(before_emb, after_emb, user_ids, item_ids, title='Embedding Comparison'):
    # CPU 이동
    before_np = before_emb.detach().cpu().numpy()
    after_np = after_emb.detach().cpu().numpy()
    
    # PCA로 2차원 변환
    pca = PCA(n_components=2)
    combined = pca.fit_transform(np.vstack([before_np, after_np]))

    num_nodes = before_np.shape[0]
    before_pca = combined[:num_nodes]
    after_pca = combined[num_nodes:]

    # 시각화
    plt.figure(figsize=(10, 5))
    
    # Before
    plt.subplot(1, 2, 1)
    plt.title("Before Training")
    plt.scatter(before_pca[user_ids, 0], before_pca[user_ids, 1], label='Users', alpha=0.6)
    plt.scatter(before_pca[item_ids, 0], before_pca[item_ids, 1], label='Items', alpha=0.6)
    plt.legend()

    # After
    plt.subplot(1, 2, 2)
    plt.title("After Training")
    plt.scatter(after_pca[user_ids, 0], after_pca[user_ids, 1], label='Users', alpha=0.6)
    plt.scatter(after_pca[item_ids, 0], after_pca[item_ids, 1], label='Items', alpha=0.6)
    plt.legend()

    plt.savefig("./figure/embedding_visualization.png")