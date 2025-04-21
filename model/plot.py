import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import torch
import numpy as np
from sklearn.decomposition import PCA
import torch.nn as nn

from models import NeuralCF
from dataset import map_graph_nodes

def plot_score_distribution(pos_score, neg_score, title="Score Distribution"):
    """
    Plot the distribution of positive and negative scores.

    Parameters:
    - pos_score: List or array of positive scores.
    - neg_score: List or array of negative scores.
    - title: Title of the plot.
    """
    sklearn_auc = roc_auc_score([1]*len(pos_score) + [0]*len(neg_score), pos_score + neg_score)
    print(f"ROC AUC Score: {sklearn_auc:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(pos_score, bins=50, alpha=0.5, label='Positive Score', color='blue')
    plt.hist(neg_score, bins=50, alpha=0.5, label='Negative Score', color='red')
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.savefig("./figure/plot_score_distribution_output3.png")
            
def all_score_visualization(test_loader, edges_index, edges_weights):
    all_scores = []
    
    model = NeuralCF(num_users=155, num_items=6496, emb_size=128)
    model.load_state_dict(torch.load("./model/checkpoint/best_model.pth"))
    model.eval()
    
    with torch.no_grad():
        for user, item, label in test_loader:
            preds = model(user, item, edges_index, edges_weights)
            all_scores.extend(preds.cpu().numpy())
            
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=50, alpha=0.7, color='green')
    plt.title("All Scores Distribution")
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("./figure/all_score_visualization_output.png")