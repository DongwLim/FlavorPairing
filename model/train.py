import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random

from dataset import map_graph_nodes, edges_index, BPRDataset
from plot import test_visualization, all_score_visualization
from models import NeuralCF
from utils import evaluate_precision_recall_k_multi

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

def train_model(model, train_loader, val_loader, edges_index, edges_weights, edges_type, num_epochs=10, lr=0.0005, weight_decay=1e-5):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)
    
    model.to(device)
    
    edges_index = edges_index.to(device)
    edges_weights = edges_weights.to(device)
    edges_type = edges_type.to(device).long()
    model.train()

    #criterion = bpr_loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    best_model = None
    best_val_loss = float('inf')

    topk = 5

    print(f"Training on {device}")
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for user, pos, neg in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            user = user.long()   
            pos = pos.long()   
            neg = neg.long()

            user, pos, neg = user.to(device), pos.to(device), neg.to(device)

            optimizer.zero_grad()

            pos_output = model(user, pos, edges_index, edges_type, edges_weights)

            num_neg_candidates = 10
            neg_candidates = torch.randint(0, 6498, (user.size(0), num_neg_candidates), device=device)

            user_expand = user.unsqueeze(1).expand_as(neg_candidates)
            user_flat = user_expand.reshape(-1)
            neg_flat = neg_candidates.reshape(-1)

            neg_scores = model(user_flat, neg_flat, edges_index, edges_type, edges_weights)
            neg_scores = neg_scores.view(user.size(0), num_neg_candidates)

            hard_neg_scores, hard_neg_indices = torch.topk(neg_scores, k=topk, dim=1)

            random_idx = torch.randint(0, topk, (user.size(0),), device=device)
            hard_neg = neg_candidates[torch.arange(user.size(0)), hard_neg_indices[torch.arange(user.size(0)), random_idx]]

            neg_output = model(user, hard_neg, edges_index, edges_type, edges_weights)
            
            loss = bpr_loss(pos_output, neg_output)
            #loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * pos.size(0)
            # Calculate ranking accuracy for BPR
            correct += (pos_output > neg_output).sum().item()  # Count correct rankings
            total += pos.size(0)  # Total number of positive samples

        avg_loss = total_loss / total
        acc = correct / total
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for user, pos, neg in val_loader:
                user = user.long()
                pos = pos.long()
                neg = neg.long()

                user, pos, neg = user.to(device), pos.to(device), neg.to(device)

                pos_output = model(user, pos, edges_index, edges_type, edges_weights)
                neg_output = model(user, neg, edges_index, edges_type, edges_weights)
                loss = bpr_loss(pos_output, neg_output)
                
                # loss = criterion(output, label)
                
                val_loss += loss.item() * pos.size(0)
                # Calculate ranking accuracy for BPR
                val_correct += (pos_output > neg_output).sum().item()  # Count correct rankings
                val_total += pos.size(0)  # Total number of positive samples
        
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"[Validation] Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f}")
        #torch.save(model.state_dict(), f"./model/checkpoint/epoch_{epoch}.pth")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            print(f"Best model saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}")
            
        # Check Early Stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            torch.save(best_model, "./model/checkpoint/best_model.pth")
            break

if __name__ == "__main__":
    #set_seed()

    print("Loading data...")
    mapping = map_graph_nodes()
    
    lid_to_idx = mapping['liquor']
    iid_to_idx = mapping['ingredient']

    #print(lid_to_idx)
    print("Loading graph data...")
    
    edge_type_map ={
        'liqr-ingr': 0,
        'ingr-ingr': 1,
        'liqr-liqr': 1,
        'ingr-fcomp': 2,
        'ingr-dcomp': 2
    }
    
    edges_indexes, edges_weights, edges_type = edges_index(edge_type_map)
    
    print("Loading dataset...")
    positive_pairs = pd.read_csv("./liquor_good_ingredients.csv")
    positive_pairs = positive_pairs[['liquor_id', 'ingredient_id']]

    negative_pairs = pd.read_csv("./liquor_bad_ingredients.csv")
    negative_pairs = negative_pairs[['liquor_id', 'ingredient_id']]

    print("Mapping liquor and ingredient IDs to indices...")
    positive_pairs['liquor_id'] = positive_pairs['liquor_id'].map(lid_to_idx)
    positive_pairs['ingredient_id'] = positive_pairs['ingredient_id'].map(iid_to_idx)

    negative_pairs['liquor_id'] = negative_pairs['liquor_id'].map(lid_to_idx)
    negative_pairs['ingredient_id'] = negative_pairs['ingredient_id'].map(iid_to_idx)

    """
    num_users = 155 # Number of unique liquor IDs
    num_items = 6498 # Number of unique ingredient IDs
    """
    
    print("Creating dataset...")
    
    train_val_pairs, test_pairs = train_test_split(positive_pairs, test_size=0.2, random_state=42)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.2, random_state=42)
    
    train_dataset = BPRDataset(positive_pairs=train_pairs, hard_negatives=negative_pairs, num_users=155, num_items=6498)
    val_dataset = BPRDataset(positive_pairs=val_pairs, hard_negatives=negative_pairs, num_users=155, num_items=6498)
    test_dataset = BPRDataset(positive_pairs=test_pairs, hard_negatives=negative_pairs, num_users=155, num_items=6498)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    torch.save(test_dataset, "test_dataset.pt")

    print("Creating model...")
    model = NeuralCF(num_users=155, num_items=6498, emb_size=128)

    print("Training model...")
    train_model(model=model, train_loader=train_loader, val_loader=val_loader, edges_type=edges_type, edges_index=edges_indexes, edges_weights=edges_weights, num_epochs=200)

    model.load_state_dict(torch.load("./model/checkpoint/best_model.pth"))
    test_visualization(model, test_loader,edges_indexes, edges_weights, edges_type)
    
    dataset = BPRDataset(positive_pairs=positive_pairs, hard_negatives=negative_pairs, num_users=155, num_items=6498)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    evaluate_precision_recall_k_multi(model, loader, edges_indexes, edges_type, edges_weights, num_items=6498)

    #all_score_visualization(edges_indexes, edges_weights, edges_type)