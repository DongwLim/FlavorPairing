import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

from dataset import InteractionDataset, map_graph_nodes, edges_index
from models import NeuralCF

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
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

def train_model(model, train_loader, val_loader, edges_index, edges_weights, num_epochs=10, lr=0.001, weight_decay=1e-5):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)
    
    model.to(device)
    model.train()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=10)

    print(f"Training on {device}")
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for user, item, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            user = user.long()   
            item = item.long()   
            label = label.float()

            user, item, label = user.to(device), item.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(user, item, edges_index, edges_weights)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * label.size(0)
            predicted = (output > 0.5).float()
            correct += (predicted == label).sum().item()
            total += label.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for user, item, label in val_loader:
                user = user.long()
                item = item.long()
                label = label.float()

                user, item, label = user.to(device), item.to(device), label.to(device)

                output = model(user, item, edges_index, edges_weights)
                loss = criterion(output, label)

                val_loss += loss.item() * label.size(0)
                predicted = (output > 0.5).float()
                val_correct += (predicted == label).sum().item()
                val_total += label.size(0)
        
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"[Validation] Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f}")
        torch.save(model.state_dict(), f"./model/checkpoint/epoch_{epoch}.pth")
        
        # Check Early Stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    print("Loading data...")
    mapping = map_graph_nodes()
    
    lid_to_idx = mapping['liquor']
    iid_to_idx = mapping['ingredient']

    #print(lid_to_idx)
    print("Loading graph data...")
    edges_indexes, edges_weights = edges_index()
    
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
    num_users = 162 # Number of unique liquor IDs
    num_items = 6491 # Number of unique ingredient IDs
    """
    
    print("Creating dataset...")
    positive_pairs['label'] = 1
    negative_pairs['label'] = 0
    all_pairs = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
    
    train_val_pairs, test_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.2, random_state=42)
    
    train_dataset = InteractionDataset(positive_pairs=train_pairs, hard_negatives=negative_pairs, num_users=162, num_items=6491)
    val_dataset = InteractionDataset(positive_pairs=val_pairs, hard_negatives=negative_pairs, num_users=162, num_items=6491)
    test_dataset = InteractionDataset(positive_pairs=test_pairs, hard_negatives=negative_pairs, num_users=162, num_items=6491)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print("Creating model...")
    model = NeuralCF(num_users=162, num_items=6491, emb_size=128)

    print("Training model...")
    train_model(model=model, train_loader=train_loader, val_loader=val_loader ,edges_index=edges_indexes, edges_weights=edges_weights, num_epochs=200)