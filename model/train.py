import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import pickle

from dataset import InteractionDataset
from models import NeuralCF

def train_model(model, train_loader, num_epochs=10, lr=0.001, weight_decay=1e-5):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)
    
    model.to(device)
    model.train()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for user, item, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            user, item, label = user.to(device), item.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(user, item)
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

        torch.save(model.state_dict(), f"./model/checkpoint/epoch_{epoch}.pth")

if __name__ == "__main__":
    positive_pairs = pd.read_csv("Expanded_Compatible_Pairs_by_Recipe.csv")
    positive_pairs = positive_pairs[['liquor_id', 'ingredient_id']]

    negative_pairs = pd.read_csv("Expanded_Incompatible_Pairs_by_Recipe.csv")
    negative_pairs = negative_pairs[['liquor_id', 'ingredient_id']]

    with open("./model/data/ingredient_key.pkl", "rb") as f:
        ingredient_keys = pickle.load(f)

    with open("./model/data/liquor_key.pkl", "rb") as f:
        liquor_keys = pickle.load(f)

    iid_to_idx = {item_id: idx for idx, item_id in enumerate(ingredient_keys)}
    lid_to_idx = {item_id: idx for idx, item_id in enumerate(liquor_keys)}

    positive_pairs['liquor_id'] = positive_pairs['liquor_id'].map(lid_to_idx)
    positive_pairs['ingredient_id'] = positive_pairs['ingredient_id'].map(iid_to_idx)

    negative_pairs['liquor_id'] = negative_pairs['liquor_id'].map(lid_to_idx)
    negative_pairs['ingredient_id'] = negative_pairs['ingredient_id'].map(iid_to_idx)

    train_dataset = InteractionDataset(positive_pairs=positive_pairs, hard_negatives=negative_pairs, num_users=22, num_items=394)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    liquor_embedding_tensor = torch.load("./model/data/liquor_init_embedding.pt")
    ingredient_embedding_tensor = torch.load("./model/data/ingredient_init_embedding.pt")

    model = NeuralCF(num_users=22, num_items=394, emb_size=128,user_init=liquor_embedding_tensor, item_init=ingredient_embedding_tensor)

    train_model(model=model, train_loader=train_loader, num_epochs=20)