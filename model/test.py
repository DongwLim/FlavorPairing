import pandas as pd
import pickle
from dataset import map_graph_nodes, edges_index
import torch
from models import NeuralCF

def predict(user_ids, item_ids, edges_indexes, edges_weights):
    model = NeuralCF(num_users=162, num_items=6491, emb_size=128)
    model.load_state_dict(torch.load("./model/checkpoint/epoch_17.pth"))
    model.eval()

    with torch.no_grad():
        output = model(torch.tensor(user_ids), torch.tensor(item_ids), edges_indexes, edges_weights)
        #print(output)
        return output

mapping = map_graph_nodes()
    
lid_to_idx = mapping['liquor']
iid_to_idx = mapping['ingredient']

edges_indexes, edges_weights = edges_index()

while True:
    liquqor, ingredient = input("술과 재료를 입력 : ").split()
    print(predict(lid_to_idx[int(liquqor)], iid_to_idx[int(ingredient)], edges_indexes, edges_weights))