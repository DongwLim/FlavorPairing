from dataset import map_graph_nodes, edges_index
import torch
from models import NeuralCF
import pandas as pd

def predict(user_ids, item_ids, edges_indexes, edges_weights, edge_type):
    model = NeuralCF(num_users=155, num_items=6498, emb_size=128)
    model.load_state_dict(torch.load("./model/checkpoint/best_model.pth"))
    model.eval()

    with torch.no_grad():
        output = model(torch.tensor(user_ids), torch.tensor(item_ids), edges_indexes, edge_type, edges_weights)
        #print(output)
        return output

def get_top_k_items(user_id, all_item, edges_indexes, edge_type, edges_weights, k=5):
    """
    user_id: 사용자 ID (int)
    all_items: 전체 아이템 목록 (list of item indices or item IDs)
    edges_indexes, edge_type, edges_weights: GNN용 그래프 정보
    k: 추천할 아이템 개수
    """
    model = NeuralCF(num_users=155, num_items=6498, emb_size=128)
    model.load_state_dict(torch.load("./model/checkpoint/best_model.pth"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    all_items = torch.arange(6498).to(device)

    user_tensor = torch.tensor([user_id] * len(all_items), device=device)
    item_tensor = torch.tensor(all_items, device=device)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor, edges_indexes.to(device), edge_type.to(device), edges_weights.to(device))

    top_k_indices = torch.topk(scores, k).indices
    top_k_indices = top_k_indices.cpu().tolist()  # 숫자만 남도록 변환
    top_k_items = [int(all_items[i]) for i in top_k_indices]

    return top_k_items


mapping = map_graph_nodes()
    
lid_to_idx = mapping['liquor']
iid_to_idx = mapping['ingredient']

edge_type_map ={
        'liqr-ingr': 0,
        'ingr-ingr': 1,
        'liqr-liqr': 1,
        'ingr-fcomp': 2,
        'ingr-dcomp': 2
    }

edges_indexes, edges_weights, edge_type = edges_index(edge_type_map)

liquor = input("술을 입력 : ")
topk = int(input("추천할 재료 개수 : "))

items = get_top_k_items(lid_to_idx[int(liquor)], iid_to_idx.keys(), edges_indexes, edge_type, edges_weights, topk)

df = pd.read_csv("./dataset/nodes_191120_updated.csv")
item_names = df[df['node_id'].isin(items)]['name'].tolist()

print(f"술 ID: {liquor}")
print(f"술 이름: {df[df['node_id'] == int(liquor)]['name'].values[0]}")
print(f"추천된 재료: {items}")
#print(f"추천된 재료: {item_names}")

"""for i in iid_to_idx.keys():
    score = predict(lid_to_idx[int(liquor)], iid_to_idx[i], edges_indexes, edges_weights, edge_type)
    if score > -2.0:
        print(f"{i} : {score}")"""
        
"""while True:
    liquqor, ingredient = input("술과 재료를 입력 : ").split()
    print(predict(lid_to_idx[int(liquqor)], iid_to_idx[int(ingredient)], edges_indexes, edges_weights, edge_type))"""