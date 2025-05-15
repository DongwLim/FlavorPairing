from dataset import map_graph_nodes, edges_index
import torch
from models import NeuralCF
from utils import smoothed_scaled_score
import pandas as pd
import json
from tqdm import tqdm

def predict(user_ids, item_ids, edges_indexes, edges_weights, edge_type):
    model = NeuralCF(num_users=155, num_items=6498, emb_size=128)
    model.load_state_dict(torch.load("./model/checkpoint/best_model.pth"))
    model.eval()

    with torch.no_grad():
        output = model(torch.tensor(user_ids), torch.tensor(item_ids), edges_indexes, edge_type, edges_weights)
        #print(output)
        return output
    
def topk_predict(user_id, edges_index, edges_weights, edge_type, topk, device='cpu'):
    model = NeuralCF(num_users=155, num_items=6498, emb_size=128)
    model.load_state_dict(torch.load("./model/checkpoint/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    num_items = 6498
    user_tensor = torch.tensor([user_id] * num_items, device=device)
    item_tensor = torch.arange(num_items, device=device)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor, edges_index, edge_type, edges_weights)
        topk_scores, topk_indices = torch.topk(scores, k=topk)

    return topk_indices.cpu().tolist()  # 추천 item ID 리스트


mapping = map_graph_nodes()
    
lid_to_idx = mapping['liquor']
iid_to_idx = mapping['ingredient']

idx_to_id = {v: k for k, v in iid_to_idx.items()}

edge_type_map ={
        'liqr-ingr': 0,
        'ingr-ingr': 1,
        'liqr-liqr': 1,
        'ingr-fcomp': 2,
        'ingr-dcomp': 2
    }

edges_indexes, edges_weights, edge_type = edges_index(edge_type_map)

df = pd.read_csv("./dataset/nodes_191120_updated.csv")

"""liquor = int(input("술을 입력 : "))
topk = int(input("topk : "))

topk_items = topk_predict(lid_to_idx[liquor], edges_indexes, edges_weights, edge_type, topk)

items_id = [idx_to_id[i] for i in topk_items]
print(items_id)
print(f"추천된 재료들 : {df[df['node_id'].isin(items_id)]['name'].values.tolist()}")"""
topk = 100
# 결과 저장 딕셔너리
all_recommendations = {}

# liquor_id 목록을 가져온다고 가정
liquor_ids = list(lid_to_idx.keys())

for liquor_id in tqdm(liquor_ids, desc="Generating top-k recommendations"):
    try:
        liquor_name = df[df['node_id'] == liquor_id]['name'].values[0]
        user_idx = lid_to_idx[liquor_id]
        
        # 추천 top-k 아이템 추출
        topk_items = topk_predict(user_idx, edges_indexes, edges_weights, edge_type, topk)

        # ID → 이름 변환
        topk_names = [df[df['node_id'] == idx_to_id[i]]['name'].values[0] for i in topk_items]

        # 결과 저장
        all_recommendations[liquor_name] = topk_names

    except Exception as e:
        print(f"Error processing liquor_id {liquor_id}: {e}")
        continue

# JSON 파일로 저장
with open("liquor_topk_recommendations_with_name.json", "w", encoding="utf-8") as f:
    json.dump(all_recommendations, f, ensure_ascii=False, indent=1)

print("저장 완료: liquor_topk_recommendations.json")


"""for i in iid_to_idx.keys():
    score = predict(lid_to_idx[int(liquor)], iid_to_idx[i], edges_indexes, edges_weights, edge_type)
    if score > -2.0:
        print(f"{i} : {score}")"""
        
"""while True:
    liquqor, ingredient = input("술과 재료를 입력 : ").split()
    print(smoothed_scaled_score(predict(lid_to_idx[int(liquqor)], iid_to_idx[int(ingredient)], edges_indexes, edges_weights, edge_type)))"""