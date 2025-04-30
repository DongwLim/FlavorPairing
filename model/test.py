from dataset import map_graph_nodes, edges_index
import torch
from models import NeuralCF

def predict(user_ids, item_ids, edges_indexes, edges_weights, edge_type):
    model = NeuralCF(num_users=155, num_items=6498, emb_size=128)
    model.load_state_dict(torch.load("./model/checkpoint/best_model.pth"))
    model.eval()

    with torch.no_grad():
        output = model(torch.tensor(user_ids), torch.tensor(item_ids), edges_indexes, edge_type, edges_weights)
        #print(output)
        return output

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

"""for i in iid_to_idx.keys():
    score = predict(lid_to_idx[int(liquor)], iid_to_idx[i], edges_indexes, edges_weights, edge_type)
    if score > -2.0:
        print(f"{i} : {score}")"""
        
while True:
    liquqor, ingredient = input("술과 재료를 입력 : ").split()
    print(predict(lid_to_idx[int(liquqor)], iid_to_idx[int(ingredient)], edges_indexes, edges_weights, edge_type))