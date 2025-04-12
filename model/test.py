import pandas as pd
import pickle

import torch
from models import NeuralCF

def predict(user_ids, item_ids):
    liquor_embedding_tensor = torch.load("./model/data/liquor_init_embedding.pt")
    ingredient_embedding_tensor = torch.load("./model/data/ingredient_init_embedding.pt")

    model = NeuralCF(num_users=23, num_items=393, emb_size=128,user_init=liquor_embedding_tensor, item_init=ingredient_embedding_tensor)
    model.load_state_dict(torch.load("./model/checkpoint/epoch_19.pth"))
    model.eval()

    with torch.no_grad():
        output = model(torch.tensor(user_ids), torch.tensor(item_ids))
        #print(output)
        return output

nodes_info = pd.read_csv("./dataset/Hub_Nodes.csv")

with open("./model/data/ingredient_key.pkl", "rb") as f:
    ingredient_keys = pickle.load(f)

with open("./model/data/liquor_key.pkl", "rb") as f:
    liquor_keys = pickle.load(f)

iid_to_idx = {item_id: idx for idx, item_id in enumerate(ingredient_keys)}
lid_to_idx = {item_id: idx for idx, item_id in enumerate(liquor_keys)}

"""for i in range(393):
    s = predict(lid_to_idx[5676], i)
    if s > 0.5:
        print(ingredient_keys[i])"""

while True:
    liquqor, ingredient = input("술과 재료를 입력 : ").split()
    print(predict(lid_to_idx[int(liquqor)], iid_to_idx[int(ingredient)]))