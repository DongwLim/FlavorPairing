import pandas as pd
from collections import defaultdict
import pickle
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

def map_graph_nodes():
    nodes_df = pd.read_csv("./dataset/nodes_191120_updated.csv")

    nodes_map = {}
    liquor_map = {}
    ingredient_map = {}
    compound_map = {}
    
    for i, row in nodes_df.iterrows():
        node_id = row['node_id']
        node_type = row['node_type']
        
        if node_type == "liquor":
            liquor_map[node_id] = i
        elif node_type == "ingredient":
            ingredient_map[node_id] = i
        elif node_type == "compound":
            compound_map[node_id] = i
        
        nodes_map[node_id] = i
    nodes_map["liquor"] = liquor_map
    nodes_map["ingredient"] = ingredient_map
    nodes_map["compound"] = compound_map
    
    return nodes_map

def edges_index(edge_type_map):
    edges_df = pd.read_csv("./dataset/edges_191120_updated.csv")
    nodes_map = map_graph_nodes()

    edges_index = []
    edges_weights = []
    edges_type = []

    for _, row in tqdm(edges_df.iterrows(), desc="Processing edges...", total=len(edges_df)):
        src, tgt = row['id_1'], row['id_2']
        type = row['edge_type']
        
        if type == "ingr-fcomp" or type == "ingr-dcomp":
            continue
        
        src_idx = nodes_map[src]
        tgt_idx = nodes_map[tgt]
        
        edges_index.append((src_idx, tgt_idx))
        edges_weights.append(row['score']) if not pd.isna(row['score']) else edges_weights.append(0.1)
        edges_type.append(edge_type_map[type])
        
    edge_index = torch.tensor(edges_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edges_weights, dtype=torch.float32)
    edges_type = torch.tensor(edges_type, dtype=torch.long)
    
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge weights shape: {edge_weights.shape}")

    return edge_index, edge_weights, edges_type

class InteractionDataset(Dataset):
    def __init__(self, positive_pairs, hard_negatives, num_users, num_items, negative_ratio=1.0):
        self.samples = []
        self.num_users = num_users
        self.num_items = num_items

        # Positive samples
        for _, row in positive_pairs.iterrows():
            self.samples.append((row['liquor_id'], row['ingredient_id'], 1))

        # Hard negatives
        for _, row in hard_negatives.iterrows():
            self.samples.append((row['liquor_id'], row['ingredient_id'], 0))  

        # Negative samples
        num_neg = int(len(positive_pairs) * negative_ratio)
        for _ in range(num_neg):
            u = random.randint(0, num_users - 1)
            i = random.randint(0, num_items - 1)
            if (u, i) not in positive_pairs:
                self.samples.append((u, i, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, item, label = self.samples[idx]
        return torch.tensor(user), torch.tensor(item), torch.tensor(label, dtype=torch.float32)

def preprocess():
    nodes_df = pd.read_csv("./dataset/nodes_191120_updated.csv")
    edges_df = pd.read_csv("./dataset/flavor diffusion/edges_191120.csv")
    
    liquors_map = []
    ingredients_map = []
    compounds_map = []
    
    for _, row in nodes_df.iterrows():
        if row['node_type'] == "liquor":
            liquors_map.append(row['node_id'])
        elif row['node_type'] == "ingredient":
            ingredients_map.append(row['node_id'])
        elif row['node_type'] == "compound":
            compounds_map.append(row['node_id'])
    
    print(len(liquors_map))
    print(len(ingredients_map))
    
    with open("./model/data/liquor_key.pkl", "wb") as f:
        pickle.dump(liquors_map, f)
        
    with open("./model/data/ingredient_key.pkl", "wb") as f:
        pickle.dump(ingredients_map, f)
    
    liqr_liqr = 0
    liqr_ingr = 0
    ingr_ingr = 0
    
    for idx, row in tqdm(edges_df.iterrows(), desc="Changing Edge Type..."):
        src, tgt = row['id_1'], row['id_2']
        etype = row['edge_type']
        
        if etype == 'ingr-ingr':
            if src in liquors_map and tgt in liquors_map:
                edges_df.at[idx, 'edge_type'] = 'liqr-liqr'
                liqr_liqr += 1
            elif (src in liquors_map) ^ (tgt in liquors_map):
                edges_df.at[idx, 'edge_type'] = 'liqr-ingr'
                liqr_ingr += 1
            else:
                ingr_ingr += 1
    
    print(f"Total prev ingr-ingr edges :\t{ingr_ingr + liqr_liqr + liqr_ingr}\nChanged to ...")
    print(f"liqr-liqr edges :\t{liqr_liqr}")
    print(f"liqr_ingr edges :\t{liqr_ingr}")
    print(f"ingr_ingr edges :\t{ingr_ingr}")
    
    edges_df.to_csv("./dataset/edges_191120_updated.csv", index=False)
            
def liquors_embbed() -> dict[int, list[float]]:
    # 파일 불러오기
    nodes_df = pd.read_csv("./dataset/nodes_191120_updated.csv")
    edges_df = pd.read_csv("./dataset/edges_191120_updated.csv")

    # 노드 타입 매핑
    node_type_map = dict(zip(nodes_df['node_id'], nodes_df['node_type']))

    # 술 ID별 연결된 compound 리스트 초기화
    liquor_to_compounds = defaultdict(list)
    
    for _, row in nodes_df.iterrows():
        if row['node_type'] == 'liquor':
            liquor_to_compounds[row['node_id']] = []

    # 조건: edge_type == 'ingr-fcomp', 한 쪽이 liquor, 한 쪽이 compound인 경우
    for _, row in edges_df.iterrows():
        src, tgt = row['id_1'], row['id_2']
        etype = row['edge_type']

        if etype == 'ingr-fcomp':
            src_type = node_type_map.get(src)
            tgt_type = node_type_map.get(tgt)

            # src가 술이고 tgt가 compound인 경우
            if src_type == 'liquor' and tgt_type == 'compound':
                liquor_to_compounds[src].append(tgt)

            # tgt가 술이고 src가 compound인 경우
            elif tgt_type == 'liquor' and src_type == 'compound':
                liquor_to_compounds[tgt].append(src)
                
        if etype == 'ingr-dcomp':
            src_type = node_type_map.get(src)
            tgt_type = node_type_map.get(tgt)

            # src가 술이고 tgt가 compound인 경우
            if src_type == 'liquor' and tgt_type == 'compound':
                liquor_to_compounds[src].append(tgt)

            # tgt가 술이고 src가 compound인 경우
            elif tgt_type == 'liquor' and src_type == 'compound':
                liquor_to_compounds[tgt].append(src)

    # 결과 예시 출력
    """for liquor_id, compound_ids in list(liquor_to_compounds.items())[:5]:
        liquor_name = nodes_df[nodes_df['node_id'] == liquor_id]['name'].values[0]
        print(f"{liquor_name} ({liquor_id}): {compound_ids[:5]} ... 총 {len(compound_ids)}개")"""

    with open("./dataset/compound_embeddings_filtered.pkl", "rb") as f:
        embbed_dict = pickle.load(f)

    liquor_avg_embeddings = {}

    for liquor_id, compound_ids in list(liquor_to_compounds.items()):
        if liquor_to_compounds[liquor_id] != []: 
            valid_vectors = [embbed_dict[cid] for cid in compound_ids if cid in embbed_dict]

            if valid_vectors:  # 유효한 벡터가 하나라도 있을 경우 평균 계산
                avg_vector = np.mean(valid_vectors, axis=0)
                liquor_avg_embeddings[liquor_id] = avg_vector
        else:
            # liquor_id에 해당하는 compound_ids가 없을 경우, 랜덤 벡터 생성
            valid_vectors = [random.random() for _ in range(128)]
            liquor_avg_embeddings[liquor_id] = valid_vectors
            #print("Error occurred for liquor_id:", liquor_id)
                
    return liquor_avg_embeddings
    # 확인용 예시 출력
    """for liquor_id, vec in list(liquor_avg_embeddings.items())[:5]:
        print(f"liquor_id {liquor_id}: {vec[:5]} ... (총 길이 {len(vec)})")"""

def ingrs_embedd() -> dict[int, list[float]]: 
    # 파일 불러오기
    nodes_df = pd.read_csv("./dataset/nodes_191120_updated.csv")
    edges_df = pd.read_csv("./dataset/edges_191120_updated.csv")

    # 노드 타입 매핑
    node_type_map = dict(zip(nodes_df['node_id'], nodes_df['node_type']))

    # 음식 ID별 연결된 compound 리스트 초기화
    ingr_to_compounds = defaultdict(list)
    
    for _, row in nodes_df.iterrows():
        if row['node_type'] == 'ingredient':
            ingr_to_compounds[row['node_id']] = []

    # 조건: edge_type == 'ingr-fcomp', 한 쪽이 ingredient, 한 쪽이 compound인 경우
    for _, row in edges_df.iterrows():
        src, tgt = row['id_1'], row['id_2']
        etype = row['edge_type']

        if etype == 'ingr-fcomp' or etype == 'ingr-dcomp':
            src_type = node_type_map.get(src)
            tgt_type = node_type_map.get(tgt)

            # src가 음식이고 tgt가 compound인 경우
            if src_type == 'ingredient' and tgt_type == 'compound':
                ingr_to_compounds[src].append(tgt)

            # tgt가 음식이고 src가 compound인 경우
            elif tgt_type == 'ingredient' and src_type == 'compound':
                ingr_to_compounds[tgt].append(src)
    
    with open("./dataset/compound_embeddings_filtered.pkl", "rb") as f:
        embbed_dict = pickle.load(f)

    ingredient_avg_embeddings = {}

    for ingredient_id, compound_ids in list(ingr_to_compounds.items()):
        if ingr_to_compounds[ingredient_id] != []:
            valid_vectors = [embbed_dict[cid] for cid in compound_ids if cid in embbed_dict]

            if valid_vectors:  # 유효한 벡터가 하나라도 있을 경우 평균 계산
                avg_vector = np.mean(valid_vectors, axis=0)
                ingredient_avg_embeddings[ingredient_id] = avg_vector
        else:
            # ingredient_id에 해당하는 compound_ids가 없을 경우, 랜덤 벡터 생성
            valid_vectors = [random.random() for _ in range(128)]
            ingredient_avg_embeddings[ingredient_id] = valid_vectors
            
    return ingredient_avg_embeddings

def make_emb():
    liquor_avg_embeddings = dict(sorted(liquors_embbed().items()))
    ingredient_avg_embeddings = dict(sorted(ingrs_embedd().items()))

    liquor_key = list(liquor_avg_embeddings.keys())
    print(liquor_key)
    print(f"liquor num\t: {len(liquor_key)}")
    ingredient_key = list(ingredient_avg_embeddings.keys())
    #print(ingredient_key)
    print(f"ingredient num\t: {len(ingredient_key)}")

    with open("./model/data/liquor_key.pkl", "wb") as f:
        pickle.dump(liquor_key, f)

    with open("./model/data/ingredient_key.pkl", "wb") as f:
        pickle.dump(ingredient_key, f)

    liquor_embedding_tensor = torch.tensor(np.stack(list(liquor_avg_embeddings.values())), dtype=torch.float32)
    ingredient_embedding_tensor = torch.tensor(np.stack(list(ingredient_avg_embeddings.values())), dtype=torch.float32)

    print("Liquor Embedding Tensor")
    print("  - Shape:", liquor_embedding_tensor.shape)
    print("  - Dtype:", liquor_embedding_tensor.dtype) 

    print("\nIngredient Embedding Tensor")
    print("  - Shape:", ingredient_embedding_tensor.shape)
    print("  - Dtype:", ingredient_embedding_tensor.dtype)

    torch.save(liquor_embedding_tensor, "./model/data/liquor_init_embedding.pt")
    torch.save(ingredient_embedding_tensor, "./model/data/ingredient_init_embedding.pt")
    
class TripletInteractionDataset(Dataset):
    def __init__(self, positive_pairs, hard_negatives=None, num_users=None, num_items=None, negative_ratio=5.0):
        self.triplets = []
        self.positive_pairs = list(positive_pairs)
        self.positive_set = set(positive_pairs)

        # Positive samples + Negative samples(random)
        for u, i in self.positive_pairs:
            for _ in range(int(negative_ratio)):
                while True:
                    j = random.randint(0, num_items - 1)
                    if (u, j) not in self.positive_set:
                        self.triplets.append((u, i, j))
                        break

        # Hard negatives
        if hard_negatives is not None:
            for u, j in hard_negatives:
                positives_for_u = [i for x, i in self.positive_pairs if x == u]
                if positives_for_u:
                    i = random.choice(positives_for_u)
                    self.triplets.append((u, i, j))  # (anchor, positive, hard negative)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        u, pos, neg = self.triplets[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(pos, dtype=torch.long), torch.tensor(neg, dtype=torch.long)

if __name__ == "__main__":
    preprocess()
