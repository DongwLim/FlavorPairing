import pandas as pd
from collections import defaultdict
import pickle
import numpy as np
import random

import torch
from torch.utils.data import Dataset

def liquors_embbed() -> dict[int, list[float]]:
    # 파일 불러오기
    nodes_df = pd.read_csv("./dataset/Hub_Nodes.csv")
    edges_df = pd.read_csv("./dataset/Hub_Edges.csv")

    # 노드 타입 매핑
    node_type_map = dict(zip(nodes_df['node_id'], nodes_df['node_type']))

    # 술 ID별 연결된 compound 리스트 초기화
    liquor_to_compounds = defaultdict(list)

    # 조건: edge_type == 'ingr-fcomp', 한 쪽이 liquor, 한 쪽이 compound인 경우
    for _, row in edges_df.iterrows():
        src, tgt = row['source'], row['target']
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

    # 결과 예시 출력
    """for liquor_id, compound_ids in list(liquor_to_compounds.items())[:5]:
        liquor_name = nodes_df[nodes_df['node_id'] == liquor_id]['name'].values[0]
        print(f"{liquor_name} ({liquor_id}): {compound_ids[:5]} ... 총 {len(compound_ids)}개")"""

    with open("./dataset/compound_embeddings_filtered.pkl", "rb") as f:
        embbed_dict = pickle.load(f)

    liquor_avg_embeddings = {}

    for liquor_id, compound_ids in list(liquor_to_compounds.items()):
        valid_vectors = [embbed_dict[cid] for cid in compound_ids if cid in embbed_dict]

        if valid_vectors:  # 유효한 벡터가 하나라도 있을 경우 평균 계산
            avg_vector = np.mean(valid_vectors, axis=0)
            liquor_avg_embeddings[liquor_id] = avg_vector

    return liquor_avg_embeddings
    # 확인용 예시 출력
    """for liquor_id, vec in list(liquor_avg_embeddings.items())[:5]:
        print(f"liquor_id {liquor_id}: {vec[:5]} ... (총 길이 {len(vec)})")"""

def ingrs_embedd() -> dict[int, list[float]]: 
    # 파일 불러오기
    nodes_df = pd.read_csv("./dataset/Hub_Nodes.csv")
    edges_df = pd.read_csv("./dataset/Hub_Edges.csv")

    # 노드 타입 매핑
    node_type_map = dict(zip(nodes_df['node_id'], nodes_df['node_type']))

    # 음식 ID별 연결된 compound 리스트 초기화
    ingr_to_compounds = defaultdict(list)

    # 조건: edge_type == 'ingr-fcomp', 한 쪽이 ingredient, 한 쪽이 compound인 경우
    for _, row in edges_df.iterrows():
        src, tgt = row['source'], row['target']
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
        valid_vectors = [embbed_dict[cid] for cid in compound_ids if cid in embbed_dict]

        if valid_vectors:  # 유효한 벡터가 하나라도 있을 경우 평균 계산
            avg_vector = np.mean(valid_vectors, axis=0)
            ingredient_avg_embeddings[ingredient_id] = avg_vector
        else:
            print("비상비상")

    return ingredient_avg_embeddings

def make_emb():
    liquor_avg_embeddings = dict(sorted(liquors_embbed().items()))
    ingredient_avg_embeddings = dict(sorted(ingrs_embedd().items()))

    liquor_key = list(liquor_avg_embeddings.keys())
    #print(liquor_key)
    print(len(liquor_key))
    ingredient_key = list(ingredient_avg_embeddings.keys())
    #print(ingredient_key)
    print(len(ingredient_key))

    with open("./model/data/liquor_key.pkl", "wb") as f:
        pickle.dump(liquor_key, f)

    with open("./model/data/ingredient_key.pkl", "wb") as f:
        pickle.dump(ingredient_key, f)

    liquor_embedding_tensor = torch.tensor(np.stack(list(liquor_avg_embeddings.values())), dtype=torch.float32)
    ingredient_embedding_tensor = torch.tensor(np.stack(list(ingredient_avg_embeddings.values())), dtype=torch.float32)

    """print("Liquor Embedding Tensor")
    print("  - Shape:", liquor_embedding_tensor.shape)
    print("  - Dtype:", liquor_embedding_tensor.dtype)
    print("  - Sample:\n", liquor_embedding_tensor[:2])  

    print("\nIngredient Embedding Tensor")
    print("  - Shape:", ingredient_embedding_tensor.shape)
    print("  - Dtype:", ingredient_embedding_tensor.dtype)
    print("  - Sample:\n", ingredient_embedding_tensor[:2])"""

    torch.save(liquor_embedding_tensor, "./model/data/liquor_init_embedding.pt")
    torch.save(ingredient_embedding_tensor, "./model/data/ingredient_init_embedding.pt")

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
    
class TripletInteractionDataset(Dataset):
    def __init__(self, positive_pairs, hard_negatives=None, num_users=None, num_items=None, negative_ratio=1.0):
        self.triplets = []
        self.positive_pairs = list(positive_pairs)
        self.positive_set = set(positive_pairs)

        # 일반 랜덤 negative sampling
        for u, i in self.positive_pairs:
            for _ in range(int(negative_ratio)):
                while True:
                    j = random.randint(0, num_items - 1)
                    if (u, j) not in self.positive_set:
                        self.triplets.append((u, i, j))
                        break

        # 하드 네거티브 추가
        if hard_negatives is not None:
            for u, j in hard_negatives:
                # u와 연결된 실제 positive 중 하나 선택
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
    make_emb()