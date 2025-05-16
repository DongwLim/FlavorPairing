import pandas as pd
from collections import defaultdict
import pickle
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from collections import defaultdict

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
    def __init__(self, positive_pairs, hard_negatives, num_users, num_items, negative_ratio=5.0):
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
    
class BPRDataset(Dataset):
    def __init__(self, positive_pairs, hard_negatives=None, num_users=None, num_items=None, negative_ratio=5.0):
        positive_pairs = positive_pairs[['liquor_id', 'ingredient_id']].to_numpy(dtype=np.int64)
        if hard_negatives is not None:
            hard_negatives = hard_negatives[['liquor_id', 'ingredient_id']].to_numpy(dtype=np.int64)

        samples = self.generate_bpr_samples(positive_pairs, hard_negatives, num_users, num_items, int(negative_ratio))
        self.BPR_samples = samples
                        
    def __len__(self):
        return len(self.BPR_samples)
    
    def __getitem__(self, idx):
        u, pos, neg = self.BPR_samples[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(pos, dtype=torch.long), torch.tensor(neg, dtype=torch.long)
    
    def generate_bpr_samples(self, positive_pairs, hard_negatives=None, num_users=None, num_items=None, negative_ratio=5):
        pos_u = positive_pairs[:, 0]
        pos_i = positive_pairs[:, 1]

        pos_set = set(map(tuple, positive_pairs.tolist()))

        total_pos = len(pos_u)
        target_size = total_pos * negative_ratio

        u_repeat = np.repeat(pos_u, negative_ratio)
        i_repeat = np.repeat(pos_i, negative_ratio)

        neg_j = np.random.randint(0, num_items, size=target_size)
        mask = np.array([(u, j) not in pos_set for u, j in zip(u_repeat, neg_j)])

        # Run loop until there are no more remaining positives
        while not np.all(mask):
            n_redraw = (~mask).sum()
            new_j = np.random.randint(0, num_items, size=n_redraw)
            neg_j[~mask] = new_j
            mask = np.array([(u, j) not in pos_set for u, j in zip(u_repeat, neg_j)])

        bpr_samples = np.stack([u_repeat, i_repeat, neg_j], axis=1)

        if hard_negatives is not None:
            hard_u = hard_negatives[:, 0]
            hard_j = hard_negatives[:, 1]

            user_to_pos = defaultdict(list)
            for u, i in zip(pos_u, pos_i):
                user_to_pos[u].append(i)

            hard_samples = []
            for u, j in zip(hard_u, hard_j):
                if user_to_pos[u]:
                    i = np.random.choice(user_to_pos[u])
                    hard_samples.append([u, i, j])
            hard_samples = np.array(hard_samples, dtype=np.int64)

            bpr_samples = np.vstack([bpr_samples, hard_samples])

        return bpr_samples