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
    def __init__(self, positive_pairs, hard_negatives=None, num_users=None, num_items=None, negative_ratio=0.5):
        self.BPR_samples = []
        self.positive_pairs = []
        self.negative_pairs = []
        self.positive_set = set()
        self.num_users = num_users
        self.num_items = num_items
        
        # Positive samples
        for _, row in positive_pairs.iterrows():
            self.positive_pairs.append((row['liquor_id'], row['ingredient_id']))
            self.positive_set.add((row['liquor_id'], row['ingredient_id']))

        # Hard negatives
        for _, row in hard_negatives.iterrows():
            self.negative_pairs.append((row['liquor_id'], row['ingredient_id']))  
        
        for pair in self.positive_pairs:
            u = pair[0]
            i = pair[1]
            for _ in range(int(negative_ratio)):
                while True:
                    j = random.randint(0, num_items - 1)
                    if (u, j) not in self.positive_set:
                        self.BPR_samples.append((u, i, j))
                        break
        
        if hard_negatives is not None:
            for pair in self.negative_pairs:
                u = pair[0]
                j = pair[1]
                positives_for_u = [i for x, i in self.positive_pairs if x == u]
                if positives_for_u:
                    i = random.choice(positives_for_u)
                    self.BPR_samples.append((u, i, j))
                        
    def __len__(self):
        return len(self.BPR_samples)
    
    def __getitem__(self, idx):
        u, pos, neg = self.BPR_samples[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(pos, dtype=torch.long), torch.tensor(neg, dtype=torch.long)
