import pandas as pd
from collections import defaultdict
import numpy as np
cimport numpy as np
from tqdm import tqdm

import torch

from libc.string cimport strcmp
from libc.stdlib cimport rand, srand
from libc.time cimport time

def get_rand(index):
    return rand() % index

def cython_map_graph_nodes():
    nodes_df = pd.read_csv("./dataset/nodes_191120_updated.csv")

    cdef dict nodes_map = {}
    cdef dict liquor_map = {}
    cdef dict ingredient_map = {}
    cdef dict compound_map = {}
    
    cdef int i
    cdef int df_len = len(nodes_df)

    node_ids = nodes_df['node_id'].to_numpy()
    node_types = nodes_df['node_type'].to_numpy()

    for i in range(df_len):
        node_id = node_ids[i]
        node_type = node_types[i]
        
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

def cython_edges_index(dict edge_type_map):
    edges_df = pd.read_csv("./dataset/edges_191120_updated.csv")
    nodes_map = cython_map_graph_nodes()

    cdef int edge_idx = 0
    cdef int offset = 0

    cdef int df_len = len(edges_df)
    cdef int src, tgt
    cdef bytes typ

    cdef np.ndarray[np.int64_t, ndim=2] edges_index = np.zeros((df_len, 2), dtype=np.int64)
    cdef np.ndarray[np.float32_t, ndim=1] edges_weights = np.zeros(df_len, dtype=np.float32)
    cdef np.ndarray[np.int64_t, ndim=1] edges_types = np.zeros(df_len, dtype=np.int64)

    edges_id_1 = edges_df['id_1'].to_numpy()
    edges_id_2 = edges_df['id_2'].to_numpy()
    edges_score = edges_df['score'].to_numpy()
    edges_type = edges_df['edge_type'].to_numpy()

    for edge_idx in tqdm(range(df_len), desc="Processing edges...", total=df_len):
        src = edges_id_1[edge_idx - offset]
        tgt = edges_id_2[edge_idx - offset]
        typ = edges_type[edge_idx - offset].encode("utf-8")

        if strcmp(typ, b'ingr-fcomp') == 0 or strcmp(typ, b'ingr-dcomp') == 0:
            offset = offset + 1     # 이번 루프는 edge_idx와 관계없음
            continue
        
        edges_index[edge_idx - offset, 0] = nodes_map[src]
        edges_index[edge_idx - offset, 1] = nodes_map[tgt]

        if not pd.isna(edges_score[edge_idx - offset]):
            edges_weights[edge_idx - offset] = edges_score[edge_idx - offset]
        else:
            edges_weights[edge_idx - offset] = 0.1

        edges_types[edge_idx - offset] = edge_type_map[typ.decode("utf-8")]

    edges_index = edges_index[:df_len - offset, :]
    edges_weights = edges_weights[:df_len - offset]
    edges_types = edges_types[:df_len - offset]

    result_index = torch.tensor(edges_index, dtype=torch.long).t().contiguous()
    result_weights = torch.tensor(edges_weights, dtype=torch.float32)
    result_types = torch.tensor(edges_types, dtype=torch.long)
    
    print(f"Edge index shape: {result_index.shape}")
    print(f"Edge weights shape: {result_weights.shape}")

    return result_index, result_weights, result_types