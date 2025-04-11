import pandas as pd
from collections import defaultdict
import pickle
import numpy as np

def liquors_embbed():
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

def ingrs_embedd():
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

        if etype == 'ingr-fcomp':
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

    return ingredient_avg_embeddings

ingrs_embedd()