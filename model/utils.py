import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
import math
import numpy as np
from tqdm import tqdm

def build_interaction_matrix_compound_overlap():
    """
    술과 음식의 조합에 공유하는 공통 컴파운드가 너무 많음
    특이 조합을 interaction matrix에 추가해야 되는데 특이 조합은 없고 모든 조합이 다 들어감
    -> 의미 없는 데이터
    """
    node_df = pd.read_csv("./dataset/Hub_Nodes.csv")
    edge_df = pd.read_csv("./dataset/Hub_Edges.csv")
    alcohol_df = pd.read_csv("./dataset/Alcohol-Related_Hub_Nodes.csv")

    compound_ids = set(node_df[node_df["node_type"] == "compound"]["node_id"])
    ingredient_ids = set(node_df[node_df["node_type"] == "ingredient"]["node_id"])
    alcohol_ids = set(alcohol_df["node_id"])

    ingr_fcomp_edges = edge_df[edge_df['edge_type'] == 'ingr-fcomp']

    # 알코올과 화합물 연결
    alcohol_compound_map = defaultdict(set)
    for _, row in ingr_fcomp_edges.iterrows():
        if row['source'] in alcohol_ids and row['target'] in compound_ids:
            alcohol_compound_map[row['source']].add(row['target'])

    # 재료와 화합물 연결
    compound_ingredient_map = defaultdict(set)
    ingredient_compound_map = defaultdict(set)
    for _, row in ingr_fcomp_edges.iterrows():
        if row['source'] in ingredient_ids and row['target'] in compound_ids:
            compound_ingredient_map[row['target']].add(row['source']) 
            ingredient_compound_map[row['source']].add(row['target'])

    # 상호작용 저장
    interaction_set = set()
    for alcohol_id, alcohol_compounds in alcohol_compound_map.items():
        for compound_id in alcohol_compounds:
            for ingredient_id in compound_ingredient_map.get(compound_id, []):
                ingredient_compounds = ingredient_compound_map.get(ingredient_id, set())
                shared_compounds = alcohol_compounds.intersection(ingredient_compounds)
                if len(shared_compounds) >= 30:
                    interaction_set.add((alcohol_id, ingredient_id))

    # DataFrame으로 변환
    interaction_df = pd.DataFrame(
        [(a, i, 1) for a, i in interaction_set],
        columns=['alcohol_id', 'ingredient_id', 'label']
    )
    interaction_df.to_csv("./filtered_interactions.csv", index=False)

def build_interaction_matrix_TF_IDF(threshold=2):
    """
    단순한 공유 화합물이 존재한다고 interaction이 있다고 간주하기 보다 술과 음식이 공유하는 '특이' 화합물에 집중
    하지만 단일 화합물 공유 기반의 연결 방식 여전히 느슨하고 보편적
    단일 화합물이 너무 많은 술과 음식이 공유하는 중 -> 특이 단일 화합물이 없다 -> 데이터에 의미가 없다 
    """
    node_df = pd.read_csv("./dataset/Hub_Nodes.csv")
    edge_df = pd.read_csv("./dataset/Hub_Edges.csv")
    alcohol_df = pd.read_csv("./dataset/Alcohol-Related_Hub_Nodes.csv")

    compound_ids = set(node_df[node_df["node_type"] == "compound"]["node_id"])
    ingredient_ids = set(node_df[node_df["node_type"] == "ingredient"]["node_id"])
    alcohol_ids = set(alcohol_df["node_id"])

    ingr_fcomp_edges = edge_df[edge_df['edge_type'] == 'ingr-fcomp']

    # 알코올-컴파운드 맵
    alcohol_compound_map = defaultdict(set)
    for _, row in ingr_fcomp_edges.iterrows():
        if row['source'] in alcohol_ids and row['target'] in compound_ids:
            alcohol_compound_map[row['source']].add(row['target'])

    # 재료-컴파운드 맵
    compound_ingredient_map = defaultdict(set)
    ingredient_compound_map = defaultdict(set)
    for _, row in ingr_fcomp_edges.iterrows():
        if row['source'] in ingredient_ids and row['target'] in compound_ids:
            compound_ingredient_map[row['target']].add(row['source']) 
            ingredient_compound_map[row['source']].add(row['target'])

    # 컴파운드 빈도 계산 (얼마나 많은 ingredient에 나타나는지)
    compound_freq = defaultdict(int)
    for ingr_id, compounds in ingredient_compound_map.items():
        for c in compounds:
            compound_freq[c] += 1

    # 전체 ingredient 수로 나눠서 희귀도 계산
    num_ingredients = len(ingredient_ids)
    compound_rarity = {
        c: math.log(num_ingredients / (1 + compound_freq[c]))  # log 스무딩
        for c in compound_freq
    }

    interaction_data = []
    for alcohol_id, alcohol_compounds in alcohol_compound_map.items():
        for ingredient_id, ingredient_compounds in ingredient_compound_map.items():
            shared_compounds = alcohol_compounds.intersection(ingredient_compounds)
            if not shared_compounds:
                continue

            # 공유된 컴파운드의 희귀도 합산 점수
            score = np.mean([compound_rarity.get(c, 0) for c in shared_compounds])

            if score >= threshold:
                interaction_data.append((alcohol_id, ingredient_id, 1))

    interaction_df = pd.DataFrame(interaction_data, columns=['alcohol_id', 'ingredient_id', 'score'])
    interaction_df.to_csv("./filtered_interactions.csv", index=False)
    print(f"총 {len(interaction_df)}개의 술-재료 상호작용이 저장되었습니다.")

#build_interaction_matrix_TF_IDF(2.5)

def build_interaction_matrix_compound_combination(n=2, rare_threshold=5, score_threshold=10):
    """
    특이한 조합이 나오긴 하지만... 특정 술에 집중됨 -> 전체성 떨어짐
    """
    node_df = pd.read_csv("./dataset/Hub_Nodes.csv")
    edge_df = pd.read_csv("./dataset/Hub_Edges.csv")
    
    compound_ids = set(node_df[node_df["node_type"] == "compound"]["node_id"])
    ingredient_ids = set(node_df[node_df["node_type"] == "ingredient"]["node_id"])
    alcohol_ids = set(node_df[node_df["node_type"] == "liquor"]["node_id"])

    ingr_fcomp_edges = edge_df[edge_df['edge_type'] == 'ingr-fcomp']
    
    alcohol_compound_map = defaultdict(set)
    ingredient_compound_map = defaultdict(set)

    for _, row in ingr_fcomp_edges.iterrows():
        if row['source'] in alcohol_ids and row['target'] in compound_ids:
            alcohol_compound_map[row['source']].add(row['target'])
        elif row['source'] in ingredient_ids and row['target'] in compound_ids:
            ingredient_compound_map[row['source']].add(row['target'])

    pair_counter = Counter()
    for compounds in tqdm(ingredient_compound_map.values(), desc="Counting compound combinations"):
        for pair in combinations(sorted(compounds), n):
            pair_counter[pair] += 1

    interaction_data = []
    for alcohol_id, alcohol_compounds in alcohol_compound_map.items():
        alcohol_combos = set(combinations(sorted(alcohol_compounds), n))
        for ingredient_id, ingredient_compounds in tqdm(ingredient_compound_map.items(), desc=f"Matching ingredients for {alcohol_id}"):
            ingredient_combos = set(combinations(sorted(ingredient_compounds), n))
            shared_rare_combos = alcohol_combos.intersection(ingredient_combos)
            rare_combos = [combo for combo in shared_rare_combos if pair_counter[combo] <= rare_threshold]
            if rare_combos and score_threshold <= len(rare_combos):
                interaction_data.append((alcohol_id, ingredient_id, len(rare_combos)))

    interaction_df = pd.DataFrame(interaction_data, columns=['alcohol_id', 'ingredient_id', 'score'])
    print(f"총 {len(interaction_df)}개의 술-재료 상호작용이 저장되었습니다.")
    interaction_df.to_csv("./filtered_interactions.csv", index=False)

#build_interaction_matrix_compound_combination(n=3)

"""
결론 : Interaction Matrix 단계에서 화학물 특성 반영 어려움 -> 그냥 하드 코딩으로 가고 임베딩 단계에서 반영하자자
"""

