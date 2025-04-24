import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import PCA
from model.dataset import map_graph_nodes, edges_index

def node_data_add_liquor():
    df =  pd.read_csv("./dataset/Hub_Nodes.csv")

    al = pd.read_csv("./dataset/Alcohol-Related_Hub_Nodes.csv")

    al['node_type'] = 'liquor'

    #print(df.head(5))
    #print(al.head(5))

    for idx, row in df.iterrows():
        if row['name'] in al['name'].values:
            df.at[idx, 'node_type'] = 'liquor'

    df.to_csv("./dataset/Hub_Nodes.csv", index=False)
    al.to_csv("./dataset/Alcohol-Related_Hub_Nodes.csv", index=False)

def get_csp():
    with open("./dataset/flavor diffusion/node2fp_revised_1120.pickle", "rb") as f:
        binary_dict = pickle.load(f)

    valid_node_vectors = {}

    for idx in range(8748): 
        try:
            vector = list(binary_dict[idx])  
            valid_node_vectors[idx] = vector
        except:
            continue  

    # 결과 확인
    print(f"임베딩이 존재하는 노드 수: {len(valid_node_vectors)}")
    sample_items = list(valid_node_vectors.items())[:3]
    for node_id, vec in sample_items:
        print(f"Node {node_id}: {vec[:5]} ...")  

    #with open("./dataset/compound_embeddings.pkl", "wb") as f:
    #    pickle.dump(valid_node_vectors, f)


"""df = pd.read_csv("./Unique_Target_Nodes_from_ingr-fcomp_Edges.csv")
df = df.drop("Unnamed: 0", axis=1)

df.to_csv("./Unique_Target_Nodes_from_ingr-fcomp_Edges.csv", index=False)"""

def get_csp_with_dim_reduction(output_dim=128):
    # 파일 경로 설정
    node_file = "./dataset/Hub_Nodes.csv"
    edge_file = "./dataset/Hub_Edges.csv"
    original_embedding_file = "./dataset/flavor diffusion/node2fp_revised_1120.pickle"
    output_file = "./dataset/compound_embeddings_filtered.pkl"

    # 노드와 엣지 불러오기
    nodes_df = pd.read_csv(node_file)
    edges_df = pd.read_csv(edge_file)

    # compound 노드 ID 중 실제로 엣지에서 target으로 등장하는 것만 추출
    compound_ids = set(nodes_df[nodes_df['node_type'] == 'compound']['node_id'])
    compound_ids_from_edges = set(edges_df['target']).intersection(compound_ids)

    # 원본 임베딩 불러오기
    with open(original_embedding_file, "rb") as f:
        full_embedding_dict = pickle.load(f)

    # 유효한 벡터만 추출
    valid_vectors = []
    valid_ids = []
    for cid in compound_ids_from_edges:
        if cid in full_embedding_dict:
            vec = list(full_embedding_dict[cid])
            valid_vectors.append(vec)
            valid_ids.append(cid)

    print(f"유효 compound 개수: {len(valid_vectors)}")

    # PCA로 차원 축소
    X = np.array(valid_vectors)
    print(f"원본 차원: {X.shape[1]} → 축소 차원: {output_dim}")
    pca = PCA(n_components=output_dim)
    X_reduced = pca.fit_transform(X)

    filtered_dict = {cid: X_reduced[i].tolist() for i, cid in enumerate(valid_ids)}

    with open(output_file, "wb") as f:
        pickle.dump(filtered_dict, f)

    print(f"저장 완료: {output_file}")

if __name__ == "__main__":
    get_csp()
    
    map = map_graph_nodes()
    
    with open("./dataset/flavor diffusion/node2fp_revised_1120.pickle", "rb") as f:
        binary_dict = pickle.load(f)
        
    nodes_df = pd.read_csv("./dataset/nodes_191120_updated.csv")
    edges_df = pd.read_csv("./dataset/edges_191120_updated.csv")
    
    