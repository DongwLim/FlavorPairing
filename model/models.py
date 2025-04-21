import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

"""
all_node_emb = GNN(edge_index)
liquor_emb = all_node_emb[liquor_ids]
ingredient_emb = all_node_emb[ingredient_ids]
score = MLP(concat(liquor_emb, ingredient_emb))
loss = BCE(score, label)
loss.backward()
"""

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, num_nodes=8298, num_relations=0, emb_size=128, hidden_layers=[128, 64, 32], user_init=None, item_init=None):
        super(NeuralCF, self).__init__()
        """
            num_users       :   술 노드의 개수
            num_items       :   음식 노드의 개수
            num_nodes       :   전체 노드의 개수
            num_relations   :   관계(edge_type)의 개수
            emb_size        :   벡터 차원 크기
            hidden_layer    :   MLP
            user_init       :   술 초기 임베딩
            item_init       :   음식 초기 임베딩
        """
        """
            GNN 구현 완료 
            CSP_Aggregation 미구현
        """
        
        self.embedding = nn.Embedding(num_nodes, emb_size) # GNN에서 사용될 노드 임베딩
        
        # GNN
        self.conv1 = GCNConv(emb_size, emb_size) # GNN layer
        self.conv2 = GCNConv(emb_size, emb_size)
        
        self.bn1 = nn.BatchNorm1d(emb_size) # Batch Normalization
        self.bn2 = nn.BatchNorm1d(emb_size)
        
        # GMF 
        self.user_emb_gmf = nn.Embedding(num_users, emb_size)
        self.item_emb_gmf = nn.Embedding(num_items, emb_size)

        # MLP 
        self.user_emb_mlp = nn.Embedding(num_users, emb_size)
        self.item_emb_mlp = nn.Embedding(num_items, emb_size)

        """if user_init is not None:
            self.user_emb_mlp.weight.data.copy_(user_init.float())
            self.user_emb_gmf.weight.data.copy_(user_init.float())
        if item_init is not None:
            self.item_emb_mlp.weight.data.copy_(item_init.float())
            self.item_emb_gmf.weight.data.copy_(item_init.float())


        nn.init.kaiming_uniform_(self.user_emb_mlp.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.item_emb_mlp.weight, nonlinearity="relu")"""

        #nn.init.orthogonal_(self.user_emb_mlp.weight)
        #nn.init.orthogonal_(self.item_emb_mlp.weight)

        layers = []
        input_size = emb_size * 2
        for h in hidden_layers:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2)) # Dropout 추가
            input_size = h
        self.mlp = nn.Sequential(*layers)

        # 최종 결과 출력층 
        self.output_layer = nn.Linear(hidden_layers[-1] + emb_size, 1)

    def forward(self, user_indices, item_indices, edge_index, edge_weight=None):
        """
            user_indices :   술 노드의 인덱스
            item_indices :   음식 노드의 인덱스
            edge_index   :   GNN에서 사용할 edge_index
            edge_weight  :   GNN에서 사용할 edge_weight (default: None)
        """
        # GNN
        x = self.embedding.weight
        x = self.conv1(x, edge_index, edge_weight)
        x = self.conv2(x, edge_index, edge_weight)
        
        liquor_emb = x[user_indices]
        ingredient_emb = x[item_indices]
        
        # GMF
        gmf_user = liquor_emb
        gmf_item = ingredient_emb
        gmf_output = gmf_user * gmf_item

        # MLP
        mlp_user = liquor_emb
        mlp_item = ingredient_emb
        mlp_input = torch.cat((mlp_user, mlp_item), dim=-1)
        mlp_output = self.mlp(mlp_input)

        # GMF + MLP
        """
        "... we concatenate the learned representations from GMF and MLP, and feed them into a final prediction layer."
        GMF + MLP 둘이 성질이 다르기 때문에 곱하거나 평균내지 않고 그냥 나란히 붙인다
        """
        final_input = torch.cat((gmf_output, mlp_output), dim=-1)
        logits = self.output_layer(final_input)

        return torch.sigmoid(logits).squeeze()
