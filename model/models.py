import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv, NNConv, MessagePassing
from torch_geometric.utils import softmax

"""
all_node_emb = GNN(edge_index)
liquor_emb = all_node_emb[liquor_ids]
ingredient_emb = all_node_emb[ingredient_ids]
score = MLP(concat(liquor_emb, ingredient_emb))
loss = BCE(score, label)
loss.backward()
"""

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, num_nodes=8298, num_relations=2, emb_size=128, hidden_layers=[128, 64, 32], user_init=None, item_init=None):
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
        self.num_nodes = num_nodes
        
        self.embedding = nn.Embedding(num_nodes, emb_size) # GNN에서 사용될 노드 임베딩

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        
        # RGNN
        """self.rgcn1 = RGCNConv(emb_size, emb_size, num_relations)
        self.rgcn2 = RGCNConv(emb_size, emb_size, num_relations)
        self.rgcn3 = RGCNConv(emb_size, emb_size, num_relations)"""
        
        self.wrgcn = WeightedRGCNConv(emb_size, emb_size, num_relations) # GNN layer
        self.wrgcn2 = WeightedRGCNConv(emb_size, emb_size, num_relations)
        self.wrgcn3 = WeightedRGCNConv(emb_size, emb_size, num_relations)
        
        # GNN
        """self.conv1 = GCNConv(emb_size, emb_size) # GNN layer
        self.conv2 = GCNConv(emb_size, emb_size)
        self.conv3 = GCNConv(emb_size, emb_size)"""
        
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

    def forward(self, user_indices, item_indices, edge_index, edge_type, edge_weight=None):
        """
            user_indices :   술 노드의 인덱스
            item_indices :   음식 노드의 인덱스
            edge_index   :   GNN에서 사용할 edge_index
            edge_weight  :   GNN에서 사용할 edge_weight (default: None)
        """
        # RGCN 기반 임베딩
        x = self.embedding(torch.arange(self.num_nodes, device=edge_index.device))
        
        """x = self.rgcn1(x, edge_index, edge_type)
        x = self.rgcn2(x, edge_index, edge_type)
        x = self.rgcn3(x, edge_index, edge_type)"""
        
        x = self.wrgcn(x, edge_index, edge_type, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.norm1(x)
        x = self.wrgcn2(x, edge_index, edge_type, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.norm2(x)
        x = self.wrgcn3(x, edge_index, edge_type, edge_weight)
        
        # GNN 기반 임베딩
        """x = self.conv1(x, edge_index, edge_weight)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.conv3(x, edge_index, edge_weight)"""

        # GNN 결과 슬라이싱
        gmf_user_emb = x[user_indices]
        gmf_item_emb = x[item_indices]
        mlp_user_emb = x[user_indices]
        mlp_item_emb = x[item_indices]

        gmf_user_emb = F.normalize(gmf_user_emb, dim=-1)
        gmf_item_emb = F.normalize(gmf_item_emb, dim=-1)

        # GMF
        gmf_output = gmf_user_emb * gmf_item_emb

        # MLP
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # GMF + MLP
        """
        "... we concatenate the learned representations from GMF and MLP, and feed them into a final prediction layer."
        GMF + MLP 둘이 성질이 다르기 때문에 곱하거나 평균내지 않고 그냥 나란히 붙인다
        """
        final_input = torch.cat([gmf_output, mlp_output], dim=-1)
        logits = self.output_layer(final_input)

        return torch.sigmoid(logits).squeeze()


class WeightedRGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, aggr='add', bias=True):
        super().__init__(aggr=aggr)
        self.num_relations = num_relations
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 각 관계(r)마다 weight matrix 생성
        self.rel_lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False) for _ in range(num_relations)
        ])

        self.root = nn.Linear(in_channels, out_channels, bias=False)  # 자기 자신
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        self.root.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        edge_type: [num_edges]
        edge_weight: [num_edges] or None
        """
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        return self.propagate(edge_index, x=x, edge_type=edge_type, edge_weight=edge_weight)

    def message(self, x_j, edge_type, edge_weight):
        """
        x_j: source node features [num_edges, in_channels]
        edge_type: edge types [num_edges]
        edge_weight: edge weights [num_edges]
        """

        # 각 관계에 따라 다른 transformation
        out = torch.zeros(x_j.size(0), self.out_channels, device=x_j.device)

        for r in range(self.num_relations):
            mask = edge_type == r
            if mask.sum() > 0:
                transformed = self.rel_lins[r](x_j[mask])
                out[mask] = edge_weight[mask].unsqueeze(-1) * transformed

        return out

    def update(self, aggr_out, x):
        out = aggr_out + self.root(x)
        if self.bias is not None:
            out = out + self.bias
        return out