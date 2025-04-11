import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=64, hidden_layers=[128, 64, 32], user_init=None, item_init=None):
        super(NeuralCF, self).__init__()
        
        # GMF 
        self.user_emb_gmf = nn.Embedding(num_users, emb_size)
        self.item_emb_gmf = nn.Embedding(num_items, emb_size)

        # MLP 
        self.user_emb_mlp = nn.Embedding(num_users, emb_size)
        self.item_emb_mlp = nn.Embedding(num_items, emb_size)

        if user_init is not None:
            self.user_emb_mlp.weight.data.copy_(user_init)
            self.user_emb_gmf.weight.data.copy_(user_init)
        if item_init is not None:
            self.item_emb_mlp.weight.data.copy_(item_init)
            self.item_emb_gmf.weight.data.copy_(item_init)

        layers = []
        input_size = emb_size * 2
        for h in hidden_layers:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            input_size = h
        self.mlp = nn.Sequential(*layers)

        # 최종 결과 출력층 
        self.output_layer = nn.Linear(hidden_layers[-1] + emb_size, 1)

    def forward(self, user_indices, item_indices):
        # GMF
        gmf_user = self.user_emb_gmf(user_indices)
        gmf_item = self.item_emb_gmf(item_indices)
        gmf_output = gmf_user * gmf_item

        # MLP
        mlp_user = self.user_emb_mlp(user_indices)
        mlp_item = self.item_emb_mlp(item_indices)
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
