import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    def __init__(self, input_dim, hyperparams):
        super(GNN, self).__init__()
        self.hyperparams = hyperparams
        self.input_dim = input_dim

        self.conv_1 = GCNConv(input_dim, self.hyperparams['hidden_gnn'])
        self.conv_2 = GCNConv(self.hyperparams['hidden_gnn'], self.hyperparams['latent_dim'])

    def get_embedding(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv_1(x, edge_index)
        x = F.relu(x)
        embed = self.conv_2(x, edge_index)
        return embed
        
    
class DQN(GNN):
    def __init__(self, input_dim, hyperparams):
        super().__init__(input_dim, hyperparams)
        embed_dim = self.hyperparams['latent_dim']
        self.linear_1 = nn.Linear(embed_dim, self.hyperparams['hidden_dqn'])
        self.linear_out = nn.Linear(self.hyperparams['hidden_dqn'], 1)

    def forward(self, data):
        embed = self.get_embedding(data)
        x = self.linear_1(embed)
        x = F.relu(x)
        x = self.linear_out(x)
        return x
