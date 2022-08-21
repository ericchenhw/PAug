from .edge_mask_generator import Edge_Mask_Generator

import torch
import torch_geometric
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import time

class Graph_Augmentation_Model(nn.Module):
    def __init__(self, encoder, emb_dim, hidden_dim, temperature, bias=1e-4):
        super(Graph_Augmentation_Model, self).__init__()
        self.edge_mask_generator = Edge_Mask_Generator(encoder, emb_dim, hidden_dim)
        self.temperature = temperature
        self.bias = bias

    def forward(self, g, return_soft=False, device=None):
        edge_logits = self.edge_mask_generator(g)
        if not return_soft:
            edge_weights = F.gumbel_softmax(edge_logits, hard=True, tau=self.temperature, dim=-1)
            edge_weights = edge_weights[:, 1]
        else:
            edge_weights = F.gumbel_softmax(edge_logits, hard=False, tau=self.temperature, dim=-1)
            edge_weights = edge_weights[:, 1]

        return edge_weights
    

