import torch
import torch_geometric
from torch import nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d


class Edge_Mask_Generator(nn.Module):
    def __init__(self, encoder, emb_dim, hidden_dim):
        super(Edge_Mask_Generator, self).__init__()
        self.encoder = encoder

        self.mlp_edge_model = Sequential(
            Linear(emb_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            Linear(hidden_dim, 2),
        )
        
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, g, device=None):
        edge_index = torch.stack(g.edges())
        if device is not None:
            edge_index = edge_index.to(device)
        _, node_emb = self.encoder(g, return_all_outputs=True)
        node_emb = torch.cat(node_emb, -1)
        src, dst = edge_index[0], edge_index[1]
        emb_src, emb_dst = node_emb[src], node_emb[dst]

        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)

        return edge_logits
        





if __name__ == "__main__":
    model = Edge_Mask_Generator()
    print("**")