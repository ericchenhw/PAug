import torch
from torch_scatter import scatter
import torch.nn.functional as F

def cal_reg(g, edge_weight, threshold, squared=False, device=None):
    """
    calculate the perturbation bound
    """
    edge_weight = edge_weight
    edge_batch_id = []
    for idx, batch_size in enumerate(g.batch_num_edges):
        edge_batch_id.extend([idx] * batch_size)
    edge_batch_id = torch.tensor(edge_batch_id, dtype=torch.int64)
    if device is not None:
        edge_batch_id = edge_batch_id.to(device)
    
    if squared:
        scale = 1 - threshold
        reg = torch.pow((F.relu(1 - scatter(edge_weight, edge_batch_id, reduce="mean") - threshold) / scale), 2).mean()
    else:
        reg = F.relu(1 - scatter(edge_weight, edge_batch_id, reduce="mean") - threshold).mean()
    return reg
