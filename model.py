import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pymetis

def metis_cluster(edge_index, num_nodes, num_clusters):
    """
    Partition a single graph into num_clusters using pymetis.
    edge_index: LongTensor of shape [2, E]
    num_nodes: int
    num_clusters: int
    Returns: parts tensor of shape [num_nodes], each entry in [0..num_clusters-1]
    """
    # build adjacency list for pymetis
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        adj[u].append(v)
        adj[v].append(u)
    _, parts = pymetis.part_graph(num_clusters, adjacency=adj)
    return torch.tensor(parts, dtype=torch.long)

def hierarchical_pool(x, parts, num_clusters):
    """
    For batch_size=1 graphs: pool node embeddings per cluster.
    x: [N, H], batch: [N], parts: [N]
    Returns: Tensor of shape [num_clusters, H]
    """
    H = x.size(1)
    pooled = []
    for cid in range(num_clusters):
        mask = (parts == cid)
        if mask.sum() > 0:
            pooled.append(x[mask].mean(dim=0))
        else:
            pooled.append(torch.zeros(H, device=x.device))
    return torch.stack(pooled, dim=0)

class GraphSAGEEncoder(nn.Module):
    def __init__(self, max_degree, node_embed_dim, hidden_dim, num_layers, num_clusters, dropout=0.2):
        super().__init__()
        self.node_embedding = nn.Embedding(max_degree + 1, node_embed_dim)
        self.convs = nn.ModuleList()
        in_dim = node_embed_dim
        for _ in range(num_layers):
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.num_clusters = num_clusters
        self.dropout = nn.Dropout(p=dropout)
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])


    def forward(self, data):
        x = self.node_embedding(data.deg_idx)
        edge_index = data.edge_index
        batch = data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        out = []
        num_graphs = batch.max().item() + 1
        for i in range(num_graphs):
            node_mask = (batch == i)
            x_i = x[node_mask]
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            edge_index_i = edge_index[:, edge_mask]
            edge_index_i = edge_index_i - edge_index_i.min()

            parts = metis_cluster(edge_index_i, x_i.size(0), self.num_clusters)
            pooled = hierarchical_pool(x_i, parts, self.num_clusters)
            out.append(pooled)

        return torch.stack(out, dim=0)  # [B, num_clusters, hidden_dim]
    
class Projector(nn.Module):
    def __init__(self, hidden_dim, llm_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, llm_dim)

    def forward(self, x):
        # x: Tensor of shape [batch_size, num_clusters, hidden_dim]
        B, K, H = x.shape
        # Flatten batch and cluster dims for linear layer
        x_flat = x.reshape(-1, H)             # [B*K, H]
        y_flat = self.linear(x_flat)          # [B*K, llm_dim]
        # Restore batch and cluster dims
        return y_flat.view(B, K, -1)          # [B, K, llm_dim

