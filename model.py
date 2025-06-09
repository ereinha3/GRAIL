import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GINConv, GATConv
from transformers import StoppingCriteria

class SAGEEncoder(nn.Module):
    def __init__(self, max_degree, node_embed_dim, hidden_dim,
                 num_layers, num_clusters, output_dim,
                 dropout=0.2, num_heads=1):
        super().__init__()
        self.node_embedding = nn.Embedding(max_degree + 1, node_embed_dim)

        self.convs  = nn.ModuleList()
        self.norms  = nn.ModuleList()
        in_dim       = node_embed_dim
        for _ in range(num_layers):
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        self.dropout      = nn.Dropout(dropout)
        self.num_clusters = num_clusters
        self.num_heads    = num_heads

        # if num_heads > 1, we learn num_heads × num_clusters scores per node,
        # then collapse heads by mean.
        self.attn_proj   = nn.Linear(hidden_dim,
                                     num_heads * num_clusters)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def __str__(self):
        return "SAGEEncoder"

    def forward(self, data):
        # 1) embed + GNN
        x           = self.node_embedding(data.deg_idx)  # [N, D_in]
        edge_index  = data.edge_index
        batch       = data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)        # [N, hidden_dim]
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        # 2) node-to-cluster attention scores
        scores = self.attn_proj(x)  # [N, num_heads * K]
        scores = scores.view(-1, self.num_heads, self.num_clusters)  # [N, H, K]

        # collapse heads if needed:
        if self.num_heads > 1:
            scores = scores.mean(dim=1)  # [N, K]
        else:
            scores = scores.squeeze(1)   # [N, K]

        # 3) per-graph, do a nodes-of-that-graph softmax & weighted sum
        B = int(batch.max().item()) + 1
        cluster_embs = []
        for i in range(B):
            mask   = (batch == i)
            x_i    = x[mask]    # [n_i, hidden_dim]
            s_i    = scores[mask]  # [n_i, K]

            # normalize across nodes **within** this graph for each cluster
            w_i = F.softmax(s_i, dim=0)  # [n_i, K]

            # weighted sum: (K x n_i) @ (n_i x hidden_dim) → (K x hidden_dim)
            c_i = w_i.transpose(0, 1) @ x_i  # [K, hidden_dim]

            # project into LLM space
            c_i = self.output_proj(c_i)      # [K, output_dim]
            cluster_embs.append(c_i)

        # [B, K, output_dim]
        return torch.stack(cluster_embs, dim=0)
    
class GINEncoder(nn.Module):
    def __init__(self, max_degree, node_embed_dim, hidden_dim,
                 num_layers, num_clusters, output_dim,
                 dropout=0.2, num_heads=1):
        super().__init__()
        self.node_embedding = nn.Embedding(max_degree + 1, node_embed_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = node_embed_dim
        # build MLP for GINConv
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.attn_proj = nn.Linear(hidden_dim, num_heads * num_clusters)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def __str__(self):
        return "GINEncoder"

    def forward(self, data):
        x = self.node_embedding(data.deg_idx)
        edge_index = data.edge_index
        batch = data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        # attention pooling as before
        scores = self.attn_proj(x).view(-1, self.num_heads, self.num_clusters)
        if self.num_heads > 1:
            scores = scores.mean(dim=1)
        else:
            scores = scores.squeeze(1)
        B = int(batch.max().item()) + 1
        cluster_embs = []
        for i in range(B):
            mask = (batch == i)
            x_i = x[mask]
            s_i = scores[mask]
            w_i = F.softmax(s_i, dim=0)
            c_i = w_i.transpose(0,1) @ x_i
            c_i = self.output_proj(c_i)
            cluster_embs.append(c_i)
        return torch.stack(cluster_embs, dim=0)

class GATEncoder(nn.Module):
    def __init__(self, max_degree, node_embed_dim, hidden_dim,
                 num_layers, num_clusters, output_dim,
                 heads=4, dropout=0.2, num_heads=1):
        super().__init__()
        self.node_embedding = nn.Embedding(max_degree + 1, node_embed_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = node_embed_dim
        for _ in range(num_layers):
            self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.attn_proj = nn.Linear(hidden_dim, num_heads * num_clusters)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def __str__(self):
        return "GATEncoder"

    def forward(self, data):
        x = self.node_embedding(data.deg_idx)
        edge_index = data.edge_index
        batch = data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)
        scores = self.attn_proj(x).view(-1, self.num_heads, self.num_clusters)
        if self.num_heads > 1:
            scores = scores.mean(dim=1)
        else:
            scores = scores.squeeze(1)
        B = int(batch.max().item()) + 1
        cluster_embs = []
        for i in range(B):
            mask = (batch == i)
            x_i = x[mask]
            s_i = scores[mask]
            w_i = F.softmax(s_i, dim=0)
            c_i = w_i.transpose(0,1) @ x_i
            c_i = self.output_proj(c_i)
            cluster_embs.append(c_i)
        return torch.stack(cluster_embs, dim=0)  # [B, K, output_dim]


class StopOnBracket(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores=None, **kwargs):
        # if last token decodes to ']', stop
        return self.tokenizer.decode([input_ids[0, -1].item()]).strip() == "]"