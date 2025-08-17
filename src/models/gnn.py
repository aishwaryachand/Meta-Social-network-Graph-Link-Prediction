"""
Simple GraphSAGE link prediction model (PyTorch Geometric).
- Encoder: SAGEConv stacks
- Decoder: dot product (score = e_u · e_v)
"""
from __future__ import annotations
import torch
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from typing import Tuple


class GraphSAGEEncoder(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, edge_index):
        h = self.emb(x)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout(h)
        return h


class DotDecoder(nn.Module):
    def forward(self, h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        return (h_u * h_v).sum(dim=-1)


class GraphSageLinkPredictor(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = GraphSAGEEncoder(num_nodes, hidden_dim, num_layers, dropout)
        self.decoder = DotDecoder()

    def node_embeddings(self, x, edge_index) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def score_pairs(self, emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        u = emb[pairs[:, 0]]
        v = emb[pairs[:, 1]]
        return self.decoder(u, v)

    def forward(self, x, edge_index, pairs: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x, edge_index)
        return self.score_pairs(emb, pairs)


def build_pyg_data(num_nodes: int, edge_index: torch.Tensor) -> Data:
    x = torch.arange(num_nodes, dtype=torch.long)  # node index features for embedding lookup
    return Data(x=x, edge_index=edge_index)
