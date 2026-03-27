"""
gnn.py — GraphSAGE-based intrusion detection model.

Architecture: 3x SAGEConv (BatchNorm + Dropout) → MLP head
Supports node-level classification (primary) and graph-level classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
except ImportError:
    raise ImportError("pip install torch-geometric")


class GNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(hidden)])
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x   # (num_nodes, hidden)


class NodeClassifier(nn.Module):
    """Primary NIDS head: classify each flow as benign or attack."""

    def __init__(self, in_channels: int, hidden: int = 128, num_classes: int = 2,
                 layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden, layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, edge_index):
        return self.head(self.encoder(x, edge_index))

    def embed(self, x, edge_index):
        return self.encoder(x, edge_index)


class GraphClassifier(nn.Module):
    """Graph-level head: classify entire traffic window."""

    def __init__(self, in_channels: int, hidden: int = 128, num_classes: int = 2,
                 layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden, layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index)
        return self.head(global_mean_pool(z, batch))


def build_model(in_channels: int, task: str = "node", **kwargs) -> nn.Module:
    """Factory: task = 'node' | 'graph'"""
    cls = NodeClassifier if task == "node" else GraphClassifier
    return cls(in_channels, **kwargs)
