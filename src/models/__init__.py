"""GNN Models for Node Classification.

This package provides implementations of:
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE (Graph Sample and Aggregate)

Usage:
    from src.models import GCN, GAT, GraphSAGE
    
    model = GCN(num_features=1433, hidden_dim=64, num_classes=7)
"""

from .gcn import GCN, DeepGCN
from .gat import GAT, GATv2
from .graphsage import GraphSAGE, DeepGraphSAGE, GraphSAGEWithSampling

__all__ = [
    'GCN',
    'DeepGCN',
    'GAT',
    'GATv2',
    'GraphSAGE',
    'DeepGraphSAGE',
    'GraphSAGEWithSampling',
]

__version__ = '1.0.0'
