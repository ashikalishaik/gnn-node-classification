"""GraphSAGE (Graph Sample and Aggregate) Implementation.

Paper: Inductive Representation Learning on Large Graphs
Authors: Hamilton et al. (2017)
ArXiv: https://arxiv.org/abs/1706.02216

GraphSAGE samples and aggregates features from node neighborhoods.
Key idea: Inductive learning - can generalize to unseen nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """GraphSAGE for node classification.
    
    Architecture:
        Input -> SAGEConv -> ReLU -> Dropout -> SAGEConv -> LogSoftmax
    
    Features:
    - Inductive learning (generalizes to new nodes)
    - Neighborhood sampling for scalability  
    - Multiple aggregation functions (mean, max, LSTM)
    
    Args:
        num_features: Number of input node features
        hidden_dim: Dimension of hidden layer
        num_classes: Number of output classes
        dropout: Dropout probability (default: 0.5)
        aggregator: Aggregation function ('mean', 'max', 'lstm')
    
    Example:
        >>> model = GraphSAGE(num_features=1433, hidden_dim=64, num_classes=7)
        >>> out = model(data.x, data.edge_index)
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_classes: int = 7,
        dropout: float = 0.5,
        aggregator: str = 'mean'
    ):
        super(GraphSAGE, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.aggregator = aggregator
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(
            num_features,
            hidden_dim,
            aggr=aggregator
        )
        
        self.conv2 = SAGEConv(
            hidden_dim,
            num_classes,
            aggr=aggregator
        )
        
    def forward(self, x, edge_index):
        """Forward pass through GraphSAGE.
        
        Args:
            x: Node feature matrix [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Log probabilities for each node [num_nodes, num_classes]
        """
        # First SAGE layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second SAGE layer
        x = self.conv2(x, edge_index)
        
        # Log softmax for classification
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index):
        """Get node embeddings from hidden layer.
        
        Args:
            x: Node feature matrix
            edge_index: Graph connectivity
            
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_features={self.num_features}, '
                f'hidden_dim={self.hidden_dim}, '
                f'num_classes={self.num_classes}, '
                f'aggregator={self.aggregator}, '
                f'params={self.count_parameters()})')


class DeepGraphSAGE(nn.Module):
    """Deeper GraphSAGE with multiple layers.
    
    For scenarios requiring more expressive models.
    
    Args:
        num_features: Number of input features
        hidden_dim: Hidden dimension
        num_classes: Number of classes
        num_layers: Number of SAGE layers (default: 3)
        dropout: Dropout probability
        aggregator: Aggregation function
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_classes: int = 7,
        num_layers: int = 3,
        dropout: float = 0.5,
        aggregator: str = 'mean'
    ):
        super(DeepGraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.input_conv = SAGEConv(num_features, hidden_dim, aggr=aggregator)
        
        # Hidden layers
        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim, aggr=aggregator)
            for _ in range(num_layers - 2)
        ])
        
        # Output layer
        self.output_conv = SAGEConv(hidden_dim, num_classes, aggr=aggregator)
        
    def forward(self, x, edge_index):
        """Forward pass through deep GraphSAGE."""
        # Input layer
        x = self.input_conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.output_conv(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSAGEWithSampling(nn.Module):
    """GraphSAGE with neighborhood sampling for large graphs.
    
    Samples fixed-size neighborhoods for scalability.
    Useful for mini-batch training on large graphs.
    
    Args:
        num_features: Number of input features
        hidden_dim: Hidden dimension
        num_classes: Number of classes
        num_neighbors: Number of neighbors to sample per layer
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_classes: int = 7,
        num_neighbors: list = [25, 10],
        dropout: float = 0.5
    ):
        super(GraphSAGEWithSampling, self).__init__()
        
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        
        # SAGE layers
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        
    def forward(self, x, edge_index):
        """Forward with neighborhood sampling.
        
        Note: For actual sampling, use NeighborLoader from PyG.
        This is a simplified version for full-batch training.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    # Test GraphSAGE model
    print("Testing GraphSAGE model...\n")
    
    # Create dummy data
    num_nodes = 100
    num_features = 1433
    num_classes = 7
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    # Test different aggregators
    for aggregator in ['mean', 'max']:
        print(f"\nTesting with {aggregator} aggregation:")
        model = GraphSAGE(
            num_features=num_features,
            num_classes=num_classes,
            aggregator=aggregator
        )
        print(model)
        print(f"Total parameters: {model.count_parameters():,}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            print(f"Output shape: {out.shape}")
            print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    
    # Test embeddings
    embeddings = model.get_embeddings(x, edge_index)
    print(f"\nEmbedding shape: {embeddings.shape}")
    
    print("\n✓ GraphSAGE model test passed!")
