"""Graph Convolutional Network (GCN) Implementation.

Paper: Semi-Supervised Classification with Graph Convolutional Networks
Authors: Kipf & Welling (2017)
ArXiv: https://arxiv.org/abs/1609.02907

GCN applies spectral graph convolutions for node classification.
Key idea: Aggregate neighbor features via normalized adjacency matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Graph Convolutional Network for node classification.
    
    Architecture:
        Input -> GCNConv -> ReLU -> Dropout -> GCNConv -> LogSoftmax
    
    Args:
        num_features: Number of input node features
        hidden_dim: Dimension of hidden layer
        num_classes: Number of output classes
        dropout: Dropout probability (default: 0.5)
    
    Example:
        >>> model = GCN(num_features=1433, hidden_dim=64, num_classes=7)
        >>> out = model(data.x, data.edge_index)
        >>> loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_classes: int = 7,
        dropout: float = 0.5
    ):
        super(GCN, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        
        # GCN layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        
    def forward(self, x, edge_index):
        """Forward pass through GCN.
        
        Args:
            x: Node feature matrix [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Log probabilities for each node [num_nodes, num_classes]
        """
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        
        # Log softmax for classification
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index):
        """Get node embeddings from hidden layer.
        
        Useful for visualization and analysis.
        
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
                f'params={self.count_parameters()})')


class DeepGCN(nn.Module):
    """Deeper GCN with multiple layers and residual connections.
    
    For scenarios requiring more expressive models.
    
    Args:
        num_features: Number of input features
        hidden_dim: Hidden dimension
        num_classes: Number of classes
        num_layers: Number of GCN layers (default: 3)
        dropout: Dropout probability
        residual: Whether to use residual connections
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_classes: int = 7,
        num_layers: int = 3,
        dropout: float = 0.5,
        residual: bool = True
    ):
        super(DeepGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        
        # Input projection
        self.input_conv = GCNConv(num_features, hidden_dim)
        
        # Hidden layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_layers - 2)
        ])
        
        # Output layer
        self.output_conv = GCNConv(hidden_dim, num_classes)
        
    def forward(self, x, edge_index):
        """Forward pass through deep GCN."""
        # Input layer
        x = self.input_conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers with residual connections
        for conv in self.convs:
            residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.residual:
                x = x + residual  # Skip connection
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.output_conv(x, edge_index)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    # Test GCN model
    print("Testing GCN model...\n")
    
    # Create dummy data
    num_nodes = 100
    num_features = 1433
    num_classes = 7
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    # Initialize model
    model = GCN(num_features=num_features, num_classes=num_classes)
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        print(f"\nOutput shape: {out.shape}")
        print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    
    # Test embeddings
    embeddings = model.get_embeddings(x, edge_index)
    print(f"Embedding shape: {embeddings.shape}")
    
    print("\n✓ GCN model test passed!")
