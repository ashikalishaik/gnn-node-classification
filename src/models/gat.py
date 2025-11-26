"""Graph Attention Network (GAT) Implementation.

Paper: Graph Attention Networks
Authors: Veličković et al. (2018)
ArXiv: https://arxiv.org/abs/1710.10903

GAT uses attention mechanism to weight neighbor contributions.
Key idea: Learn importance of neighbors via self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """Graph Attention Network for node classification.
    
    Architecture:
        Input -> GATConv (multi-head) -> ELU -> Dropout -> GATConv -> LogSoftmax
    
    Features:
    - Multi-head attention for robustness
    - Learnable attention weights
    - Better performance on heterophilic graphs
    
    Args:
        num_features: Number of input node features
        hidden_dim: Dimension per attention head in hidden layer
        num_classes: Number of output classes
        heads: Number of attention heads in first layer (default: 8)
        output_heads: Number of attention heads in output layer (default: 1)
        dropout: Dropout probability (default: 0.6)
        concat: Whether to concatenate or average attention heads
    
    Example:
        >>> model = GAT(num_features=1433, hidden_dim=8, num_classes=7, heads=8)
        >>> out = model(data.x, data.edge_index)
        >>> # out.shape = [num_nodes, num_classes]
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 8,
        num_classes: int = 7,
        heads: int = 8,
        output_heads: int = 1,
        dropout: float = 0.6,
        concat: bool = True
    ):
        super(GAT, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.heads = heads
        self.output_heads = output_heads
        self.dropout = dropout
        self.concat = concat
        
        # First GAT layer with multiple heads
        self.conv1 = GATConv(
            num_features,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=concat
        )
        
        # Calculate input dim for second layer
        # If concat=True: heads * hidden_dim, else: hidden_dim
        second_input_dim = heads * hidden_dim if concat else hidden_dim
        
        # Second GAT layer
        self.conv2 = GATConv(
            second_input_dim,
            num_classes,
            heads=output_heads,
            dropout=dropout,
            concat=False  # Average for output
        )
        
    def forward(self, x, edge_index, return_attention_weights=False):
        """Forward pass through GAT.
        
        Args:
            x: Node feature matrix [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            If return_attention_weights=False:
                Log probabilities [num_nodes, num_classes]
            Else:
                (log_probs, (edge_index, attention_weights))
        """
        # First GAT layer with multi-head attention
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Second GAT layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention_weights:
            x, (edge_index, attention_weights) = self.conv2(
                x, edge_index, return_attention_weights=True
            )
            return F.log_softmax(x, dim=1), (edge_index, attention_weights)
        else:
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index):
        """Get node embeddings from hidden layer.
        
        Args:
            x: Node feature matrix
            edge_index: Graph connectivity
            
        Returns:
            Node embeddings [num_nodes, heads * hidden_dim] (if concat)
            or [num_nodes, hidden_dim] (if average)
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        return x
    
    def get_attention_weights(self, x, edge_index):
        """Extract attention weights for visualization.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            (edge_index, attention_weights): Attention for each edge
        """
        self.eval()
        with torch.no_grad():
            _, attention_data = self.forward(
                x, edge_index, return_attention_weights=True
            )
        return attention_data
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_features={self.num_features}, '
                f'hidden_dim={self.hidden_dim}, '
                f'heads={self.heads}, '
                f'num_classes={self.num_classes}, '
                f'params={self.count_parameters()})')


class GATv2(nn.Module):
    """GATv2: Improved Graph Attention Network.
    
    Uses dynamic attention (GATv2Conv) for better expressiveness.
    Paper: How Attentive are Graph Attention Networks? (Brody et al., 2021)
    
    Args:
        num_features: Number of input features
        hidden_dim: Hidden dimension per head
        num_classes: Number of classes
        heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 8,
        num_classes: int = 7,
        heads: int = 8,
        dropout: float = 0.6
    ):
        super(GATv2, self).__init__()
        
        try:
            from torch_geometric.nn import GATv2Conv
            
            self.conv1 = GATv2Conv(
                num_features,
                hidden_dim,
                heads=heads,
                dropout=dropout,
                concat=True
            )
            
            self.conv2 = GATv2Conv(
                heads * hidden_dim,
                num_classes,
                heads=1,
                dropout=dropout,
                concat=False
            )
            
            self.dropout = dropout
            self.available = True
            
        except ImportError:
            print("Warning: GATv2Conv not available in your PyG version")
            self.available = False
    
    def forward(self, x, edge_index):
        """Forward pass through GATv2."""
        if not self.available:
            raise RuntimeError("GATv2Conv not available")
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    # Test GAT model
    print("Testing GAT model...\n")
    
    # Create dummy data
    num_nodes = 100
    num_features = 1433
    num_classes = 7
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    # Initialize model
    model = GAT(num_features=num_features, num_classes=num_classes)
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        print(f"\nOutput shape: {out.shape}")
        print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    
    # Test attention weights
    edge_idx, attention = model.get_attention_weights(x, edge_index)
    print(f"\nAttention weights shape: {attention.shape}")
    print(f"Attention range: [{attention.min():.3f}, {attention.max():.3f}]")
    
    # Test embeddings
    embeddings = model.get_embeddings(x, edge_index)
    print(f"Embedding shape: {embeddings.shape}")
    
    print("\n✓ GAT model test passed!")
