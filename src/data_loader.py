"""Data loading and preprocessing for GNN node classification.

This module handles:
- Loading the Cora citation network dataset
- Preprocessing graph data
- Creating train/val/test splits
- Data augmentation and normalization
"""

import os
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
from typing import Tuple, Optional


class DataLoader:
    """Load and preprocess graph datasets.
    
    Supports:
    - Cora citation network
    - Citeseer citation network
    - Pubmed citation network
    
    Args:
        dataset_name: Name of dataset ('Cora', 'Citeseer', 'Pubmed')
        root: Root directory for data storage
        normalize_features: Whether to normalize node features
    """
    
    def __init__(
        self,
        dataset_name: str = 'Cora',
        root: str = 'data/',
        normalize_features: bool = True
    ):
        self.dataset_name = dataset_name
        self.root = root
        self.normalize_features = normalize_features
        
    def load_data(self):
        """Load dataset from PyTorch Geometric.
        
        Returns:
            data: PyG Data object with:
                - x: Node features [num_nodes, num_features]
                - edge_index: Graph connectivity [2, num_edges]
                - y: Node labels [num_nodes]
                - train_mask, val_mask, test_mask: Boolean masks
        """
        print(f"Loading {self.dataset_name} dataset...")
        
        # Load dataset with optional normalization
        transform = NormalizeFeatures() if self.normalize_features else None
        dataset = Planetoid(
            root=self.root,
            name=self.dataset_name,
            transform=transform
        )
        
        data = dataset[0]  # Get the single graph
        
        # Print dataset statistics
        self._print_statistics(data, dataset)
        
        return data, dataset
    
    def _print_statistics(self, data, dataset):
        """Print dataset statistics."""
        print(f"\nDataset: {self.dataset_name}")
        print(f"="*50)
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Number of features: {data.num_features}")
        print(f"Number of classes: {dataset.num_classes}")
        print(f"\nAverage node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")
        
        # Split statistics
        print(f"\nData splits:")
        print(f"Training nodes: {data.train_mask.sum().item()}")
        print(f"Validation nodes: {data.val_mask.sum().item()}")
        print(f"Test nodes: {data.test_mask.sum().item()}")
        print(f"="*50)


def load_cora(normalize: bool = True):
    """Convenience function to load Cora dataset.
    
    Args:
        normalize: Whether to normalize features
        
    Returns:
        data: PyG Data object
        dataset: PyG Dataset object
    """
    loader = DataLoader(dataset_name='Cora', normalize_features=normalize)
    return loader.load_data()


def load_citeseer(normalize: bool = True):
    """Convenience function to load Citeseer dataset.
    
    Args:
        normalize: Whether to normalize features
        
    Returns:
        data: PyG Data object
        dataset: PyG Dataset object
    """
    loader = DataLoader(dataset_name='Citeseer', normalize_features=normalize)
    return loader.load_data()


def load_pubmed(normalize: bool = True):
    """Convenience function to load Pubmed dataset.
    
    Args:
        normalize: Whether to normalize features
        
    Returns:
        data: PyG Data object
        dataset: PyG Dataset object
    """
    loader = DataLoader(dataset_name='Pubmed', normalize_features=normalize)
    return loader.load_data()


if __name__ == '__main__':
    # Test data loading
    print("Testing data loader...\n")
    
    # Load Cora dataset
    data, dataset = load_cora()
    
    # Verify data integrity
    assert data.x.shape[0] == data.num_nodes
    assert data.y.shape[0] == data.num_nodes
    assert data.edge_index.shape[0] == 2
    
    print("\n✓ Data loader test passed!")
