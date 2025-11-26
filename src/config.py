"""Configuration file for GNN project.

Centralizes all hyperparameters and settings for reproducibility.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Model selection
    model_name: str = 'gcn'  # 'gcn', 'gat', 'graphsage'
    
    # Common hyperparameters
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    
    # GAT-specific
    num_heads: int = 8
    head_concat: bool = True
    
    # GraphSAGE-specific
    aggregator: str = 'mean'  # 'mean', 'max', 'lstm'
    normalize: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimization
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    
    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Reproducibility
    seed: int = 42
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = 'gnn-node-classification'
    wandb_entity: Optional[str] = None
    
    # Checkpointing
    save_dir: str = 'checkpoints'
    save_best_only: bool = True


@dataclass
class DataConfig:
    """Data configuration."""
    
    dataset_name: str = 'Cora'  # 'Cora', 'Citeseer', 'Pubmed'
    data_dir: str = './data'
    normalize_features: bool = True
    
    # Data splits (if not using predefined splits)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()


# Predefined configurations for different models
GCN_CONFIG = Config(
    model=ModelConfig(
        model_name='gcn',
        hidden_dim=64,
        num_layers=2,
        dropout=0.5
    ),
    training=TrainingConfig(
        lr=0.01,
        weight_decay=5e-4,
        epochs=200,
        patience=20
    )
)

GAT_CONFIG = Config(
    model=ModelConfig(
        model_name='gat',
        hidden_dim=64,
        num_layers=2,
        dropout=0.6,
        num_heads=8,
        head_concat=True
    ),
    training=TrainingConfig(
        lr=0.005,
        weight_decay=5e-4,
        epochs=200,
        patience=20
    )
)

GRAPHSAGE_CONFIG = Config(
    model=ModelConfig(
        model_name='graphsage',
        hidden_dim=64,
        num_layers=2,
        dropout=0.5,
        aggregator='mean',
        normalize=True
    ),
    training=TrainingConfig(
        lr=0.01,
        weight_decay=5e-4,
        epochs=200,
        patience=20
    )
)

# Model-specific hyperparameter search spaces
HYPERPARAMETER_SEARCH_SPACES = {
    'gcn': {
        'hidden_dim': [32, 64, 128, 256],
        'num_layers': [2, 3, 4],
        'dropout': [0.3, 0.5, 0.7],
        'lr': [0.001, 0.005, 0.01, 0.05],
        'weight_decay': [1e-5, 5e-5, 1e-4, 5e-4]
    },
    'gat': {
        'hidden_dim': [32, 64, 128],
        'num_layers': [2, 3],
        'dropout': [0.4, 0.6, 0.8],
        'num_heads': [4, 8, 16],
        'lr': [0.001, 0.005, 0.01],
        'weight_decay': [1e-5, 5e-5, 1e-4, 5e-4]
    },
    'graphsage': {
        'hidden_dim': [32, 64, 128, 256],
        'num_layers': [2, 3, 4],
        'dropout': [0.3, 0.5, 0.7],
        'aggregator': ['mean', 'max'],
        'lr': [0.001, 0.005, 0.01, 0.05],
        'weight_decay': [1e-5, 5e-5, 1e-4, 5e-4]
    }
}


def get_config(model_name: str) -> Config:
    """
    Get predefined configuration for a model.
    
    Args:
        model_name: Name of the model ('gcn', 'gat', 'graphsage')
    
    Returns:
        Configuration object
    """
    configs = {
        'gcn': GCN_CONFIG,
        'gat': GAT_CONFIG,
        'graphsage': GRAPHSAGE_CONFIG
    }
    
    if model_name.lower() not in configs:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(configs.keys())}")
    
    return configs[model_name.lower()]


if __name__ == '__main__':
    # Example usage
    print("Available configurations:")
    for name in ['gcn', 'gat', 'graphsage']:
        config = get_config(name)
        print(f"\n{name.upper()} Config:")
        print(f"  Hidden dim: {config.model.hidden_dim}")
        print(f"  Layers: {config.model.num_layers}")
        print(f"  Dropout: {config.model.dropout}")
        print(f"  LR: {config.training.lr}")
        print(f"  Device: {config.training.device}")
