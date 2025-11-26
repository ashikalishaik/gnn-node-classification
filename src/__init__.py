"""GNN Node Classification Package.

A production-ready Graph Neural Network package for node classification
with support for multiple architectures (GCN, GAT, GraphSAGE).
"""

__version__ = '1.0.0'
__author__ = 'Ashikal Ishaik'

# Import key components for easy access
from .data_loader import load_cora, load_citeseer, load_pubmed, DataLoader
from .models import GCN, GAT, GraphSAGE, DeepGCN, GATv2, DeepGraphSAGE
from .train import Trainer, get_model
from .evaluate import ModelEvaluator, plot_training_history, compare_models, save_results
from .config import Config, ModelConfig, TrainingConfig, DataConfig, get_config

__all__ = [
    # Data loading
    'load_cora',
    'load_citeseer',
    'load_pubmed',
    'DataLoader',
    # Models
    'GCN',
    'GAT',
    'GraphSAGE',
    'DeepGCN',
    'GATv2',
    'DeepGraphSAGE',
    # Training
    'Trainer',
    'get_model',
    # Evaluation
    'ModelEvaluator',
    'plot_training_history',
    'compare_models',
    'save_results',
    # Configuration
    'Config',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'get_config',
]
