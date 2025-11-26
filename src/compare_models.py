"""Compare multiple GNN models on node classification task.

Trains GCN, GAT, and GraphSAGE models and compares their performance.
Generates comparison plots and saves detailed results.
"""

import os
import sys
import torch
import numpy as np
import random
import json
from datetime import datetime

from data_loader import load_cora
from train import get_model, Trainer
from evaluate import ModelEvaluator, plot_training_history, compare_models, save_results
from config import get_config


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_model(
    model_name: str,
    data,
    device: str,
    results_dir: str = 'results',
    verbose: bool = True
):
    """
    Train a single model and return its results.
    
    Args:
        model_name: Name of model ('gcn', 'gat', 'graphsage')
        data: PyTorch Geometric data object
        device: Device to train on
        results_dir: Directory to save results
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing model results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")
    
    # Get configuration
    config = get_config(model_name)
    
    # Create model
    num_features = data.num_features
    num_classes = data.y.max().item() + 1
    
    model_kwargs = {
        'hidden_dim': config.model.hidden_dim,
        'dropout': config.model.dropout
    }
    
    if model_name == 'gat':
        model_kwargs['heads'] = config.model.num_heads
    elif model_name == 'graphsage':
        model_kwargs['aggr'] = config.model.aggregator
    
    model = get_model(
        model_name,
        num_features,
        num_classes,
        **model_kwargs
    )
    
    if verbose:
        print(f"Model created with {model.count_parameters():,} parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        data=data,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        patience=config.training.patience,
        save_dir=os.path.join(results_dir, 'checkpoints', model_name)
    )
    
    # Train model
    history = trainer.train(
        epochs=config.training.epochs,
        verbose=verbose
    )
    
    # Evaluate on all splits
    evaluator = ModelEvaluator(
        class_names=[f'Class {i}' for i in range(num_classes)]
    )
    
    train_metrics = evaluator.evaluate_model(model, data, data.train_mask, 'train')
    val_metrics = evaluator.evaluate_model(model, data, data.val_mask, 'val')
    test_metrics = evaluator.evaluate_model(model, data, data.test_mask, 'test')
    
    # Combine all metrics
    all_metrics = {**train_metrics, **val_metrics, **test_metrics}
    
    if verbose:
        print(f"\nFinal Results for {model_name.upper()}:")
        print(f"  Train Accuracy: {train_metrics['train_accuracy']:.4f}")
        print(f"  Val Accuracy: {val_metrics['val_accuracy']:.4f}")
        print(f"  Test Accuracy: {test_metrics['test_accuracy']:.4f}")
        print(f"  Test F1 (macro): {test_metrics['test_f1_macro']:.4f}")
    
    # Save training history plot
    plot_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_training_history(
        history,
        save_path=os.path.join(plot_dir, f'{model_name}_training_history.png')
    )
    
    # Save confusion matrix
    evaluator.plot_confusion_matrix(
        split_name='test',
        save_path=os.path.join(plot_dir, f'{model_name}_confusion_matrix.png')
    )
    
    return {
        'model_name': model_name,
        'metrics': all_metrics,
        'history': history,
        'config': {
            'hidden_dim': config.model.hidden_dim,
            'num_layers': config.model.num_layers,
            'dropout': config.model.dropout,
            'lr': config.training.lr,
            'weight_decay': config.training.weight_decay
        }
    }


def compare_all_models(
    models: list = ['gcn', 'gat', 'graphsage'],
    dataset_name: str = 'Cora',
    seed: int = 42,
    results_dir: str = 'results',
    device: str = None
):
    """
    Train and compare multiple GNN models.
    
    Args:
        models: List of model names to compare
        dataset_name: Name of dataset
        seed: Random seed
        results_dir: Directory to save results
        device: Device to use (auto-detects if None)
    
    Returns:
        Dictionary with all results
    """
    # Setup
    set_seed(seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Models to compare: {models}")
    print(f"Results will be saved to: {results_dir}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading {dataset_name} dataset...")
    if dataset_name.lower() == 'cora':
        data = load_cora(normalize_features=True)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented yet")
    
    print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Features: {data.num_features}, Classes: {data.y.max().item() + 1}")
    
    # Train all models
    all_results = {}
    for model_name in models:
        try:
            results = train_single_model(
                model_name=model_name,
                data=data,
                device=device,
                results_dir=results_dir,
                verbose=True
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    # Create comparison plot
    if len(all_results) > 1:
        print(f"\nCreating comparison plot...")
        comparison_metrics = {
            name: results['metrics']
            for name, results in all_results.items()
        }
        plot_dir = os.path.join(results_dir, 'plots')
        compare_models(
            comparison_metrics,
            save_path=os.path.join(plot_dir, 'model_comparison.png')
        )
    
    # Create summary table
    print(f"\n{'='*80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'Params':<12}")
    print(f"{'-'*80}")
    
    for model_name, results in all_results.items():
        metrics = results['metrics']
        # Count parameters (approximate)
        param_count = f"~{results['config']['hidden_dim'] * 2}K"
        
        print(f"{model_name.upper():<15} "
              f"{metrics['train_accuracy']:<12.4f} "
              f"{metrics['val_accuracy']:<12.4f} "
              f"{metrics['test_accuracy']:<12.4f} "
              f"{metrics['test_f1_macro']:<12.4f} "
              f"{param_count:<12}")
    
    # Save detailed results
    results_file = os.path.join(results_dir, 'comparison_results.json')
    save_results(all_results, results_file)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Find best model
    best_model = max(
        all_results.items(),
        key=lambda x: x[1]['metrics']['test_accuracy']
    )
    print(f"\nBest Model: {best_model[0].upper()} "
          f"(Test Accuracy: {best_model[1]['metrics']['test_accuracy']:.4f})")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare GNN models')
    parser.add_argument(
        '--models',
        nargs='+',
        default=['gcn', 'gat', 'graphsage'],
        choices=['gcn', 'gat', 'graphsage'],
        help='Models to compare'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='Cora',
        help='Dataset name'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu, auto-detect if not specified)'
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_all_models(
        models=args.models,
        dataset_name=args.dataset,
        seed=args.seed,
        results_dir=args.results_dir,
        device=args.device
    )
    
    print("\nComparison complete!")
