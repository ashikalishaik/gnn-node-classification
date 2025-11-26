"""Evaluation utilities for GNN models.

Provides comprehensive evaluation functionality:
- Multiple metrics (accuracy, F1, precision, recall)
- Per-class performance analysis
- Confusion matrix generation and visualization
- Training history plotting
- Model comparison utilities
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import json
import os
from typing import Dict, List, Optional, Tuple


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names for visualization
        """
        self.class_names = class_names
        self.results = {}
    
    @torch.no_grad()
    def evaluate_model(
        self,
        model: torch.nn.Module,
        data,
        mask: torch.Tensor,
        split_name: str = 'test'
    ) -> Dict[str, float]:
        """
        Evaluate model on a data split.
        
        Args:
            model: Trained GNN model
            data: PyTorch Geometric data object
            mask: Boolean mask for the split
            split_name: Name of the split (train/val/test)
        
        Returns:
            Dictionary of metrics
        """
        model.eval()
        device = next(model.parameters()).device
        data = data.to(device)
        
        # Get predictions
        out = model(data)
        pred = out.argmax(dim=1)
        
        # Extract labels and predictions for this split
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        
        # Compute metrics
        metrics = {
            f'{split_name}_accuracy': accuracy_score(y_true, y_pred),
            f'{split_name}_f1_macro': f1_score(y_true, y_pred, average='macro'),
            f'{split_name}_f1_micro': f1_score(y_true, y_pred, average='micro'),
            f'{split_name}_f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            f'{split_name}_precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            f'{split_name}_recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # Store results for later analysis
        self.results[split_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'metrics': metrics
        }
        
        return metrics
    
    def get_classification_report(self, split_name: str = 'test') -> str:
        """
        Generate detailed classification report.
        
        Args:
            split_name: Name of the split
        
        Returns:
            Classification report string
        """
        if split_name not in self.results:
            raise ValueError(f"No results for split '{split_name}'")
        
        y_true = self.results[split_name]['y_true']
        y_pred = self.results[split_name]['y_pred']
        
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0
        )
    
    def plot_confusion_matrix(
        self,
        split_name: str = 'test',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot confusion matrix.
        
        Args:
            split_name: Name of the split
            save_path: Path to save the plot
            figsize: Figure size
        """
        if split_name not in self.results:
            raise ValueError(f"No results for split '{split_name}'")
        
        y_true = self.results[split_name]['y_true']
        y_pred = self.results[split_name]['y_pred']
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {split_name.capitalize()} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_per_class_metrics(
        self,
        split_name: str = 'test',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot per-class F1, precision, and recall.
        
        Args:
            split_name: Name of the split
            save_path: Path to save the plot
            figsize: Figure size
        """
        if split_name not in self.results:
            raise ValueError(f"No results for split '{split_name}'")
        
        y_true = self.results[split_name]['y_true']
        y_pred = self.results[split_name]['y_pred']
        
        # Compute per-class metrics
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        # Create plot
        x = np.arange(len(f1_per_class))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
        ax.bar(x, recall_per_class, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Class Metrics - {split_name.capitalize()} Set')
        ax.set_xticks(x)
        if self.class_names:
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics plot saved to {save_path}")
        
        plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def compare_models(
    results_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Compare multiple models across different metrics.
    
    Args:
        results_dict: Dictionary mapping model names to their metrics
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Extract metrics to compare
    metrics_to_plot = ['test_accuracy', 'test_f1_macro', 'test_precision_macro', 'test_recall_macro']
    model_names = list(results_dict.keys())
    
    # Prepare data
    data = {metric: [] for metric in metrics_to_plot}
    for model_name in model_names:
        for metric in metrics_to_plot:
            value = results_dict[model_name].get(metric, 0)
            data[metric].append(value)
    
    # Create plot
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, metric in enumerate(metrics_to_plot):
        offset = width * (i - len(metrics_to_plot)/2 + 0.5)
        ax.bar(x + offset, data[metric], width, label=metric.replace('test_', '').replace('_', ' ').title())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison Across Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.show()


def save_results(
    results: Dict,
    filepath: str
):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary of results
        filepath: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filepath}")


if __name__ == '__main__':
    # Example usage
    print("Evaluation utilities loaded successfully!")
    print("\nAvailable functions:")
    print("- ModelEvaluator: Comprehensive model evaluation")
    print("- plot_training_history: Plot training curves")
    print("- compare_models: Compare multiple models")
    print("- save_results: Save results to JSON")
