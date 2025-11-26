"""Real-time inference module for GNN models.

Load trained models and perform node classification inference.
Supports batch and single-node predictions.
"""

import torch
import os
import argparse
from typing import Dict, List, Optional, Union
import numpy as np

from data_loader import load_cora
from train import get_model
from config import get_config


class GNNInference:
    """Inference engine for trained GNN models."""
    
    def __init__(
        self,
        model_path: str,
        model_name: str = 'gcn',
        device: str = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            model_name: Type of model ('gcn', 'gat', 'graphsage')
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model architecture config
        self.config = get_config(model_name)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model info from checkpoint
        num_features = checkpoint.get('num_features')
        num_classes = checkpoint.get('num_classes')
        
        # Create model
        model_kwargs = {
            'hidden_dim': self.config.model.hidden_dim,
            'dropout': self.config.model.dropout
        }
        
        if model_name == 'gat':
            model_kwargs['heads'] = self.config.model.num_heads
        elif model_name == 'graphsage':
            model_kwargs['aggr'] = self.config.model.aggregator
        
        self.model = get_model(
            model_name,
            num_features,
            num_classes,
            **model_kwargs
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.num_classes = num_classes
        
        print(f"Model loaded: {model_name}")
        print(f"Device: {self.device}")
        print(f"Parameters: {self.model.count_parameters():,}")
    
    @torch.no_grad()
    def predict(
        self,
        data,
        node_indices: Optional[Union[int, List[int]]] = None,
        return_probabilities: bool = False
    ) -> Union[int, List[int], torch.Tensor]:
        """
        Perform inference on nodes.
        
        Args:
            data: PyTorch Geometric data object
            node_indices: Specific node(s) to predict. If None, predicts all nodes.
            return_probabilities: If True, return class probabilities
        
        Returns:
            Predictions (class labels or probabilities)
        """
        data = data.to(self.device)
        
        # Get model output
        out = self.model(data)
        
        # Select specific nodes if requested
        if node_indices is not None:
            if isinstance(node_indices, int):
                out = out[node_indices:node_indices+1]
            else:
                out = out[node_indices]
        
        # Return probabilities or class labels
        if return_probabilities:
            probs = torch.softmax(out, dim=1)
            return probs.cpu()
        else:
            preds = out.argmax(dim=1)
            if node_indices is not None and isinstance(node_indices, int):
                return preds.item()
            return preds.cpu().tolist()
    
    def predict_with_confidence(
        self,
        data,
        node_index: int
    ) -> Dict[str, Union[int, float]]:
        """
        Predict node class with confidence score.
        
        Args:
            data: PyTorch Geometric data object
            node_index: Node to predict
        
        Returns:
            Dictionary with prediction and confidence
        """
        probs = self.predict(data, node_index, return_probabilities=True)
        probs = probs.squeeze()
        
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()
        
        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'probabilities': probs.tolist()
        }


def main():
    """CLI for model inference."""
    parser = argparse.ArgumentParser(description='GNN Model Inference')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='gcn',
        choices=['gcn', 'gat', 'graphsage'],
        help='Model architecture'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='Cora',
        help='Dataset name'
    )
    parser.add_argument(
        '--node-id',
        type=int,
        default=None,
        help='Specific node ID to predict (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    if args.dataset.lower() == 'cora':
        data = load_cora(normalize_features=True)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    print(f"Dataset: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Initialize inference engine
    inference = GNNInference(
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device
    )
    
    # Perform inference
    if args.node_id is not None:
        # Single node prediction with confidence
        result = inference.predict_with_confidence(data, args.node_id)
        print(f"\nNode {args.node_id} Prediction:")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probabilities: {np.array(result['probabilities'])}")
    else:
        # Predict all nodes
        print("\nPredicting all nodes...")
        predictions = inference.predict(data)
        print(f"Predictions shape: {len(predictions)}")
        print(f"Sample predictions (first 10): {predictions[:10]}")
    
    print("\nInference complete!")


if __name__ == '__main__':
    main()
