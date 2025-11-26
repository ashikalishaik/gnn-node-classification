"""Training pipeline for GNN models.

Provides comprehensive training functionality:
- Early stopping
- Model checkpointing
- Learning rate scheduling
- Progress logging
- WandB integration (optional)
"""

import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from data_loader import load_cora
from models import GCN, GAT, GraphSAGE

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Install with: pip install wandb")


class Trainer:
    """GNN model trainer with early stopping and checkpointing.
    
    Args:
        model: GNN model (GCN, GAT, or GraphSAGE)
        data: PyG Data object
        device: Device to train on
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        save_dir: Directory to save models
    """
    
    def __init__(
        self,
        model,
        data,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=0.01,
        weight_decay=5e-4,
        patience=20,
        save_dir='results/models'
    ):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.patience = patience
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer and scheduler
        self.optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Training state
        self.best_val_acc = 0
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_losses = []
        self.val_accs = []
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(self.data.x, self.data.edge_index)
        
        # Compute loss only on training nodes
        loss = F.nll_loss(
            out[self.data.train_mask],
            self.data.y[self.data.train_mask]
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, mask):
        """Evaluate model on given mask.
        
        Args:
            mask: Boolean mask for evaluation
            
        Returns:
            accuracy: Classification accuracy
        """
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)
        
        correct = pred[mask] == self.data.y[mask]
        accuracy = correct.sum().item() / mask.sum().item()
        
        return accuracy
    
    def train(self, epochs=200, verbose=True, use_wandb=False):
        """Full training loop with early stopping.
        
        Args:
            epochs: Maximum number of epochs
            verbose: Whether to print progress
            use_wandb: Whether to log to Weights & Biases
            
        Returns:
            History dictionary with training metrics
        """
        if use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb not available, disabling logging")
            use_wandb = False
        
        if verbose:
            print(f"\nTraining {self.model.__class__.__name__}")
            print(f"Device: {self.device}")
            print(f"Parameters: {self.model.count_parameters():,}")
            print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Evaluate
            train_acc = self.evaluate(self.data.train_mask)
            val_acc = self.evaluate(self.data.val_mask)
            test_acc = self.evaluate(self.data.test_mask)
            self.val_accs.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Print progress
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Train: {train_acc:.4f} | "
                      f"Val: {val_acc:.4f} | "
                      f"Test: {test_acc:.4f}")
            
            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model.pt')
                
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        # Training complete
        elapsed = time.time() - start_time
        
        if verbose:
            print("-" * 80)
            print(f"Training complete in {elapsed:.2f}s")
            print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        
        # Load best model and get final test accuracy
        self.load_checkpoint('best_model.pt')
        final_test_acc = self.evaluate(self.data.test_mask)
        
        if verbose:
            print(f"Final test accuracy: {final_test_acc:.4f}")
            print("-" * 80)
        
        return {
            'train_losses': self.train_losses,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'final_test_acc': final_test_acc,
            'training_time': elapsed
        }
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
        }, path)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']


def get_model(model_name, num_features, num_classes, **kwargs):
    """Get model by name."""
    if model_name.lower() == 'gcn':
        return GCN(num_features, num_classes=num_classes, **kwargs)
    elif model_name.lower() == 'gat':
        return GAT(num_features, num_classes=num_classes, **kwargs)
    elif model_name.lower() == 'graphsage':
        return GraphSAGE(num_features, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train GNN for node classification')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphsage'],
                       help='GNN model to train')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # GAT specific
    parser.add_argument('--heads', type=int, default=8,
                       help='Number of attention heads (GAT only)')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--save-dir', type=str, default='results/models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project='gnn-node-classification',
            config=vars(args),
            name=f"{args.model}-{int(time.time())}"
        )
    
    # Load data
    print("Loading Cora dataset...")
    data, dataset = load_cora()
    
    # Create model
    model_kwargs = {
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout
    }
    
    if args.model == 'gat':
        model_kwargs['heads'] = args.heads
    
    model = get_model(
        args.model,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        **model_kwargs
    )
    
    # Train
    trainer = Trainer(
        model,
        data,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_dir=args.save_dir
    )
    
    history = trainer.train(
        epochs=args.epochs,
        verbose=True,
        use_wandb=args.wandb
    )
    
    print("\n✓ Training complete!")
    print(f"Best model saved to: {args.save_dir}/best_model.pt")


if __name__ == '__main__':
    main()
