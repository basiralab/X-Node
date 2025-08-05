#!/usr/bin/env python3
"""
Training script for baseline GNN models.
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_utils import load_graph, prepare_pytorch_geometric_data, get_device, set_random_seeds
from models.baseline_models import get_baseline_model


def train_epoch(model, data, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: GNN model
        data: PyTorch Geometric data object
        optimizer: Optimizer
        device: Device to use
        
    Returns:
        float: Training loss
    """
    model.train()
    optimizer.zero_grad()
    
    data = data.to(device)
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, data, device, mask_type='val'):
    """
    Evaluate model performance.
    
    Args:
        model: GNN model
        data: PyTorch Geometric data object
        device: Device to use
        mask_type: Type of mask to use ('train', 'val', 'test')
        
    Returns:
        tuple: (accuracy, f1_score, loss)
    """
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        
        if mask_type == 'train':
            mask = data.train_mask
        elif mask_type == 'val':
            mask = data.val_mask
        else:
            mask = data.test_mask
        
        loss = F.nll_loss(out[mask], data.y[mask])
        pred = out[mask].argmax(dim=1)
        acc = accuracy_score(data.y[mask].cpu(), pred.cpu())
        f1 = f1_score(data.y[mask].cpu(), pred.cpu(), average='weighted')
        
        return acc, f1, loss.item()


def train_model(model, data, device, epochs=200, lr=0.01, weight_decay=5e-4, patience=20):
    """
    Train a GNN model.
    
    Args:
        model: GNN model
        data: PyTorch Geometric data object
        device: Device to use
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        patience: Early stopping patience
        
    Returns:
        dict: Training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Train
        train_loss = train_epoch(model, data, optimizer, device)
        
        # Evaluate
        val_acc, val_f1, val_loss = evaluate(model, data, device, 'val')
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break
    
    return history


def cross_validate(model_class, data, device, n_folds=5, epochs=200, lr=0.01, weight_decay=5e-4):
    """
    Perform k-fold cross-validation.
    
    Args:
        model_class: Model class to instantiate
        data: PyTorch Geometric data object
        device: Device to use
        n_folds: Number of folds
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        dict: Cross-validation results
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(data.x.size(0)))):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        # Create masks for this fold
        train_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
        val_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        
        # Create new data object with fold-specific masks
        fold_data = data.clone()
        fold_data.train_mask = train_mask
        fold_data.val_mask = val_mask
        
        # Initialize model
        model = model_class(
            in_channels=data.x.size(1),
            hidden_channels=64,
            out_channels=len(torch.unique(data.y))
        )
        
        # Train model
        history = train_model(model, fold_data, device, epochs, lr, weight_decay)
        
        # Evaluate on test set
        test_acc, test_f1, test_loss = evaluate(model, data, device, 'test')
        
        fold_results.append({
            'fold': fold + 1,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_loss': test_loss,
            'best_val_acc': max(history['val_acc']),
            'history': history
        })
        
        print(f"Fold {fold + 1} - Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    
    # Compute average results
    avg_test_acc = np.mean([r['test_acc'] for r in fold_results])
    avg_test_f1 = np.mean([r['test_f1'] for r in fold_results])
    avg_test_loss = np.mean([r['test_loss'] for r in fold_results])
    
    return {
        'fold_results': fold_results,
        'avg_test_acc': avg_test_acc,
        'avg_test_f1': avg_test_f1,
        'avg_test_loss': avg_test_loss,
        'std_test_acc': np.std([r['test_acc'] for r in fold_results]),
        'std_test_f1': np.std([r['test_f1'] for r in fold_results])
    }


def plot_training_curves(history, save_path):
    """
    Plot training curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.plot(history['val_f1'], label='Val F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train baseline GNN models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gcn', 'gat', 'gin'],
                       help='Model type to train')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., organcmnist)')
    parser.add_argument('--graph_file', type=str, required=True,
                       help='Path to graph file')
    parser.add_argument('--output_dir', type=str, default='../results',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seeds(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load graph and prepare data
    print(f"Loading graph from {args.graph_file}")
    G = load_graph(args.graph_file)
    data = prepare_pytorch_geometric_data(G)
    
    print(f"Dataset: {data.x.size(0)} nodes, {data.edge_index.size(1)} edges")
    print(f"Features: {data.x.size(1)} dimensions")
    print(f"Classes: {len(torch.unique(data.y))}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model class
    model_class = lambda **kwargs: get_baseline_model(args.model, **kwargs)
    
    # Perform cross-validation
    print(f"\nStarting {args.n_folds}-fold cross-validation for {args.model.upper()}")
    cv_results = cross_validate(
        model_class, data, device, 
        n_folds=args.n_folds,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Print results
    print(f"\nCross-validation results for {args.model.upper()}:")
    print(f"Average Test Accuracy: {cv_results['avg_test_acc']:.4f} ± {cv_results['std_test_acc']:.4f}")
    print(f"Average Test F1: {cv_results['avg_test_f1']:.4f} ± {cv_results['std_test_f1']:.4f}")
    print(f"Average Test Loss: {cv_results['avg_test_loss']:.4f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_baseline_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {
        'model': args.model,
        'dataset': args.dataset,
        'avg_test_acc': float(cv_results['avg_test_acc']),
        'avg_test_f1': float(cv_results['avg_test_f1']),
        'avg_test_loss': float(cv_results['avg_test_loss']),
        'std_test_acc': float(cv_results['std_test_acc']),
        'std_test_f1': float(cv_results['std_test_f1']),
        'fold_results': []
    }
    
    for fold_result in cv_results['fold_results']:
        serializable_fold = {
            'fold': fold_result['fold'],
            'test_acc': float(fold_result['test_acc']),
            'test_f1': float(fold_result['test_f1']),
            'test_loss': float(fold_result['test_loss']),
            'best_val_acc': float(fold_result['best_val_acc'])
        }
        serializable_results['fold_results'].append(serializable_fold)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Plot training curves for the first fold
    if cv_results['fold_results']:
        first_fold_history = cv_results['fold_results'][0]['history']
        plot_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_training_curves.png")
        plot_training_curves(first_fold_history, plot_path)
        print(f"Training curves saved to {plot_path}")


if __name__ == '__main__':
    main() 