#!/usr/bin/env python3
"""
Generate language model explanations for graph topological features.
"""

import argparse
import os
import sys
import torch
import numpy as np
import json
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_utils import load_graph, prepare_pytorch_geometric_data, get_device, set_random_seeds
from utils.explanation_utils import ExplanationGenerator, save_explanations, create_explanation_summary
from reasoners.reasoner_models import get_reasoner_model


def load_trained_model(model_path, model_type, in_channels, hidden_channels, out_channels, device):
    """
    Load a trained reasoner model.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model ('gcn', 'gat', 'gin')
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension
        device: Device to use
        
    Returns:
        nn.Module: Loaded model
    """
    model = get_reasoner_model(model_type, in_channels, hidden_channels, out_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def generate_node_explanations(model, data, G, topological_features, explanation_generator, 
                              device, num_nodes=50):
    """
    Generate explanations for selected nodes.
    
    Args:
        model: Trained reasoner model
        data: PyTorch Geometric data object
        G: NetworkX graph
        topological_features: Dictionary of topological features
        explanation_generator: ExplanationGenerator instance
        device: Device to use
        num_nodes: Number of nodes to explain
        
    Returns:
        dict: Dictionary of explanations for each node
    """
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        # Get model predictions and explanations
        out, explanations = model(data.x, data.edge_index, data.context)
        predictions = out.argmax(dim=1)
        confidences = torch.softmax(out, dim=1).max(dim=1)[0]
    
    # Select nodes to explain (mix of correct and incorrect predictions)
    test_indices = torch.where(data.test_mask)[0].cpu().numpy()
    true_labels = data.y[data.test_mask].cpu().numpy()
    pred_labels = predictions[data.test_mask].cpu().numpy()
    conf_scores = confidences[data.test_mask].cpu().numpy()
    
    # Find correct and incorrect predictions
    correct_mask = (true_labels == pred_labels)
    incorrect_mask = ~correct_mask
    
    # Select nodes
    correct_indices = test_indices[correct_mask][:num_nodes//2]
    incorrect_indices = test_indices[incorrect_mask][:num_nodes//2]
    selected_indices = np.concatenate([correct_indices, incorrect_indices])
    
    if len(selected_indices) < num_nodes:
        # If not enough incorrect predictions, add more correct ones
        remaining_correct = test_indices[correct_mask][num_nodes//2:num_nodes-len(selected_indices)]
        selected_indices = np.concatenate([selected_indices, remaining_correct])
    
    explanations_dict = {}
    
    print(f"Generating explanations for {len(selected_indices)} nodes...")
    
    for i, node_idx in enumerate(tqdm(selected_indices, desc="Generating explanations")):
        node_explanations = {}
        
        # Get node information
        prediction = pred_labels[test_indices == node_idx][0]
        true_label = true_labels[test_indices == node_idx][0]
        confidence = conf_scores[test_indices == node_idx][0]
        
        # 1. Topological explanation
        topological_explanation = explanation_generator.generate_topological_explanation(
            node_features=data.x[node_idx].cpu().numpy(),
            topological_features=topological_features,
            node_idx=int(node_idx),
            prediction=int(prediction),
            true_label=int(true_label),
            confidence=float(confidence)
        )
        node_explanations['topological_explanation'] = topological_explanation
        
        # 2. Feature importance explanation
        feature_importance = explanations[node_idx].cpu().numpy()
        feature_importance_explanation = explanation_generator.generate_feature_importance_explanation(
            topological_features=topological_features,
            feature_importance=feature_importance,
            node_idx=int(node_idx)
        )
        node_explanations['feature_importance_explanation'] = feature_importance_explanation
        
        # 3. Neighborhood explanation
        neighborhood_explanation = explanation_generator.generate_neighborhood_explanation(
            G=G,
            node_idx=int(node_idx),
            topological_features=topological_features
        )
        node_explanations['neighborhood_explanation'] = neighborhood_explanation
        
        explanations_dict[str(node_idx)] = {
            'prediction': int(prediction),
            'true_label': int(true_label),
            'confidence': float(confidence),
            'correct': bool(prediction == true_label),
            'explanations': node_explanations
        }
    
    return explanations_dict


def generate_global_explanation(G, topological_features, explanation_generator, predictions):
    """
    Generate global explanation for the entire graph.
    
    Args:
        G: NetworkX graph
        topological_features: Dictionary of topological features
        explanation_generator: ExplanationGenerator instance
        predictions: Model predictions for all nodes
        
    Returns:
        str: Global explanation
    """
    global_explanation = explanation_generator.generate_global_pattern_explanation(
        G=G,
        topological_features=topological_features,
        class_predictions=predictions
    )
    
    return global_explanation


def main():
    parser = argparse.ArgumentParser(description='Generate explanations for graph nodes')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gcn', 'gat', 'gin'],
                       help='Model type')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., organcmnist)')
    parser.add_argument('--graph_file', type=str, required=True,
                       help='Path to graph file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='../results',
                       help='Output directory for explanations')
    parser.add_argument('--num_nodes', type=int, default=50,
                       help='Number of nodes to explain')
    parser.add_argument('--api_key', type=str, required=True,
                       help='Grok API key')
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
    
    # Load trained model
    print(f"Loading model from {args.model_path}")
    model = load_trained_model(
        args.model_path,
        args.model,
        data.x.size(1),
        64,  # hidden_channels
        len(torch.unique(data.y)),
        device
    )
    
    # Initialize explanation generator
    explanation_generator = ExplanationGenerator(args.api_key)
    
    # Compute topological features
    from utils.graph_utils import compute_topological_features
    topological_features = compute_topological_features(G)
    
    # Generate node explanations
    node_explanations = generate_node_explanations(
        model, data, G, topological_features, explanation_generator,
        device, args.num_nodes
    )
    
    # Generate global explanation
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index, data.context)
        predictions = out.argmax(dim=1).cpu().numpy()
    
    global_explanation = generate_global_explanation(
        G, topological_features, explanation_generator, predictions
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save explanations
    explanations_file = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_explanations.json")
    
    all_explanations = {
        'model': args.model,
        'dataset': args.dataset,
        'global_explanation': global_explanation,
        'node_explanations': node_explanations,
        'metadata': {
            'num_nodes_explained': len(node_explanations),
            'total_nodes': data.x.size(0),
            'num_classes': len(torch.unique(data.y))
        }
    }
    
    save_explanations(all_explanations, explanations_file)
    
    # Create summary
    summary_file = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_explanation_summary.md")
    create_explanation_summary(all_explanations, summary_file)
    
    # Print statistics
    correct_predictions = sum(1 for node_data in node_explanations.values() 
                            if node_data['correct'])
    total_predictions = len(node_explanations)
    
    print(f"\nExplanation generation completed!")
    print(f"Explanations saved to: {explanations_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Nodes explained: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy in explained set: {correct_predictions/total_predictions:.3f}")
    
    # Print sample explanations
    print(f"\nSample explanations:")
    for i, (node_idx, node_data) in enumerate(list(node_explanations.items())[:3]):
        print(f"\nNode {node_idx} (Pred: {node_data['prediction']}, True: {node_data['true_label']}, Correct: {node_data['correct']}):")
        print(f"Topological: {node_data['explanations']['topological_explanation'][:200]}...")
    
    print(f"\nGlobal explanation preview:")
    print(f"{global_explanation[:300]}...")


if __name__ == '__main__':
    main() 