"""
Explanation utility functions for XNode framework.
Provides functions for generating language model explanations
and managing explanation outputs.
"""

import os
import json
import numpy as np
from groq import Groq
from typing import Dict, List, Tuple, Optional
import torch
import networkx as nx


class ExplanationGenerator:
    """
    Class for generating explanations using Grok API.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the explanation generator.
        
        Args:
            api_key: Grok API key
        """
        self.client = Groq(api_key=api_key)
        
    def generate_topological_explanation(self, 
                                       node_features: np.ndarray,
                                       topological_features: Dict[str, np.ndarray],
                                       node_idx: int,
                                       prediction: int,
                                       true_label: int,
                                       confidence: float) -> str:
        """
        Generate explanation for a node based on its topological features.
        
        Args:
            node_features: Node feature vector
            topological_features: Dictionary of topological features
            node_idx: Node index
            prediction: Model prediction
            true_label: True label
            confidence: Prediction confidence
            
        Returns:
            str: Generated explanation
        """
        # Extract features for this node
        node_deg = topological_features['degrees'][node_idx]
        node_clust = topological_features['clustering'][node_idx]
        node_two_hop = topological_features['two_hop_agreement'][node_idx]
        node_eigen = topological_features['eigenvector_centrality'][node_idx]
        node_deg_cent = topological_features['degree_centrality'][node_idx]
        node_avg_weight = topological_features['avg_edge_weight'][node_idx]
        
        prompt = f"""
        Analyze the topological features of a node in a medical image classification graph and explain why the model made its prediction.
        
        Node Information:
        - Node Index: {node_idx}
        - Prediction: Class {prediction}
        - True Label: Class {true_label}
        - Confidence: {confidence:.3f}
        
        Topological Features:
        - Degree: {node_deg:.3f} (number of connections)
        - Clustering Coefficient: {node_clust:.3f} (local clustering measure)
        - Two-hop Agreement: {node_two_hop:.3f} (fraction of 2-hop neighbors with same label)
        - Eigenvector Centrality: {node_eigen:.3f} (global importance measure)
        - Degree Centrality: {node_deg_cent:.3f} (local importance measure)
        - Average Edge Weight: {node_avg_weight:.3f} (mean similarity to neighbors)
        
        Please provide a clear explanation of:
        1. How the node's position in the graph structure influenced the prediction
        2. Which topological features were most important for this classification
        3. What the local neighborhood structure tells us about this node
        4. Whether the prediction makes sense given the topological context
        
        Focus on the graph-theoretic interpretation rather than the raw image features.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def generate_feature_importance_explanation(self,
                                              topological_features: Dict[str, np.ndarray],
                                              feature_importance: np.ndarray,
                                              node_idx: int) -> str:
        """
        Generate explanation of feature importance for a node.
        
        Args:
            topological_features: Dictionary of topological features
            feature_importance: Feature importance scores
            node_idx: Node index
            
        Returns:
            str: Generated explanation
        """
        feature_names = ['Degree', 'Clustering', 'Two-hop Agreement', 
                        'Eigenvector Centrality', 'Degree Centrality', 'Avg Edge Weight']
        
        # Get feature values and importance for this node
        node_features = []
        for name in ['degrees', 'clustering', 'two_hop_agreement', 
                    'eigenvector_centrality', 'degree_centrality', 'avg_edge_weight']:
            node_features.append(topological_features[name][node_idx])
        
        # Sort by importance
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        prompt = f"""
        Analyze the importance of topological features for node {node_idx} in a medical image classification graph.
        
        Feature Values and Importance:
        """
        
        for i, idx in enumerate(sorted_indices):
            prompt += f"\n{i+1}. {feature_names[idx]}: {node_features[idx]:.3f} (Importance: {feature_importance[idx]:.3f})"
        
        prompt += f"""
        
        Please explain:
        1. Which topological features are most important for this node's classification
        2. What these feature values tell us about the node's role in the graph
        3. How the feature importance ranking relates to the node's structural position
        4. What insights this provides about the model's decision-making process
        
        Focus on the graph-theoretic interpretation and medical relevance.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating feature importance explanation: {str(e)}"
    
    def generate_neighborhood_explanation(self,
                                        G,
                                        node_idx: int,
                                        topological_features: Dict[str, np.ndarray]) -> str:
        """
        Generate explanation of a node's neighborhood structure.
        
        Args:
            G: NetworkX graph
            node_idx: Node index
            topological_features: Dictionary of topological features
            
        Returns:
            str: Generated explanation
        """
        # Get neighborhood information
        neighbors = list(G.neighbors(node_idx))
        neighbor_labels = [G.nodes[n]['y'] for n in neighbors]
        neighbor_weights = [G.edges[node_idx, n]['weight'] for n in neighbors]
        
        # Count labels in neighborhood
        unique_labels, label_counts = np.unique(neighbor_labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, label_counts))
        
        # Get 2-hop neighbors
        two_hop = set()
        for n in neighbors:
            two_hop.update(G.neighbors(n))
        two_hop.discard(node_idx)
        two_hop.discard(node_idx)
        
        prompt = f"""
        Analyze the neighborhood structure of node {node_idx} in a medical image classification graph.
        
        Neighborhood Information:
        - Number of direct neighbors: {len(neighbors)}
        - Label distribution in neighborhood: {label_distribution}
        - Average edge weight to neighbors: {np.mean(neighbor_weights):.3f}
        - Number of 2-hop neighbors: {len(two_hop)}
        
        Node's Topological Features:
        - Degree: {topological_features['degrees'][node_idx]:.3f}
        - Clustering Coefficient: {topological_features['clustering'][node_idx]:.3f}
        - Two-hop Agreement: {topological_features['two_hop_agreement'][node_idx]:.3f}
        
        Please explain:
        1. What the neighborhood structure reveals about this node's classification
        2. How the local clustering and label consistency influence predictions
        3. What the edge weights tell us about feature similarity
        4. How this neighborhood analysis helps understand the model's reasoning
        
        Focus on the structural patterns and their medical interpretation.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating neighborhood explanation: {str(e)}"
    
    def generate_global_pattern_explanation(self,
                                          G,
                                          topological_features: Dict[str, np.ndarray],
                                          class_predictions: np.ndarray) -> str:
        """
        Generate explanation of global graph patterns.
        
        Args:
            G: NetworkX graph
            topological_features: Dictionary of topological features
            class_predictions: Model predictions for all nodes
            
        Returns:
            str: Generated explanation
        """
        # Compute global statistics
        avg_degree = np.mean(topological_features['degrees'])
        avg_clustering = np.mean(topological_features['clustering'])
        avg_two_hop = np.mean(topological_features['two_hop_agreement'])
        
        # Class distribution
        unique_classes, class_counts = np.unique(class_predictions, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        
        # Graph density
        density = nx.density(G)
        
        prompt = f"""
        Analyze the global patterns in a medical image classification graph.
        
        Global Graph Statistics:
        - Number of nodes: {G.number_of_nodes()}
        - Number of edges: {G.number_of_edges()}
        - Graph density: {density:.3f}
        - Average degree: {avg_degree:.3f}
        - Average clustering coefficient: {avg_clustering:.3f}
        - Average two-hop agreement: {avg_two_hop:.3f}
        
        Class Distribution:
        {class_distribution}
        
        Please explain:
        1. What these global patterns reveal about the dataset structure
        2. How the graph topology reflects medical image similarities
        3. What the clustering and agreement measures tell us about class separability
        4. How these global insights help understand the model's performance
        
        Focus on the medical and structural interpretation of the patterns.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating global pattern explanation: {str(e)}"


def save_explanations(explanations: Dict, filename: str):
    """
    Save explanations to a JSON file.
    
    Args:
        explanations: Dictionary of explanations
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(explanations, f, indent=2)
    print(f"Saved explanations to {filename}")


def load_explanations(filename: str) -> Dict:
    """
    Load explanations from a JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        dict: Loaded explanations
    """
    with open(filename, 'r') as f:
        explanations = json.load(f)
    print(f"Loaded explanations from {filename}")
    return explanations


def create_explanation_summary(explanations: Dict, output_file: str):
    """
    Create a summary of all explanations.
    
    Args:
        explanations: Dictionary of explanations
        output_file: Output filename
    """
    with open(output_file, 'w') as f:
        f.write("# XNode Explanation Summary\n\n")
        
        for node_idx, node_explanations in explanations.items():
            f.write(f"## Node {node_idx}\n\n")
            
            for explanation_type, explanation in node_explanations.items():
                f.write(f"### {explanation_type.replace('_', ' ').title()}\n\n")
                f.write(f"{explanation}\n\n")
                f.write("---\n\n")
    
    print(f"Created explanation summary: {output_file}") 