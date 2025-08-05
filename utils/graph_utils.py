"""
Graph utility functions for XNode framework.
Provides functions for graph construction, topological feature computation,
and data preprocessing.
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pickle
import os


def compute_topological_features(G):
    """
    Compute topological features for all nodes in the graph.
    
    Args:
        G: NetworkX graph object
        
    Returns:
        dict: Dictionary containing topological features for each node
    """
    print("Computing topological features...")
    
    # Basic features
    degrees = np.array([G.degree(n) for n in G.nodes()])
    clustering = np.array([nx.clustering(G, n) for n in G.nodes()])
    
    # Two-hop agreement
    two_hop_agreement = []
    for n in tqdm(G.nodes(), desc="Computing two-hop agreement"):
        two_hop = set(nx.single_source_shortest_path_length(G, n, cutoff=2).keys()) - {n}
        if two_hop:
            same_label = sum(1 for m in two_hop if G.nodes[m]['y'] == G.nodes[n]['y'])
            two_hop_agreement.append(same_label / len(two_hop))
        else:
            two_hop_agreement.append(0.0)
    two_hop_agreement = np.array(two_hop_agreement)
    
    # Centrality measures
    eigenvector_centrality = np.array(list(nx.eigenvector_centrality_numpy(G, weight='weight').values()))
    degree_centrality = np.array(list(nx.degree_centrality(G).values()))
    
    # Average edge weight
    avg_edge_weight = []
    for n in G.nodes():
        if G.degree(n) > 0:
            weights = [G.edges[n, m]['weight'] for m in G.neighbors(n)]
            avg_edge_weight.append(np.mean(weights))
        else:
            avg_edge_weight.append(0.0)
    avg_edge_weight = np.array(avg_edge_weight)
    
    # Betweenness centrality (optional, can be slow for large graphs)
    try:
        betweenness_centrality = np.array(list(nx.betweenness_centrality(G, weight='weight').values()))
    except:
        print("Warning: Betweenness centrality computation failed, using zeros")
        betweenness_centrality = np.zeros(len(G.nodes()))
    
    return {
        'degrees': degrees,
        'clustering': clustering,
        'two_hop_agreement': two_hop_agreement,
        'eigenvector_centrality': eigenvector_centrality,
        'degree_centrality': degree_centrality,
        'avg_edge_weight': avg_edge_weight,
        'betweenness_centrality': betweenness_centrality
    }


def create_context_vectors(topological_features):
    """
    Create standardized context vectors from topological features.
    
    Args:
        topological_features: Dictionary of topological features
        
    Returns:
        torch.Tensor: Standardized context vectors
    """
    # Stack features
    feature_names = ['degrees', 'clustering', 'two_hop_agreement', 
                    'eigenvector_centrality', 'degree_centrality', 'avg_edge_weight']
    
    context_vectors = np.vstack([topological_features[name] for name in feature_names]).T
    
    # Standardize
    scaler = StandardScaler()
    context_vectors = scaler.fit_transform(context_vectors)
    
    return torch.tensor(context_vectors, dtype=torch.float)


def prepare_pytorch_geometric_data(G, topological_features=None):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.
    
    Args:
        G: NetworkX graph object
        topological_features: Optional pre-computed topological features
        
    Returns:
        Data: PyTorch Geometric Data object
    """
    # Extract basic data
    x = np.array([G.nodes[n]['x'] for n in G.nodes()])
    y = np.array([G.nodes[n]['y'] for n in G.nodes()])
    edge_index = np.array(list(G.edges())).T
    
    # Create masks
    train_mask = np.array([G.nodes[n]['train'] for n in G.nodes()])
    val_mask = np.array([G.nodes[n]['val'] for n in G.nodes()])
    test_mask = np.array([G.nodes[n]['test'] for n in G.nodes()])
    
    # Compute topological features if not provided
    if topological_features is None:
        topological_features = compute_topological_features(G)
    
    # Create context vectors
    context = create_context_vectors(topological_features)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
        context=context
    )
    
    return data


def build_knn_graph_from_features(features, labels, k=8, metric='cosine'):
    """
    Build k-NN graph from feature vectors.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Label array (n_samples,)
        k: Number of neighbors
        metric: Distance metric for k-NN
        
    Returns:
        NetworkX graph object
    """
    print(f"Building k-NN graph with k={k}, metric={metric}")
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(features)
    dists, inds = nbrs.kneighbors(features)
    
    # Create edges
    edge_list = []
    for i in tqdm(range(len(features)), desc="Building edges"):
        for r in range(1, k):  # Skip self (r=0)
            j = inds[i, r]
            if i < j:  # Avoid duplicate edges
                sim = 1 - dists[i, r]  # Convert distance to similarity
                edge_list.append((i, j, sim))
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i in tqdm(range(len(features)), desc="Adding nodes"):
        G.add_node(i, x=features[i], y=labels[i])
    
    # Add edges
    for edge in tqdm(edge_list, desc="Adding edges"):
        G.add_edge(edge[0], edge[1], weight=edge[2])
    
    return G


def save_graph(G, filename):
    """
    Save graph to pickle file.
    
    Args:
        G: NetworkX graph object
        filename: Output filename
    """
    with open(filename, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved graph to {filename}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def load_graph(filename):
    """
    Load graph from pickle file.
    
    Args:
        filename: Input filename
        
    Returns:
        NetworkX graph object
    """
    with open(filename, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph from {filename}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_device():
    """
    Get the best available device (MPS, CUDA, or CPU).
    
    Returns:
        torch.device: Device object
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 