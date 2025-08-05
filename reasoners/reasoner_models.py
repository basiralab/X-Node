"""
Reasoner models for XNode framework.
Combines GNN architectures with explanation generation capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.data import Data


class Reasoner(nn.Module):
    """
    Reasoner MLP to generate explanation vectors from topological context.
    """
    
    def __init__(self, context_dim, hidden_dim, expl_dim):
        super(Reasoner, self).__init__()
        self.fc1 = nn.Linear(context_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, expl_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, context):
        h = F.relu(self.fc1(context))
        h = self.dropout(h)
        return self.fc2(h)


class Decoder(nn.Module):
    """
    Decoder MLP to reconstruct embeddings from explanation vectors.
    """
    
    def __init__(self, expl_dim, embed_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(expl_dim, embed_dim)
        
    def forward(self, expl):
        return self.fc(expl)


class GCNReasoner(nn.Module):
    """
    GCN-based reasoner model with explanation generation.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 context_dim=6, expl_dim=32, num_layers=2, dropout=0.5):
        super(GCNReasoner, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.expl_dim = expl_dim
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Reasoner and decoder
        self.reasoner = Reasoner(context_dim, hidden_channels, expl_dim)
        self.decoder = Decoder(expl_dim, hidden_channels)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
        # Reconstruction loss weight
        self.recon_weight = 0.1
        
    def forward(self, x, edge_index, context):
        # GCN forward pass
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        # Generate explanations
        explanations = self.reasoner(context)
        
        # Reconstruct embeddings
        reconstructed = self.decoder(explanations)
        
        # Combine original and reconstructed embeddings
        combined = x + self.recon_weight * reconstructed
        
        # Final classification
        out = self.classifier(combined)
        
        return F.log_softmax(out, dim=1), explanations


class GATReasoner(nn.Module):
    """
    GAT-based reasoner model with explanation generation.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 context_dim=6, expl_dim=32, num_layers=2, heads=8, 
                 dropout=0.5, negative_slope=0.2):
        super(GATReasoner, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.expl_dim = expl_dim
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, 
                                 dropout=dropout, negative_slope=negative_slope))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                    heads=heads, dropout=dropout, 
                                    negative_slope=negative_slope))
        
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                    heads=1, dropout=dropout, 
                                    negative_slope=negative_slope))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Reasoner and decoder
        self.reasoner = Reasoner(context_dim, hidden_channels, expl_dim)
        self.decoder = Decoder(expl_dim, hidden_channels)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
        # Reconstruction loss weight
        self.recon_weight = 0.1
        
    def forward(self, x, edge_index, context):
        # GAT forward pass
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        # Generate explanations
        explanations = self.reasoner(context)
        
        # Reconstruct embeddings
        reconstructed = self.decoder(explanations)
        
        # Combine original and reconstructed embeddings
        combined = x + self.recon_weight * reconstructed
        
        # Final classification
        out = self.classifier(combined)
        
        return F.log_softmax(out, dim=1), explanations


class MLP(nn.Module):
    """
    Multi-layer perceptron for GIN.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
    
    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.lins[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
        
        x = self.lins[-1](x)
        return x


class GINReasoner(nn.Module):
    """
    GIN-based reasoner model with explanation generation.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 context_dim=6, expl_dim=32, num_layers=2, dropout=0.5):
        super(GINReasoner, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.expl_dim = expl_dim
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(MLP(in_channels, hidden_channels, hidden_channels)))
        
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        
        if num_layers > 1:
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Reasoner and decoder
        self.reasoner = Reasoner(context_dim, hidden_channels, expl_dim)
        self.decoder = Decoder(expl_dim, hidden_channels)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
        # Reconstruction loss weight
        self.recon_weight = 0.1
        
    def forward(self, x, edge_index, context):
        # GIN forward pass
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        # Generate explanations
        explanations = self.reasoner(context)
        
        # Reconstruct embeddings
        reconstructed = self.decoder(explanations)
        
        # Combine original and reconstructed embeddings
        combined = x + self.recon_weight * reconstructed
        
        # Final classification
        out = self.classifier(combined)
        
        return F.log_softmax(out, dim=1), explanations


def get_reasoner_model(model_type, in_channels, hidden_channels, out_channels, **kwargs):
    """
    Factory function to create reasoner models.
    
    Args:
        model_type: Type of model ('gcn', 'gat', 'gin')
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension (number of classes)
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: Initialized reasoner model
    """
    model_type = model_type.lower()
    
    if model_type == 'gcn':
        return GCNReasoner(in_channels, hidden_channels, out_channels, **kwargs)
    elif model_type == 'gat':
        return GATReasoner(in_channels, hidden_channels, out_channels, **kwargs)
    elif model_type == 'gin':
        return GINReasoner(in_channels, hidden_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 