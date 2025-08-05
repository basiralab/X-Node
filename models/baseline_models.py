"""
Baseline GNN models for XNode framework.
Implements standard GCN, GAT, and GIN architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.data import Data


class GCN(nn.Module):
    """
    Graph Convolutional Network baseline model.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    """
    Graph Attention Network baseline model.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 heads=8, dropout=0.5, negative_slope=0.2):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, 
                                 dropout=dropout, negative_slope=negative_slope))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                    heads=heads, dropout=dropout, 
                                    negative_slope=negative_slope))
        
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * heads, out_channels, 
                                    heads=1, dropout=dropout, 
                                    negative_slope=negative_slope))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


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


class GIN(nn.Module):
    """
    Graph Isomorphism Network baseline model.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(MLP(in_channels, hidden_channels, hidden_channels)))
        
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels)))
        
        if num_layers > 1:
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, out_channels)))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


def get_baseline_model(model_type, in_channels, hidden_channels, out_channels, **kwargs):
    """
    Factory function to create baseline models.
    
    Args:
        model_type: Type of model ('gcn', 'gat', 'gin')
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension (number of classes)
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: Initialized model
    """
    model_type = model_type.lower()
    
    if model_type == 'gcn':
        return GCN(in_channels, hidden_channels, out_channels, **kwargs)
    elif model_type == 'gat':
        return GAT(in_channels, hidden_channels, out_channels, **kwargs)
    elif model_type == 'gin':
        return GIN(in_channels, hidden_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 