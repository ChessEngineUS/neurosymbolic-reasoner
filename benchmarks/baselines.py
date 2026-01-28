"""Baseline models for comparison."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class TransformerBaseline(nn.Module):
    """Standard Transformer baseline for visual reasoning.
    
    Based on 'Attention is All You Need' (Vaswani et al., 2017)
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 768,
                 num_layers: int = 6, num_heads: int = 8, num_classes: int = 1000):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_proj(x)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        
        return {'logits': logits}


class NeuralModuleNetworkBaseline(nn.Module):
    """Neural Module Network baseline.
    
    Based on 'Neural Module Networks' (Andreas et al., 2016)
    Simplified implementation for comparison.
    """
    
    def __init__(self, visual_dim: int = 512, hidden_dim: int = 512,
                 num_modules: int = 10, num_classes: int = 1000):
        super().__init__()
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Define module networks
        self.modules = nn.ModuleDict({
            'find': self._create_attention_module(hidden_dim),
            'filter': self._create_attention_module(hidden_dim),
            'relate': self._create_relation_module(hidden_dim),
            'and': self._create_combine_module(hidden_dim),
            'or': self._create_combine_module(hidden_dim)
        })
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def _create_attention_module(self, dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Softmax(dim=1)
        )
    
    def _create_relation_module(self, dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def _create_combine_module(self, dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode visual features
        features = self.visual_encoder(x)
        
        # Apply find module
        attention = self.modules['find'](features)
        attended = (features * attention).sum(dim=1)
        
        # Classify
        logits = self.classifier(attended)
        
        return {'logits': logits}


class RelationNetworkBaseline(nn.Module):
    """Relation Network baseline.
    
    Based on 'A Simple Neural Network Module for Relational Reasoning' 
    (Santoro et al., 2017)
    """
    
    def __init__(self, object_dim: int = 512, hidden_dim: int = 512,
                 num_classes: int = 1000):
        super().__init__()
        
        # g_theta: processes object pairs
        self.g_theta = nn.Sequential(
            nn.Linear(object_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # f_phi: aggregates relations
        self.f_phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, num_objects, object_dim = x.shape
        
        # Compute all pairs
        relations = []
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    pair = torch.cat([x[:, i], x[:, j]], dim=1)
                    relation = self.g_theta(pair)
                    relations.append(relation)
        
        # Aggregate relations
        relations = torch.stack(relations, dim=1)
        aggregated = relations.sum(dim=1)
        
        # Final classification
        logits = self.f_phi(aggregated)
        
        return {'logits': logits}


class FilmBaseline(nn.Module):
    """FiLM (Feature-wise Linear Modulation) baseline.
    
    Based on 'FiLM: Visual Reasoning with a General Conditioning Layer'
    (Perez et al., 2018)
    """
    
    def __init__(self, visual_dim: int = 512, text_dim: int = 512,
                 hidden_dim: int = 512, num_layers: int = 4,
                 num_classes: int = 1000):
        super().__init__()
        
        self.text_encoder = nn.LSTM(text_dim, hidden_dim, batch_first=True)
        
        # FiLM layers
        self.film_layers = nn.ModuleList([
            self._create_film_layer(visual_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def _create_film_layer(self, input_dim: int, output_dim: int) -> nn.Module:
        return nn.ModuleDict({
            'conv': nn.Linear(input_dim, output_dim),
            'gamma': nn.Linear(output_dim, output_dim),
            'beta': nn.Linear(output_dim, output_dim)
        })
    
    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode text
        _, (text_encoding, _) = self.text_encoder(text)
        text_encoding = text_encoding.squeeze(0)
        
        # Apply FiLM layers
        x = visual
        for layer in self.film_layers:
            x = layer['conv'](x)
            gamma = layer['gamma'](text_encoding).unsqueeze(1)
            beta = layer['beta'](text_encoding).unsqueeze(1)
            x = gamma * x + beta
            x = F.relu(x)
        
        # Pool and classify
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        
        return {'logits': logits}
