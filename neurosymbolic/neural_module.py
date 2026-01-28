"""Neural perception and learning module with T4 GPU optimization."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class PerceptionEncoder(nn.Module):
    """Transformer-based encoder for perception with efficient T4 optimization."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 768, 
                 num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.output_proj(x)
        x = self.layer_norm(x)
        return x


class ConceptLearner(nn.Module):
    """Learn continuous representations of symbolic concepts."""
    
    def __init__(self, num_concepts: int = 100, concept_dim: int = 256, 
                 hidden_dim: int = 768):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        
        self.concept_embeddings = nn.Embedding(num_concepts, concept_dim)
        
        self.concept_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_concepts)
        )
        
        self.composer = nn.Sequential(
            nn.Linear(concept_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, concept_dim)
        )
        
    def detect_concepts(self, encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = encoded.mean(dim=1)
        logits = self.concept_detector(pooled)
        probs = torch.sigmoid(logits)
        return logits, probs
    
    def compose_concepts(self, concept_ids: List[int]) -> torch.Tensor:
        if len(concept_ids) == 1:
            return self.concept_embeddings(torch.tensor([concept_ids[0]]))
        
        embeddings = self.concept_embeddings(torch.tensor(concept_ids))
        result = embeddings[0]
        for i in range(1, len(embeddings)):
            combined = torch.cat([result, embeddings[i]], dim=-1)
            result = self.composer(combined)
        return result


class NeuralModule(nn.Module):
    """Complete neural module integrating perception and concept learning."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 768,
                 num_concepts: int = 100, concept_dim: int = 256):
        super().__init__()
        
        self.encoder = PerceptionEncoder(input_dim, hidden_dim)
        self.concept_learner = ConceptLearner(num_concepts, concept_dim, hidden_dim)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(x)
        concept_logits, concept_probs = self.concept_learner.detect_concepts(encoded)
        
        return {
            'encoded': encoded,
            'concept_logits': concept_logits,
            'concept_probs': concept_probs
        }
    
    def optimize_for_t4(self):
        self.to(self.device)
        if hasattr(torch, 'compile'):
            self.encoder = torch.compile(self.encoder, mode='reduce-overhead')
            self.concept_learner = torch.compile(self.concept_learner, mode='reduce-overhead')
        return self
