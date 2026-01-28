"""Integration layer connecting neural and symbolic modules."""
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .neural_module import NeuralModule
from .symbolic_module import SymbolicReasoner, Predicate, Rule


class NeuralSymbolicBridge(nn.Module):
    """Bridge between continuous neural and discrete symbolic representations."""
    
    def __init__(self, hidden_dim: int = 768, num_predicates: int = 50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_predicates = num_predicates
        
        self.predicate_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_predicates)
        )
        
        self.argument_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def neural_to_symbolic(self, encoded: torch.Tensor, 
                          threshold: float = 0.7) -> List[Dict[str, Any]]:
        pooled = encoded.mean(dim=1)
        
        predicate_logits = self.predicate_classifier(pooled)
        predicate_probs = torch.softmax(predicate_logits, dim=-1)
        
        confidence = self.confidence_estimator(pooled)
        
        predicates = []
        for i in range(pooled.shape[0]):
            top_pred_idx = torch.argmax(predicate_probs[i]).item()
            top_pred_prob = predicate_probs[i, top_pred_idx].item()
            
            if top_pred_prob > threshold:
                predicates.append({
                    'predicate_id': top_pred_idx,
                    'confidence': confidence[i].item(),
                    'probability': top_pred_prob
                })
        
        return predicates
    
    def symbolic_to_neural(self, predicates: List[Predicate]) -> torch.Tensor:
        embeddings = []
        for pred in predicates:
            pred_id = hash(pred.name) % self.num_predicates
            embedding = torch.randn(self.hidden_dim)
            embeddings.append(embedding)
        
        if not embeddings:
            return torch.zeros(1, self.hidden_dim)
        
        return torch.stack(embeddings)


class NeurosymbolicSystem:
    """Complete neurosymbolic system integrating neural perception and symbolic reasoning."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 768,
                 num_concepts: int = 100, num_predicates: int = 50):
        self.neural_module = NeuralModule(input_dim, hidden_dim, num_concepts)
        self.symbolic_module = SymbolicReasoner()
        self.bridge = NeuralSymbolicBridge(hidden_dim, num_predicates)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.neural_module.to(self.device)
        self.bridge.to(self.device)
        
        self.predicate_map = {}
        self.concept_map = {}
        
    def add_knowledge(self, knowledge: Dict[str, Any]):
        self.symbolic_module.add_knowledge(knowledge)
        
        if 'predicate_map' in knowledge:
            self.predicate_map.update(knowledge['predicate_map'])
        
        if 'concept_map' in knowledge:
            self.concept_map.update(knowledge['concept_map'])
    
    def perceive_and_reason(self, input_data: torch.Tensor, 
                           query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if input_data.device != self.device:
            input_data = input_data.to(self.device)
        
        with torch.no_grad():
            neural_output = self.neural_module(input_data)
        
        extracted_predicates = self.bridge.neural_to_symbolic(
            neural_output['encoded']
        )
        
        for pred_dict in extracted_predicates:
            pred_id = pred_dict['predicate_id']
            if pred_id in self.predicate_map:
                pred_name = self.predicate_map[pred_id]
                predicate = Predicate(pred_name, 0, [])
                self.symbolic_module.kb.add_fact(predicate)
        
        reasoning_result = None
        if query is not None:
            reasoning_result = self.symbolic_module.reason(query)
            explanation = self.symbolic_module.explain(query)
        else:
            explanation = None
        
        return {
            'neural_output': {
                'concepts': neural_output['concept_probs'].cpu().numpy(),
                'encoded': neural_output['encoded'].cpu().numpy()
            },
            'extracted_predicates': extracted_predicates,
            'reasoning_result': reasoning_result,
            'explanation': explanation
        }
    
    def train_step(self, input_data: torch.Tensor, labels: Dict[str, torch.Tensor],
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.neural_module.train()
        self.bridge.train()
        optimizer.zero_grad()
        
        neural_output = self.neural_module(input_data)
        
        concept_loss = nn.BCEWithLogitsLoss()(
            neural_output['concept_logits'],
            labels['concepts']
        )
        
        loss = concept_loss
        loss.backward()
        optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'concept_loss': concept_loss.item()
        }
    
    def optimize_for_t4(self):
        self.neural_module.optimize_for_t4()
        if hasattr(torch, 'compile'):
            self.bridge = torch.compile(self.bridge, mode='reduce-overhead')
        return self
    
    def save(self, path: str):
        torch.save({
            'neural_module': self.neural_module.state_dict(),
            'bridge': self.bridge.state_dict(),
            'predicate_map': self.predicate_map,
            'concept_map': self.concept_map
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.neural_module.load_state_dict(checkpoint['neural_module'])
        self.bridge.load_state_dict(checkpoint['bridge'])
        self.predicate_map = checkpoint['predicate_map']
        self.concept_map = checkpoint['concept_map']
