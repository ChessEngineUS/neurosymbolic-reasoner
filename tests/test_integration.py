"""Tests for integration module."""
import pytest
import torch
from neurosymbolic.integration import NeuralSymbolicBridge, NeurosymbolicSystem
from neurosymbolic.symbolic_module import Predicate


class TestNeuralSymbolicBridge:
    
    def test_initialization(self):
        bridge = NeuralSymbolicBridge(hidden_dim=768, num_predicates=50)
        assert bridge.hidden_dim == 768
        assert bridge.num_predicates == 50
    
    def test_neural_to_symbolic(self):
        bridge = NeuralSymbolicBridge(hidden_dim=768, num_predicates=50)
        batch_size, seq_len = 4, 16
        encoded = torch.randn(batch_size, seq_len, 768)
        
        predicates = bridge.neural_to_symbolic(encoded, threshold=0.5)
        
        assert isinstance(predicates, list)
        for pred in predicates:
            assert 'predicate_id' in pred
            assert 'confidence' in pred
            assert 'probability' in pred
    
    def test_symbolic_to_neural(self):
        bridge = NeuralSymbolicBridge(hidden_dim=768, num_predicates=50)
        predicates = [
            Predicate(name='mammal', arity=1, args=['dog']),
            Predicate(name='has_fur', arity=1, args=['dog'])
        ]
        
        embedding = bridge.symbolic_to_neural(predicates)
        
        assert embedding.shape[-1] == 768
        assert not torch.isnan(embedding).any()


class TestNeurosymbolicSystem:
    
    def test_initialization(self):
        system = NeurosymbolicSystem(
            input_dim=512,
            hidden_dim=768,
            num_concepts=50,
            num_predicates=30
        )
        
        assert system.neural_module is not None
        assert system.symbolic_module is not None
        assert system.bridge is not None
    
    def test_add_knowledge(self):
        system = NeurosymbolicSystem()
        knowledge = {
            'facts': [{'name': 'test', 'arity': 0, 'args': []}],
            'predicate_map': {0: 'test'}
        }
        
        system.add_knowledge(knowledge)
        assert len(system.predicate_map) > 0
    
    def test_perceive_and_reason(self):
        system = NeurosymbolicSystem(
            input_dim=512,
            hidden_dim=768,
            num_concepts=50,
            num_predicates=30
        )
        
        # Add knowledge
        knowledge = {
            'facts': [{'name': 'mammal', 'arity': 1, 'args': ['dog']}],
            'predicate_map': {0: 'mammal'}
        }
        system.add_knowledge(knowledge)
        
        # Test perception and reasoning
        input_data = torch.randn(4, 16, 512)
        query = {'name': 'mammal', 'arity': 1, 'args': ['dog']}
        
        result = system.perceive_and_reason(input_data, query=query)
        
        assert 'neural_output' in result
        assert 'extracted_predicates' in result
        assert 'reasoning_result' in result
        assert result['reasoning_result'] is not None
    
    def test_train_step(self):
        system = NeurosymbolicSystem(
            input_dim=512,
            hidden_dim=768,
            num_concepts=50
        )
        
        batch_size = 4
        input_data = torch.randn(batch_size, 16, 512)
        labels = {'concepts': torch.zeros(batch_size, 50)}
        
        optimizer = torch.optim.Adam(
            list(system.neural_module.parameters()) + 
            list(system.bridge.parameters()),
            lr=1e-4
        )
        
        losses = system.train_step(input_data, labels, optimizer)
        
        assert 'total_loss' in losses
        assert 'concept_loss' in losses
        assert losses['total_loss'] >= 0
    
    def test_save_load(self, tmp_path):
        system = NeurosymbolicSystem()
        
        # Save
        save_path = tmp_path / "model.pt"
        system.save(str(save_path))
        assert save_path.exists()
        
        # Load
        system2 = NeurosymbolicSystem()
        system2.load(str(save_path))
        assert system2.predicate_map == system.predicate_map
    
    def test_optimize_for_t4(self):
        system = NeurosymbolicSystem()
        system_opt = system.optimize_for_t4()
        assert system_opt is not None
