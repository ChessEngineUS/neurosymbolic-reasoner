"""Tests for neural module."""
import pytest
import torch
from neurosymbolic.neural_module import NeuralModule, PerceptionEncoder, ConceptLearner


class TestPerceptionEncoder:
    
    def test_initialization(self):
        encoder = PerceptionEncoder(input_dim=512, hidden_dim=768)
        assert encoder.input_dim == 512
        assert encoder.hidden_dim == 768
    
    def test_forward_pass(self):
        encoder = PerceptionEncoder(input_dim=512, hidden_dim=768)
        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, 512)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, seq_len, 768)
        assert not torch.isnan(output).any()
    
    def test_with_mask(self):
        encoder = PerceptionEncoder(input_dim=512, hidden_dim=768)
        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, 512)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -2:] = True
        
        output = encoder(x, mask=mask)
        
        assert output.shape == (batch_size, seq_len, 768)


class TestConceptLearner:
    
    def test_initialization(self):
        learner = ConceptLearner(num_concepts=100, concept_dim=256, hidden_dim=768)
        assert learner.num_concepts == 100
        assert learner.concept_dim == 256
    
    def test_detect_concepts(self):
        learner = ConceptLearner(num_concepts=100, concept_dim=256, hidden_dim=768)
        batch_size, seq_len = 4, 16
        encoded = torch.randn(batch_size, seq_len, 768)
        
        logits, probs = learner.detect_concepts(encoded)
        
        assert logits.shape == (batch_size, 100)
        assert probs.shape == (batch_size, 100)
        assert (probs >= 0).all() and (probs <= 1).all()
    
    def test_compose_concepts(self):
        learner = ConceptLearner(num_concepts=100, concept_dim=256, hidden_dim=768)
        
        # Single concept
        result = learner.compose_concepts([5])
        assert result.shape[-1] == 256
        
        # Multiple concepts
        result = learner.compose_concepts([5, 10, 15])
        assert result.shape[-1] == 256


class TestNeuralModule:
    
    def test_initialization(self):
        module = NeuralModule(input_dim=512, hidden_dim=768)
        assert module.encoder is not None
        assert module.concept_learner is not None
    
    def test_forward_pass(self):
        module = NeuralModule(input_dim=512, hidden_dim=768, num_concepts=50)
        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, 512)
        
        output = module(x)
        
        assert 'encoded' in output
        assert 'concept_logits' in output
        assert 'concept_probs' in output
        assert output['encoded'].shape == (batch_size, seq_len, 768)
        assert output['concept_logits'].shape == (batch_size, 50)
        assert output['concept_probs'].shape == (batch_size, 50)
    
    def test_gpu_optimization(self):
        module = NeuralModule(input_dim=512, hidden_dim=768)
        module_opt = module.optimize_for_t4()
        assert module_opt is not None
