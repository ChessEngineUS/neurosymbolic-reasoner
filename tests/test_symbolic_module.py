"""Tests for symbolic module."""
import pytest
from neurosymbolic.symbolic_module import (
    Predicate, Rule, KnowledgeBase, SymbolicReasoner, LogicOperator
)


class TestPredicate:
    
    def test_initialization(self):
        pred = Predicate(name='mammal', arity=1, args=['dog'])
        assert pred.name == 'mammal'
        assert pred.arity == 1
        assert pred.args == ['dog']
    
    def test_string_representation(self):
        pred = Predicate(name='mammal', arity=1, args=['dog'])
        assert str(pred) == 'mammal(dog)'
        
        pred_zero = Predicate(name='true', arity=0, args=[])
        assert str(pred_zero) == 'true'
    
    def test_hashing(self):
        pred1 = Predicate(name='mammal', arity=1, args=['dog'])
        pred2 = Predicate(name='mammal', arity=1, args=['dog'])
        assert hash(pred1) == hash(pred2)


class TestRule:
    
    def test_initialization(self):
        premises = [Predicate(name='mammal', arity=1, args=['?x'])]
        conclusion = Predicate(name='warm_blooded', arity=1, args=['?x'])
        rule = Rule(premises=premises, conclusion=conclusion, confidence=0.9)
        
        assert len(rule.premises) == 1
        assert rule.conclusion.name == 'warm_blooded'
        assert rule.confidence == 0.9
    
    def test_string_representation(self):
        premises = [Predicate(name='mammal', arity=1, args=['dog'])]
        conclusion = Predicate(name='warm_blooded', arity=1, args=['dog'])
        rule = Rule(premises=premises, conclusion=conclusion)
        
        rule_str = str(rule)
        assert 'mammal(dog)' in rule_str
        assert 'warm_blooded(dog)' in rule_str


class TestKnowledgeBase:
    
    def test_initialization(self):
        kb = KnowledgeBase()
        assert len(kb.facts) == 0
        assert len(kb.rules) == 0
    
    def test_add_fact(self):
        kb = KnowledgeBase()
        fact = Predicate(name='mammal', arity=1, args=['dog'])
        kb.add_fact(fact)
        
        assert fact in kb.facts
        assert len(kb.facts) == 1
    
    def test_add_rule(self):
        kb = KnowledgeBase()
        premises = [Predicate(name='mammal', arity=1, args=['?x'])]
        conclusion = Predicate(name='warm_blooded', arity=1, args=['?x'])
        rule = Rule(premises=premises, conclusion=conclusion)
        
        kb.add_rule(rule)
        assert len(kb.rules) == 1
    
    def test_query_direct_fact(self):
        kb = KnowledgeBase()
        fact = Predicate(name='mammal', arity=1, args=['dog'])
        kb.add_fact(fact)
        
        is_true, confidence = kb.query(fact)
        assert is_true
        assert confidence == 1.0
    
    def test_query_inference(self):
        kb = KnowledgeBase()
        
        # Add facts
        kb.add_fact(Predicate(name='mammal', arity=1, args=['dog']))
        kb.add_fact(Predicate(name='has_fur', arity=1, args=['dog']))
        
        # Add rule
        premises = [
            Predicate(name='mammal', arity=1, args=['?x']),
            Predicate(name='has_fur', arity=1, args=['?x'])
        ]
        conclusion = Predicate(name='warm_blooded', arity=1, args=['?x'])
        rule = Rule(premises=premises, conclusion=conclusion, confidence=0.95)
        kb.add_rule(rule)
        
        # Query inferred fact
        query = Predicate(name='warm_blooded', arity=1, args=['dog'])
        is_true, confidence = kb.query(query)
        
        assert is_true
        assert confidence == 0.95
    
    def test_backward_chain(self):
        kb = KnowledgeBase()
        kb.add_fact(Predicate(name='mammal', arity=1, args=['dog']))
        
        premises = [Predicate(name='mammal', arity=1, args=['?x'])]
        conclusion = Predicate(name='animal', arity=1, args=['?x'])
        kb.add_rule(Rule(premises=premises, conclusion=conclusion))
        
        goal = Predicate(name='animal', arity=1, args=['dog'])
        paths = kb.backward_chain(goal)
        
        assert len(paths) > 0


class TestSymbolicReasoner:
    
    def test_initialization(self):
        reasoner = SymbolicReasoner()
        assert reasoner.kb is not None
    
    def test_add_knowledge(self):
        reasoner = SymbolicReasoner()
        knowledge = {
            'facts': [
                {'name': 'mammal', 'arity': 1, 'args': ['dog']}
            ],
            'rules': [
                {
                    'premises': [{'name': 'mammal', 'arity': 1, 'args': ['?x']}],
                    'conclusion': {'name': 'animal', 'arity': 1, 'args': ['?x']},
                    'confidence': 1.0
                }
            ]
        }
        
        reasoner.add_knowledge(knowledge)
        assert len(reasoner.kb.facts) == 1
        assert len(reasoner.kb.rules) == 1
    
    def test_reason_forward(self):
        reasoner = SymbolicReasoner()
        reasoner.kb.add_fact(Predicate(name='mammal', arity=1, args=['dog']))
        
        query = {'name': 'mammal', 'arity': 1, 'args': ['dog']}
        result = reasoner.reason(query, method='forward')
        
        assert result['answer'] == True
        assert result['method'] == 'forward_chaining'
        assert 'confidence' in result
    
    def test_reason_backward(self):
        reasoner = SymbolicReasoner()
        reasoner.kb.add_fact(Predicate(name='mammal', arity=1, args=['dog']))
        
        query = {'name': 'mammal', 'arity': 1, 'args': ['dog']}
        result = reasoner.reason(query, method='backward')
        
        assert result['answer'] == True
        assert result['method'] == 'backward_chaining'
    
    def test_explain(self):
        reasoner = SymbolicReasoner()
        reasoner.kb.add_fact(Predicate(name='mammal', arity=1, args=['dog']))
        
        query = {'name': 'mammal', 'arity': 1, 'args': ['dog']}
        explanation = reasoner.explain(query)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
