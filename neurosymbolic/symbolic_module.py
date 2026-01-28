"""Symbolic reasoning and logic module."""
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class LogicOperator(Enum):
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    IFF = "↔"
    FORALL = "∀"
    EXISTS = "∃"


@dataclass
class Predicate:
    name: str
    arity: int
    args: List[str]
    
    def __str__(self) -> str:
        if self.arity == 0:
            return self.name
        return f"{self.name}({', '.join(self.args)})"
    
    def __hash__(self) -> int:
        return hash((self.name, tuple(self.args)))


@dataclass
class Rule:
    premises: List[Predicate]
    conclusion: Predicate
    confidence: float = 1.0
    
    def __str__(self) -> str:
        premises_str = " ∧ ".join(str(p) for p in self.premises)
        return f"{premises_str} → {self.conclusion} [conf: {self.confidence:.2f}]"


class KnowledgeBase:
    
    def __init__(self):
        self.facts: Set[Predicate] = set()
        self.rules: List[Rule] = []
        self.concepts: Dict[str, Any] = {}
        
    def add_fact(self, fact: Predicate):
        self.facts.add(fact)
    
    def add_rule(self, rule: Rule):
        self.rules.append(rule)
    
    def query(self, predicate: Predicate, max_depth: int = 5) -> Tuple[bool, float]:
        if predicate in self.facts:
            return True, 1.0
        return self._forward_chain(predicate, max_depth)
    
    def _forward_chain(self, goal: Predicate, max_depth: int, 
                       current_depth: int = 0) -> Tuple[bool, float]:
        if current_depth >= max_depth:
            return False, 0.0
        
        for rule in self.rules:
            if self._unify(rule.conclusion, goal):
                all_satisfied = True
                min_confidence = rule.confidence
                
                for premise in rule.premises:
                    satisfied, conf = self.query(premise, max_depth - current_depth - 1)
                    if not satisfied:
                        all_satisfied = False
                        break
                    min_confidence = min(min_confidence, conf)
                
                if all_satisfied:
                    return True, min_confidence * rule.confidence
        
        return False, 0.0
    
    def _unify(self, pred1: Predicate, pred2: Predicate) -> bool:
        if pred1.name != pred2.name or pred1.arity != pred2.arity:
            return False
        return pred1.args == pred2.args or any(a.startswith('?') for a in pred1.args)
    
    def backward_chain(self, goal: Predicate, max_depth: int = 5) -> List[List[Predicate]]:
        paths = []
        self._backward_chain_recursive(goal, [], paths, max_depth)
        return paths
    
    def _backward_chain_recursive(self, goal: Predicate, current_path: List[Predicate],
                                  paths: List[List[Predicate]], max_depth: int):
        if len(current_path) >= max_depth:
            return
        
        if goal in self.facts:
            paths.append(current_path + [goal])
            return
        
        for rule in self.rules:
            if self._unify(rule.conclusion, goal):
                new_path = current_path + [goal]
                for premise in rule.premises:
                    self._backward_chain_recursive(premise, new_path, paths, max_depth)


class SymbolicReasoner:
    
    def __init__(self):
        self.kb = KnowledgeBase()
        
    def add_knowledge(self, knowledge: Dict[str, Any]):
        if 'facts' in knowledge:
            for fact_dict in knowledge['facts']:
                fact = Predicate(**fact_dict)
                self.kb.add_fact(fact)
        
        if 'rules' in knowledge:
            for rule_dict in knowledge['rules']:
                premises = [Predicate(**p) for p in rule_dict['premises']]
                conclusion = Predicate(**rule_dict['conclusion'])
                confidence = rule_dict.get('confidence', 1.0)
                rule = Rule(premises, conclusion, confidence)
                self.kb.add_rule(rule)
    
    def reason(self, query: Dict[str, Any], method: str = 'forward') -> Dict[str, Any]:
        predicate = Predicate(**query)
        
        if method == 'forward':
            is_true, confidence = self.kb.query(predicate)
            return {
                'answer': is_true,
                'confidence': confidence,
                'method': 'forward_chaining'
            }
        elif method == 'backward':
            paths = self.kb.backward_chain(predicate)
            return {
                'answer': len(paths) > 0,
                'proof_paths': [[str(p) for p in path] for path in paths],
                'method': 'backward_chaining'
            }
        else:
            raise ValueError(f"Unknown reasoning method: {method}")
    
    def explain(self, query: Dict[str, Any]) -> str:
        predicate = Predicate(**query)
        paths = self.kb.backward_chain(predicate)
        
        if not paths:
            return f"Cannot prove {predicate}"
        
        explanation = f"Proving {predicate}:\n"
        for i, path in enumerate(paths[:3], 1):
            explanation += f"\nPath {i}:\n"
            for step in path:
                explanation += f"  - {step}\n"
        
        return explanation
