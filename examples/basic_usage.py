"""Basic usage example of the neurosymbolic system."""
import torch
from neurosymbolic import NeurosymbolicSystem

def main():
    print("=== Neurosymbolic System Demo ===")
    print("\n1. Initializing system...")
    
    system = NeurosymbolicSystem(
        input_dim=512,
        hidden_dim=768,
        num_concepts=50,
        num_predicates=30
    )
    
    system.optimize_for_t4()
    
    print("\n2. Adding symbolic knowledge...")
    knowledge = {
        'facts': [
            {'name': 'mammal', 'arity': 1, 'args': ['dog']},
            {'name': 'mammal', 'arity': 1, 'args': ['cat']},
            {'name': 'has_fur', 'arity': 1, 'args': ['dog']},
            {'name': 'has_fur', 'arity': 1, 'args': ['cat']}
        ],
        'rules': [
            {
                'premises': [
                    {'name': 'mammal', 'arity': 1, 'args': ['?x']},
                    {'name': 'has_fur', 'arity': 1, 'args': ['?x']}
                ],
                'conclusion': {'name': 'warm_blooded', 'arity': 1, 'args': ['?x']},
                'confidence': 0.95
            }
        ],
        'predicate_map': {
            0: 'mammal',
            1: 'has_fur',
            2: 'warm_blooded'
        }
    }
    
    system.add_knowledge(knowledge)
    
    print("\n3. Processing perceptual input...")
    batch_size = 4
    seq_len = 16
    input_data = torch.randn(batch_size, seq_len, 512)
    
    if torch.cuda.is_available():
        input_data = input_data.cuda()
        print(f"   Using GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n4. Running integrated perception and reasoning...")
    query = {'name': 'warm_blooded', 'arity': 1, 'args': ['dog']}
    
    result = system.perceive_and_reason(input_data, query=query)
    
    print("\n=== Results ===")
    print(f"\nExtracted predicates: {len(result['extracted_predicates'])}")
    for i, pred in enumerate(result['extracted_predicates'][:3]):
        print(f"  {i+1}. Predicate ID: {pred['predicate_id']}, "
              f"Confidence: {pred['confidence']:.3f}, "
              f"Probability: {pred['probability']:.3f}")
    
    if result['reasoning_result']:
        print(f"\nReasoning result:")
        print(f"  Answer: {result['reasoning_result']['answer']}")
        print(f"  Method: {result['reasoning_result']['method']}")
        if 'confidence' in result['reasoning_result']:
            print(f"  Confidence: {result['reasoning_result']['confidence']:.3f}")
    
    if result['explanation']:
        print(f"\nExplanation:\n{result['explanation']}")
    
    print("\n5. Demonstrating pure symbolic reasoning...")
    query2 = {'name': 'mammal', 'arity': 1, 'args': ['dog']}
    reasoning = system.symbolic_module.reason(query2, method='backward')
    
    print(f"\nQuery: Is dog a mammal?")
    print(f"Answer: {reasoning['answer']}")
    print(f"Proof paths found: {len(reasoning.get('proof_paths', []))}")
    
    print("\n=== Demo Complete ===")

if __name__ == '__main__':
    main()
