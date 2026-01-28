"""Training example for the neurosymbolic system."""
import torch
import torch.optim as optim
from neurosymbolic import NeurosymbolicSystem
from tqdm import tqdm

def generate_synthetic_data(num_samples: int = 100, seq_len: int = 16, 
                           input_dim: int = 512, num_concepts: int = 50):
    """Generate synthetic training data."""
    inputs = torch.randn(num_samples, seq_len, input_dim)
    
    concept_labels = torch.zeros(num_samples, num_concepts)
    for i in range(num_samples):
        num_active = torch.randint(1, 6, (1,)).item()
        active_concepts = torch.randperm(num_concepts)[:num_active]
        concept_labels[i, active_concepts] = 1.0
    
    return inputs, concept_labels

def train_epoch(system, dataloader, optimizer, device):
    """Train for one epoch."""
    total_loss = 0.0
    num_batches = 0
    
    for batch_inputs, batch_labels in dataloader:
        batch_inputs = batch_inputs.to(device)
        labels_dict = {'concepts': batch_labels.to(device)}
        
        losses = system.train_step(batch_inputs, labels_dict, optimizer)
        total_loss += losses['total_loss']
        num_batches += 1
    
    return total_loss / num_batches

def main():
    print("=== Neurosymbolic System Training ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n1. Initializing system...")
    system = NeurosymbolicSystem(
        input_dim=512,
        hidden_dim=768,
        num_concepts=50,
        num_predicates=30
    ).optimize_for_t4()
    
    print("\n2. Generating synthetic training data...")
    num_samples = 500
    batch_size = 16
    
    all_inputs, all_labels = generate_synthetic_data(
        num_samples=num_samples,
        seq_len=16,
        input_dim=512,
        num_concepts=50
    )
    
    dataset = torch.utils.data.TensorDataset(all_inputs, all_labels)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    print(f"   Training samples: {num_samples}")
    print(f"   Batch size: {batch_size}")
    
    print("\n3. Setting up optimizer...")
    optimizer = optim.AdamW(
        list(system.neural_module.parameters()) + 
        list(system.bridge.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    
    print("\n4. Training...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        avg_loss = train_epoch(system, dataloader, optimizer, device)
        print(f"   Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    print("\n5. Saving model...")
    system.save('neurosymbolic_model.pt')
    print("   Model saved to neurosymbolic_model.pt")
    
    print("\n6. Testing inference...")
    test_input = torch.randn(4, 16, 512).to(device)
    
    with torch.no_grad():
        result = system.perceive_and_reason(test_input)
    
    print(f"   Extracted predicates: {len(result['extracted_predicates'])}")
    print(f"   Concept activations shape: {result['neural_output']['concepts'].shape}")
    
    print("\n=== Training Complete ===")

if __name__ == '__main__':
    main()
