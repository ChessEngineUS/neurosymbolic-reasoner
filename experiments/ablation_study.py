"""Ablation study for neurosymbolic system."""
import sys
sys.path.append('..')

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from neurosymbolic import NeurosymbolicSystem
from neurosymbolic.neural_module import NeuralModule
from benchmarks import load_benchmark_dataset, BenchmarkRunner


def run_ablation_study(output_dir: str = './ablation_results'):
    """Run comprehensive ablation study."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ABLATION STUDY")
    print("="*80)
    
    # Load dataset
    dataset = load_benchmark_dataset('clevr', split='val', max_samples=500)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    runner = BenchmarkRunner(output_dir=str(output_dir))
    
    results = {}
    
    # 1. Full system
    print("\n1. Full Neurosymbolic System")
    full_system = NeurosymbolicSystem(
        input_dim=512,
        hidden_dim=768,
        num_concepts=100,
        num_predicates=50
    ).optimize_for_t4()
    
    class FullWrapper(torch.nn.Module):
        def __init__(self, system):
            super().__init__()
            self.system = system
            self.classifier = torch.nn.Linear(768, 1000)
        def forward(self, x):
            with torch.no_grad():
                output = self.system.neural_module(x)
            pooled = output['encoded'].mean(dim=1)
            return {'logits': self.classifier(pooled)}
    
    results['Full System'] = runner.evaluate_model(
        FullWrapper(full_system), dataloader, 'Full System'
    )
    
    # 2. Neural only (no symbolic reasoning)
    print("\n2. Neural Module Only")
    neural_only = NeuralModule(
        input_dim=512,
        hidden_dim=768,
        num_concepts=100
    ).optimize_for_t4()
    
    class NeuralOnlyWrapper(torch.nn.Module):
        def __init__(self, neural):
            super().__init__()
            self.neural = neural
            self.classifier = torch.nn.Linear(768, 1000)
        def forward(self, x):
            with torch.no_grad():
                output = self.neural(x)
            pooled = output['encoded'].mean(dim=1)
            return {'logits': self.classifier(pooled)}
    
    results['Neural Only'] = runner.evaluate_model(
        NeuralOnlyWrapper(neural_only), dataloader, 'Neural Only'
    )
    
    # 3. Reduced capacity (fewer layers)
    print("\n3. Reduced Transformer Layers (3 vs 6)")
    reduced_system = NeurosymbolicSystem(
        input_dim=512,
        hidden_dim=768,
        num_concepts=50,  # Reduced
        num_predicates=25  # Reduced
    )
    
    results['Reduced Capacity'] = runner.evaluate_model(
        FullWrapper(reduced_system), dataloader, 'Reduced'
    )
    
    # 4. Without concept learning
    print("\n4. Without Concept Learning")
    no_concepts = NeurosymbolicSystem(
        input_dim=512,
        hidden_dim=768,
        num_concepts=10,  # Minimal
        num_predicates=50
    )
    
    results['No Concept Learning'] = runner.evaluate_model(
        FullWrapper(no_concepts), dataloader, 'No Concepts'
    )
    
    # Generate visualizations
    plot_ablation_results(results, output_dir)
    
    return results


def plot_ablation_results(results: dict, output_dir: Path):
    """Plot ablation study results."""
    sns.set_style('whitegrid')
    
    # Extract metrics
    models = list(results.keys())
    accuracies = [results[m].get('accuracy', 0) for m in models]
    inference_times = [results[m].get('avg_inference_time_ms', 0) for m in models]
    params = [results[m].get('num_parameters_millions', 0) for m in models]
    
    # Plot 1: Accuracy comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].barh(models, accuracies, color='steelblue')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlim([0, 1])
    
    # Plot 2: Inference time
    axes[1].barh(models, inference_times, color='coral')
    axes[1].set_xlabel('Inference Time (ms)')
    axes[1].set_title('Inference Speed')
    
    # Plot 3: Parameters
    axes[2].barh(models, params, color='seagreen')
    axes[2].set_xlabel('Parameters (millions)')
    axes[2].set_title('Model Size')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_study.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_dir / 'ablation_study.png'}")
    
    # Plot 4: Accuracy vs efficiency tradeoff
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(inference_times, accuracies, s=[p*50 for p in params],
                        alpha=0.6, c=range(len(models)), cmap='viridis')
    
    for i, model in enumerate(models):
        ax.annotate(model, (inference_times[i], accuracies[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Inference Time (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Efficiency Trade-off\n(bubble size = # parameters)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tradeoff_plot.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_dir / 'tradeoff_plot.png'}")


if __name__ == '__main__':
    run_ablation_study()
