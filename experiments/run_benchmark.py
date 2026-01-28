"""Main script to run comprehensive benchmarks for ICML submission."""
import sys
sys.path.append('..')

import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from neurosymbolic import NeurosymbolicSystem
from benchmarks import (
    load_benchmark_dataset,
    TransformerBaseline,
    NeuralModuleNetworkBaseline,
    RelationNetworkBaseline,
    BenchmarkRunner
)


def setup_neurosymbolic_model(num_classes: int = 1000) -> NeurosymbolicSystem:
    """Setup neurosymbolic system."""
    system = NeurosymbolicSystem(
        input_dim=512,
        hidden_dim=768,
        num_concepts=100,
        num_predicates=50
    )
    system.optimize_for_t4()
    
    # Add domain knowledge
    knowledge = {
        'facts': [
            {'name': 'shape', 'arity': 2, 'args': ['cube', 'object']},
            {'name': 'shape', 'arity': 2, 'args': ['sphere', 'object']},
            {'name': 'shape', 'arity': 2, 'args': ['cylinder', 'object']},
        ],
        'rules': [
            {
                'premises': [
                    {'name': 'shape', 'arity': 2, 'args': ['?x', 'object']},
                ],
                'conclusion': {'name': 'is_object', 'arity': 1, 'args': ['?x']},
                'confidence': 1.0
            }
        ]
    }
    system.add_knowledge(knowledge)
    
    return system


def setup_baselines(num_classes: int = 1000) -> dict:
    """Setup baseline models."""
    return {
        'Transformer': TransformerBaseline(
            input_dim=512,
            hidden_dim=768,
            num_layers=6,
            num_heads=8,
            num_classes=num_classes
        ),
        'NeuralModuleNetwork': NeuralModuleNetworkBaseline(
            visual_dim=512,
            hidden_dim=512,
            num_classes=num_classes
        ),
        'RelationNetwork': RelationNetworkBaseline(
            object_dim=512,
            hidden_dim=512,
            num_classes=num_classes
        )
    }


def create_neurosymbolic_wrapper(system: NeurosymbolicSystem, num_classes: int):
    """Wrap neurosymbolic system to match baseline interface."""
    class NeurosymbolicWrapper(torch.nn.Module):
        def __init__(self, system, num_classes):
            super().__init__()
            self.system = system
            self.classifier = torch.nn.Linear(768, num_classes)
            
        def forward(self, x):
            with torch.no_grad():
                output = self.system.neural_module(x)
            encoded = output['encoded']
            pooled = encoded.mean(dim=1)
            logits = self.classifier(pooled)
            return {'logits': logits}
    
    return NeurosymbolicWrapper(system, num_classes)


def main(args):
    """Run main benchmark."""
    print("="*80)
    print("NEUROSYMBOLIC REASONING BENCHMARK")
    print("ICML 2026 Submission")
    print("="*80)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = {}
    dataloaders = {}
    task_types = {}
    
    if 'clevr' in args.datasets:
        clevr_val = load_benchmark_dataset('clevr', split='val', max_samples=args.max_samples)
        dataloaders['CLEVR'] = DataLoader(clevr_val, batch_size=args.batch_size, shuffle=False)
        task_types['CLEVR'] = 'classification'
    
    if 'babi' in args.datasets:
        babi_val = load_benchmark_dataset('babi', split='test', max_samples=args.max_samples)
        dataloaders['bAbI'] = DataLoader(babi_val, batch_size=args.batch_size, shuffle=False)
        task_types['bAbI'] = 'qa'
    
    if 'vqa' in args.datasets:
        vqa_val = load_benchmark_dataset('vqa', split='val', max_samples=args.max_samples)
        dataloaders['VQA'] = DataLoader(vqa_val, batch_size=args.batch_size, shuffle=False)
        task_types['VQA'] = 'qa'
    
    # Estimate number of classes from data
    num_classes = args.num_classes
    
    # Setup models
    print("\nSetting up models...")
    models = {}
    
    if not args.baselines_only:
        print("  - Neurosymbolic System (Ours)")
        neurosymbolic = setup_neurosymbolic_model(num_classes)
        models['Neurosymbolic (Ours)'] = create_neurosymbolic_wrapper(neurosymbolic, num_classes)
    
    if not args.neurosymbolic_only:
        baselines = setup_baselines(num_classes)
        for name, model in baselines.items():
            print(f"  - {name}")
            models[name] = model
    
    # Run benchmark
    runner = BenchmarkRunner(device=device, output_dir=args.output_dir)
    
    results = runner.run_full_benchmark(
        models=models,
        datasets=dataloaders,
        task_types=task_types
    )
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    
    # Generate LaTeX table for paper
    generate_latex_table(results, Path(args.output_dir) / 'results_table.tex')
    print(f"\nLaTeX table saved to: {args.output_dir}/results_table.tex")


def generate_latex_table(results: dict, output_file: Path):
    """Generate LaTeX table for ICML paper."""
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Benchmark Results on Visual Reasoning Tasks}")
    latex.append(r"\label{tab:results}")
    latex.append(r"\begin{tabular}{lccc}")
    latex.append(r"\toprule")
    latex.append(r"Model & CLEVR & bAbI & VQA \\")
    latex.append(r"\midrule")
    
    # Collect all model names
    all_models = set()
    for dataset_results in results.values():
        all_models.update(dataset_results.keys())
    
    # Generate rows
    for model_name in sorted(all_models):
        row = [model_name]
        for dataset_name in ['CLEVR', 'bAbI', 'VQA']:
            if dataset_name in results:
                metrics = results[dataset_name].get(model_name, {})
                acc = metrics.get('accuracy', 0.0)
                row.append(f"{acc:.3f}")
            else:
                row.append("-")
        latex.append(" & ".join(row) + r" \\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    with open(output_file, 'w') as f:
        f.write("\n".join(latex))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run neurosymbolic benchmarks')
    parser.add_argument('--datasets', nargs='+', default=['clevr', 'babi', 'vqa'],
                       help='Datasets to benchmark on')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum samples per dataset')
    parser.add_argument('--num-classes', type=int, default=1000,
                       help='Number of output classes')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--baselines-only', action='store_true',
                       help='Only run baseline models')
    parser.add_argument('--neurosymbolic-only', action='store_true',
                       help='Only run neurosymbolic model')
    
    args = parser.parse_args()
    main(args)
