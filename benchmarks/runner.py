"""Benchmark runner for comprehensive evaluation."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm
import time
import json
from pathlib import Path

from .metrics import (
    compute_accuracy,
    compute_f1_scores,
    compute_reasoning_metrics,
    compute_efficiency_metrics,
    generate_benchmark_report
)


class BenchmarkRunner:
    """Run comprehensive benchmarks comparing multiple models."""
    
    def __init__(self, device: str = 'cuda', output_dir: str = './benchmark_results'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        
    def evaluate_model(self, model: nn.Module, dataloader: DataLoader,
                      model_name: str, task_type: str = 'classification') -> Dict[str, float]:
        """Evaluate a single model.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation data
            model_name: Name of the model
            task_type: Type of task ('classification', 'qa', 'reasoning')
            
        Returns:
            Dictionary of metrics
        """
        model.to(self.device)
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_logits = []
        
        print(f"\nEvaluating {model_name}...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{model_name}"):
                # Handle different batch formats
                if 'visual_features' in batch:
                    inputs = batch['visual_features'].to(self.device)
                elif 'features' in batch:
                    inputs = batch['features'].to(self.device)
                else:
                    raise ValueError("Batch must contain 'visual_features' or 'features'")
                
                # Forward pass
                outputs = model(inputs)
                
                if 'logits' in outputs:
                    logits = outputs['logits']
                    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_logits.extend(logits.cpu().numpy())
                else:
                    predictions = outputs.get('predictions', [])
                
                all_predictions.extend(predictions)
                
                # Get targets
                if 'answer' in batch:
                    # Convert answers to indices if needed
                    targets = self._process_answers(batch['answer'])
                    all_targets.extend(targets)
        
        # Compute metrics
        metrics = {}
        
        if task_type == 'classification' and all_targets:
            metrics['accuracy'] = compute_accuracy(all_predictions, all_targets)
            
            if len(set(all_targets)) > 2:  # Multi-class
                f1_metrics = compute_f1_scores(all_predictions, all_targets)
                metrics.update(f1_metrics)
        
        # Compute efficiency metrics
        sample_batch = next(iter(dataloader))
        if 'visual_features' in sample_batch:
            sample_input = sample_batch['visual_features'][:4].to(self.device)
        else:
            sample_input = sample_batch['features'][:4].to(self.device)
        
        efficiency = compute_efficiency_metrics(model, sample_input)
        metrics.update(efficiency)
        
        return metrics
    
    def _process_answers(self, answers: List[Any]) -> List[int]:
        """Convert answers to integer indices."""
        if isinstance(answers[0], torch.Tensor):
            return answers.cpu().numpy().tolist()
        elif isinstance(answers[0], str):
            # Create answer vocabulary
            unique_answers = sorted(set(answers))
            answer_to_idx = {ans: idx for idx, ans in enumerate(unique_answers)}
            return [answer_to_idx[ans] for ans in answers]
        else:
            return answers
    
    def run_full_benchmark(self, models: Dict[str, nn.Module],
                          datasets: Dict[str, DataLoader],
                          task_types: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Run full benchmark across multiple models and datasets.
        
        Args:
            models: Dictionary mapping model names to model instances
            datasets: Dictionary mapping dataset names to dataloaders
            task_types: Dictionary mapping dataset names to task types
            
        Returns:
            Nested dictionary of results
        """
        all_results = {}
        
        for dataset_name, dataloader in datasets.items():
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*80}")
            
            dataset_results = {}
            
            for model_name, model in models.items():
                try:
                    metrics = self.evaluate_model(
                        model, dataloader, model_name,
                        task_type=task_types.get(dataset_name, 'classification')
                    )
                    dataset_results[model_name] = metrics
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
                    dataset_results[model_name] = {'error': str(e)}
            
            all_results[dataset_name] = dataset_results
        
        # Save results
        self._save_results(all_results)
        
        # Generate reports
        self._generate_reports(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict):
        """Save results to JSON."""
        output_file = self.output_dir / 'full_benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    def _generate_reports(self, results: Dict):
        """Generate detailed reports."""
        for dataset_name, dataset_results in results.items():
            report_file = self.output_dir / f'{dataset_name}_report.txt'
            
            with open(report_file, 'w') as f:
                report = generate_benchmark_report(
                    dataset_results,
                    str(self.output_dir / f'{dataset_name}_results.json')
                )
                f.write(report)
            
            print(f"\nReport for {dataset_name} saved to {report_file}")
    
    def train_and_evaluate(self, model: nn.Module, train_loader: DataLoader,
                          val_loader: DataLoader, model_name: str,
                          num_epochs: int = 10, lr: float = 1e-4) -> Dict[str, Any]:
        """Train and evaluate a model.
        
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader  
            model_name: Name of the model
            num_epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Dictionary with training history and final metrics
        """
        model.to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'val_accuracy': []}
        
        print(f"\nTraining {model_name}...")
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                
                if 'visual_features' in batch:
                    inputs = batch['visual_features'].to(self.device)
                else:
                    inputs = batch['features'].to(self.device)
                
                # Get targets
                targets = self._process_answers(batch['answer'])
                targets = torch.tensor(targets).to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                
                if 'logits' in outputs:
                    logits = outputs['logits']
                    loss = criterion(logits, targets)
                else:
                    continue
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_metrics = self.evaluate_model(model, val_loader, model_name)
            val_accuracy = val_metrics.get('accuracy', 0.0)
            history['val_accuracy'].append(val_accuracy)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
                  f"Val Accuracy = {val_accuracy:.4f}")
        
        return {
            'history': history,
            'final_metrics': val_metrics
        }
