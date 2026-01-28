"""Evaluation metrics for benchmarking."""
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from sklearn.metrics import f1_score, precision_recall_fscore_support
import json


def compute_accuracy(predictions: List[Any], targets: List[Any]) -> float:
    """Compute accuracy."""
    correct = sum(1 for pred, target in zip(predictions, targets) if str(pred).lower() == str(target).lower())
    return correct / len(predictions) if predictions else 0.0


def compute_f1_scores(predictions: List[int], targets: List[int],
                     average: str = 'macro') -> Dict[str, float]:
    """Compute F1 scores."""
    try:
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=average, zero_division=0
        )
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    except:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }


def compute_ece(confidences: List[float], correct: List[bool],
               num_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    confidences = np.array(confidences)
    correct = np.array(correct)
    
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    
    for i in range(num_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            bin_confidence = confidences[mask].mean()
            bin_accuracy = correct[mask].mean()
            ece += mask.sum() / len(confidences) * abs(bin_confidence - bin_accuracy)
    
    return float(ece)


def generate_benchmark_report(results: Dict[str, Dict[str, Any]],
                            output_file: str = 'benchmark_results.json') -> str:
    """Generate comprehensive benchmark report.
    
    Args:
        results: Dictionary mapping model names to metric dictionaries
        output_file: Path to save JSON report
        
    Returns:
        Formatted report string
    """
    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate text report
    report = []
    report.append("=" * 80)
    report.append("BENCHMARK RESULTS")
    report.append("=" * 80)
    report.append("")
    
    # Find best model for each metric
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    best_models = {}
    for metric in all_metrics:
        best_score = -float('inf')
        best_model = None
        
        for model_name, model_results in results.items():
            if metric in model_results:
                score = model_results[metric]
                # Handle both numeric and string values
                try:
                    if isinstance(score, (int, float)):
                        if score > best_score:
                            best_score = score
                            best_model = model_name
                except (TypeError, ValueError):
                    pass
        
        if best_model is not None:
            best_models[metric] = (best_model, best_score)
    
    # Print results by model
    for model_name, model_results in results.items():
        report.append(f"\n{model_name}:")
        report.append("-" * 60)
        
        for metric, value in sorted(model_results.items()):
            is_best = best_models.get(metric, (None, None))[0] == model_name
            marker = " \u2605" if is_best else ""
            
            # Format value based on type
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            
            report.append(f"  {metric:30s}: {value_str}{marker}")
    
    # Summary
    report.append("\n" + "=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    
    for metric, (best_model, best_score) in sorted(best_models.items()):
        if isinstance(best_score, float):
            score_str = f"({best_score:.4f})"
        else:
            score_str = f"({best_score})"
        report.append(f"  {metric:30s}: {best_model:20s} {score_str}")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    return report_text


def compute_efficiency_metrics(model: torch.nn.Module,
                             sample_input: torch.Tensor,
                             num_runs: int = 100) -> Dict[str, float]:
    """Compute model efficiency metrics.
    
    Args:
        model: Model to evaluate
        sample_input: Sample input tensor
        num_runs: Number of inference runs
        
        Returns:
        Dictionary with efficiency metrics
    """
    import time
    
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)
    
    # Warmup
    model.eval()
    for _ in range(10):
        with torch.no_grad():
            _ = model(sample_input)
    
    # Measure inference time
    times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            _ = model(sample_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times.append(time.time() - start)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    return {
        'avg_inference_time_ms': float(np.mean(times) * 1000),
        'std_inference_time_ms': float(np.std(times) * 1000),
        'throughput_samples_per_sec': float(sample_input.shape[0] / np.mean(times)),
        'num_parameters': int(num_params),
        'num_parameters_millions': float(num_params / 1e6)
    }
