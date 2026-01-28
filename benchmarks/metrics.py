"""Evaluation metrics for benchmarking."""
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from sklearn.metrics import f1_score, precision_recall_fscore_support
import json


def compute_accuracy(predictions: List[Any], targets: List[Any]) -> float:
    """Compute accuracy."""
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    return correct / len(predictions) if predictions else 0.0


def compute_f1_scores(predictions: List[int], targets: List[int],
                     average: str = 'macro') -> Dict[str, float]:
    """Compute F1 scores."""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=average, zero_division=0
    )
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def compute_reasoning_metrics(predictions: List[Dict], 
                            targets: List[Dict]) -> Dict[str, float]:
    """Compute reasoning-specific metrics.
    
    Args:
        predictions: List of prediction dictionaries with 'answer' and 'explanation'
        targets: List of target dictionaries
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Answer accuracy
    pred_answers = [p['answer'] for p in predictions]
    target_answers = [t['answer'] for t in targets]
    metrics['answer_accuracy'] = compute_accuracy(pred_answers, target_answers)
    
    # Explanation quality (if available)
    if 'explanation' in predictions[0]:
        explanation_lengths = [len(p.get('explanation', '').split()) for p in predictions]
        metrics['avg_explanation_length'] = np.mean(explanation_lengths)
    
    # Confidence calibration
    if 'confidence' in predictions[0]:
        confidences = [p['confidence'] for p in predictions]
        correct = [p == t for p, t in zip(pred_answers, target_answers)]
        
        # Expected Calibration Error (ECE)
        metrics['ece'] = compute_ece(confidences, correct)
        metrics['avg_confidence'] = np.mean(confidences)
    
    return metrics


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


def compute_program_accuracy(pred_programs: List[List[str]],
                           target_programs: List[List[str]]) -> float:
    """Compute program execution accuracy."""
    correct = 0
    for pred, target in zip(pred_programs, target_programs):
        if pred == target:
            correct += 1
    return correct / len(pred_programs) if pred_programs else 0.0


def generate_benchmark_report(results: Dict[str, Dict[str, float]],
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
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        best_models[metric] = (best_model, best_score)
    
    # Print results by model
    for model_name, model_results in results.items():
        report.append(f"\n{model_name}:")
        report.append("-" * 60)
        
        for metric, value in sorted(model_results.items()):
            is_best = best_models[metric][0] == model_name
            marker = " ★" if is_best else ""
            report.append(f"  {metric:30s}: {value:.4f}{marker}")
    
    # Summary
    report.append("\n" + "=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    
    for metric, (best_model, best_score) in sorted(best_models.items()):
        report.append(f"  {metric:30s}: {best_model:20s} ({best_score:.4f})")
    
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
        'avg_inference_time_ms': np.mean(times) * 1000,
        'std_inference_time_ms': np.std(times) * 1000,
        'throughput_samples_per_sec': sample_input.shape[0] / np.mean(times),
        'num_parameters': num_params,
        'num_parameters_millions': num_params / 1e6
    }
