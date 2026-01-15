"""
Visualization utilities for training metrics and model analysis.

This module provides functions to plot training curves, visualize metrics,
and analyze model performance.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
import os


def plot_training_curves(train_metrics: Dict[int, float], valid_metrics: Dict[int, float],
                        metric_name: str = "AUC", save_path: Optional[str] = None) -> None:
    """
    Plot training and validation curves for a given metric.
    
    Creates a line plot comparing training and validation metrics across epochs.
    Useful for identifying overfitting and convergence patterns.
    
    Args:
        train_metrics (Dict[int, float]): Dictionary mapping epoch numbers to training metric values.
        valid_metrics (Dict[int, float]): Dictionary mapping epoch numbers to validation metric values.
        metric_name (str, optional): Name of the metric being plotted (e.g., 'AUC', 'Loss', 'Accuracy').
            Defaults to "AUC".
        save_path (str, optional): Path to save the plot. If None, plot is displayed but not saved.
            Defaults to None.
    
    Returns:
        None: Displays or saves the plot.
    
    Example:
        >>> train_auc = {1: 0.75, 2: 0.78, 3: 0.80}
        >>> valid_auc = {1: 0.72, 2: 0.76, 3: 0.79}
        >>> plot_training_curves(train_auc, valid_auc, "AUC", "plots/auc_curve.png")
    """
    epochs = sorted(train_metrics.keys())
    train_values = [train_metrics[ep] for ep in epochs]
    valid_values = [valid_metrics[ep] for ep in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, 'b-', label=f'Train {metric_name}', linewidth=2)
    plt.plot(epochs, valid_values, 'r-', label=f'Valid {metric_name}', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Training and Validation {metric_name} Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_all_metrics(train_metrics: Dict[str, Dict[int, float]], 
                     valid_metrics: Dict[str, Dict[int, float]],
                     save_dir: Optional[str] = None) -> None:
    """
    Plot all training metrics (AUC, Accuracy, Loss) in a single figure.
    
    Creates a 3-panel subplot showing AUC, Accuracy, and Loss curves
    for both training and validation sets.
    
    Args:
        train_metrics (Dict[str, Dict[int, float]]): Dictionary with keys 'auc', 'accuracy', 'loss'
            mapping to epoch-value dictionaries.
        valid_metrics (Dict[str, Dict[int, float]]): Dictionary with keys 'auc', 'accuracy', 'loss'
            mapping to epoch-value dictionaries.
        save_dir (str, optional): Directory to save plots. If None, plots are displayed but not saved.
            Defaults to None.
    
    Returns:
        None: Displays or saves the plots.
    
    Example:
        >>> train = {'auc': {1: 0.75}, 'accuracy': {1: 0.70}, 'loss': {1: 0.5}}
        >>> valid = {'auc': {1: 0.72}, 'accuracy': {1: 0.68}, 'loss': {1: 0.52}}
        >>> plot_all_metrics(train, valid, "plots/")
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['auc', 'accuracy', 'loss']
    metric_labels = ['AUC', 'Accuracy', 'Loss']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        if metric in train_metrics and metric in valid_metrics:
            train_dict = train_metrics[metric]
            valid_dict = valid_metrics[metric]
            
            epochs = sorted(set(list(train_dict.keys()) + list(valid_dict.keys())))
            train_values = [train_dict.get(ep, 0) for ep in epochs]
            valid_values = [valid_dict.get(ep, 0) for ep in epochs]
            
            ax.plot(epochs, train_values, 'b-', label=f'Train {label}', linewidth=2)
            ax.plot(epochs, valid_values, 'r-', label=f'Valid {label}', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(f'{label} Over Time', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'all_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metric_comparison(metrics_dict: Dict[str, float], 
                           title: str = "Metric Comparison",
                           save_path: Optional[str] = None) -> None:
    """
    Create a bar plot comparing different metric values.
    
    Useful for comparing final test metrics or comparing different model configurations.
    
    Args:
        metrics_dict (Dict[str, float]): Dictionary mapping metric names to values.
        title (str, optional): Plot title. Defaults to "Metric Comparison".
        save_path (str, optional): Path to save the plot. If None, plot is displayed but not saved.
            Defaults to None.
    
    Returns:
        None: Displays or saves the plot.
    
    Example:
        >>> metrics = {'AUC': 0.85, 'Accuracy': 0.80, 'F1-Score': 0.82}
        >>> plot_metric_comparison(metrics, "Test Set Performance")
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(names)])
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, max(values) * 1.1 if max(values) > 0 else 1.0)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
