"""
Module Đánh giá Models cho Phát hiện Gian lận Thẻ Tín dụng

Module này cung cấp các hàm để:
1. Tính toán metrics (Precision, Recall, F1, AUC)
2. Vẽ biểu đồ (Confusion Matrix, ROC Curve)
3. So sánh và đánh giá hiệu năng của các models

Giải thích Metrics:
- Precision: Trong số các dự đoán "gian lận", có bao nhiêu là đúng
- Recall: Trong số các giao dịch gian lận thực sự, model bắt được bao nhiêu
- F1-Score: Trung bình của Precision và Recall
- AUC: Khả năng phân biệt giữa gian lận và bình thường (0-1, càng cao càng tốt)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    accuracy_score, classification_report
)
from pathlib import Path


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Tính toán các metrics để đánh giá model
    
    Metrics được tính:
    - Accuracy: Tỉ lệ dự đoán đúng tổng thể
    - Precision: Trong số các dự đoán "gian lận", có bao nhiêu là đúng
      Ví dụ: Nếu model dự đoán 10 giao dịch là gian lận, và 8 cái đúng → Precision = 0.8
    - Recall: Trong số các giao dịch gian lận thực sự, model bắt được bao nhiêu
      Ví dụ: Có 10 giao dịch gian lận thực sự, model bắt được 7 → Recall = 0.7
    - F1-Score: Trung bình điều hòa của Precision và Recall (cân bằng cả hai)
    - AUC: Khả năng phân biệt giữa gian lận và bình thường (0-1)
    
    Parameters:
    -----------
    y_true : array-like
        Nhãn thực sự (ground truth) - 0 = bình thường, 1 = gian lận
    y_pred : array-like
        Nhãn dự đoán từ model - 0 = bình thường, 1 = gian lận
    y_pred_proba : array-like, tùy chọn
        Xác suất dự đoán là gian lận (0.0 đến 1.0) - dùng để tính AUC
    
    Returns:
    --------
    metrics : dict
        Dictionary chứa tất cả metrics đã tính
        Ví dụ: {'accuracy': 0.99, 'precision': 0.85, 'recall': 0.80, 'f1': 0.82, 'auc': 0.95}
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Calculate AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['auc'] = None
    else:
        metrics['auc'] = None
    
    return metrics


def get_metrics_dict(y_true, y_pred, y_pred_proba=None, model_name='Model'):
    """
    Get metrics as a dictionary with model name.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for positive class
    model_name : str, default='Model'
        Name of the model
    
    Returns:
    --------
    metrics_dict : dict
        Dictionary with model name and metrics
    """
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    metrics['model'] = model_name
    return metrics


def print_metrics(y_true, y_pred, y_pred_proba=None, model_name='Model'):
    """
    Print metrics in a formatted table.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for positive class
    model_name : str, default='Model'
        Name of the model
    """
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    print(f"\n{'='*50}")
    print(f"Metrics for {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    if metrics['auc'] is not None:
        print(f"AUC:       {metrics['auc']:.4f}")
    print(f"{'='*50}\n")


def plot_confusion_matrix(y_true, y_pred, model_name='Model', 
                          save_path=None, figsize=(8, 6)):
    """
    Plot confusion matrix with annotations.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str, default='Model'
        Name of the model
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(8, 6)
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curve(y_true, y_pred_proba, model_name='Model',
                   save_path=None, figsize=(8, 6), ax=None):
    """
    Plot ROC curve with AUC score.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    model_name : str, default='Model'
        Name of the model
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(8, 6)
        Figure size (only used if ax is None)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object (None if ax is provided)
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        fig = None
        return_fig = False
    
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if return_fig:
        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        return fig
    else:
        return None


def plot_precision_recall_curve(y_true, y_pred_proba, model_name='Model',
                               save_path=None, figsize=(8, 6)):
    """
    Plot Precision-Recall curve.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    model_name : str, default='Model'
        Name of the model
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(8, 6)
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, label=model_name, linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    
    return fig


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name='Model',
                   save_dir=None, plot_cm=True, plot_roc=True, plot_pr=False):
    """
    Comprehensive model evaluation function.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for positive class
    model_name : str, default='Model'
        Name of the model
    save_dir : str or Path, optional
        Directory to save plots. If None, plots are not saved.
    plot_cm : bool, default=True
        Whether to plot confusion matrix
    plot_roc : bool, default=True
        Whether to plot ROC curve
    plot_pr : bool, default=False
        Whether to plot Precision-Recall curve
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all calculated metrics
    figures : dict
        Dictionary containing figure objects
    """
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Print metrics
    print_metrics(y_true, y_pred, y_pred_proba, model_name)
    
    # Prepare save paths
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        cm_path = save_dir / f'{model_name}_confusion_matrix.png'
        roc_path = save_dir / f'{model_name}_roc_curve.png'
        pr_path = save_dir / f'{model_name}_pr_curve.png'
    else:
        cm_path = None
        roc_path = None
        pr_path = None
    
    figures = {}
    
    # Plot confusion matrix
    if plot_cm:
        fig_cm = plot_confusion_matrix(y_true, y_pred, model_name, cm_path)
        figures['confusion_matrix'] = fig_cm
    
    # Plot ROC curve
    if plot_roc and y_pred_proba is not None:
        fig_roc = plot_roc_curve(y_true, y_pred_proba, model_name, roc_path)
        figures['roc_curve'] = fig_roc
    
    # Plot Precision-Recall curve
    if plot_pr and y_pred_proba is not None:
        fig_pr = plot_precision_recall_curve(y_true, y_pred_proba, model_name, pr_path)
        figures['precision_recall_curve'] = fig_pr
    
    return metrics, figures


def compare_models_roc(y_true_dict, y_pred_proba_dict, save_path=None, figsize=(10, 8)):
    """
    Plot ROC curves for multiple models on the same plot.
    
    Parameters:
    -----------
    y_true_dict : dict
        Dictionary mapping model names to true labels
    y_pred_proba_dict : dict
        Dictionary mapping model names to predicted probabilities
    save_path : str or Path, optional
        Path to save the figure
    figsize : tuple, default=(10, 8)
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name in y_pred_proba_dict.keys():
        y_true = y_true_dict[model_name]
        y_pred_proba = y_pred_proba_dict[model_name]
        plot_roc_curve(y_true, y_pred_proba, model_name, ax=ax)
    
    ax.set_title('ROC Curves Comparison')
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves comparison saved to {save_path}")
    
    return fig


def export_metrics_to_csv(metrics_list, save_path):
    """
    Export metrics from multiple models to CSV file.
    
    Parameters:
    -----------
    metrics_list : list of dict
        List of metrics dictionaries (from get_metrics_dict or evaluate_model)
    save_path : str or Path
        Path to save CSV file
    """
    df = pd.DataFrame(metrics_list)
    
    # Reorder columns to have 'model' first
    cols = ['model'] + [col for col in df.columns if col != 'model']
    df = df[cols]
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Metrics exported to {save_path}")


def plot_metrics_comparison(metrics_list, save_path=None, figsize=(12, 6)):
    """
    Plot bar chart comparing metrics across models.
    
    Parameters:
    -----------
    metrics_list : list of dict
        List of metrics dictionaries
    save_path : str or Path, optional
        Path to save the figure
    figsize : tuple, default=(12, 6)
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    df = pd.DataFrame(metrics_list)
    
    # Select metrics to plot (exclude model name and auc if not available)
    metric_cols = ['accuracy', 'precision', 'recall', 'f1']
    if 'auc' in df.columns and df['auc'].notna().any():
        metric_cols.append('auc')
    
    # Prepare data for plotting
    plot_data = df[['model'] + metric_cols].set_index('model')
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    plot_data.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Metrics Comparison Across Models')
    ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison chart saved to {save_path}")
    
    return fig
