"""
Project Sentinel: Sepsis-Associated AKI Early Warning System
Evaluation module.
"""

import os
from typing import Dict, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate the Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True)
    
    ece = 0.0
    total_samples = len(y_prob)
    
    for i in range(1, n_bins + 1):
        bin_mask = (bin_indices == i)
        if not np.any(bin_mask):
            continue
            
        bin_probs = y_prob[bin_mask]
        bin_true = y_true[bin_mask]
        
        bin_accuracy = np.mean(bin_true)
        bin_confidence = np.mean(bin_probs)
        bin_weight = len(bin_probs) / total_samples
        
        ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
    return ece

def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate overall model discrimination and calibration metrics."""
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece
    }

def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate classification metrics at a specific probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0
    
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1
    }

def plot_calibration_curves(results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Optional[str] = None) -> None:
    """Plot calibration curves for multiple models."""
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for model_name, (y_true, y_prob) in results_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="s", label=model_name)
        
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curves")
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()

def plot_roc_curves(results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Optional[str] = None) -> None:
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(8, 8))
    
    for model_name, (y_true, y_prob) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()

def plot_pr_curves(results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Optional[str] = None) -> None:
    """Plot Precision-Recall curves for multiple models."""
    plt.figure(figsize=(8, 8))
    
    for model_name, (y_true, y_prob) in results_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{model_name} (AUPRC = {auprc:.3f})")
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()

def decision_curve_analysis(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Perform Decision Curve Analysis to calculate Net Benefit."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
        
    net_benefits = []
    n_samples = len(y_true)
    prevalence = np.mean(y_true)
    
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        net_benefit = (tp / n_samples) - (fp / n_samples) * (pt / (1 - pt))
        net_benefits.append(net_benefit)
        
    treat_all_nb = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    treat_none_nb = np.zeros_like(thresholds)
    
    return pd.DataFrame({
        "threshold": thresholds,
        "model_net_benefit": net_benefits,
        "treat_all_net_benefit": treat_all_nb,
        "treat_none_net_benefit": treat_none_nb
    })

def plot_decision_curves(dca_results: Dict[str, pd.DataFrame], output_path: Optional[str] = None) -> None:
    """Plot Decision Curve Analysis results."""
    plt.figure(figsize=(10, 8))
    
    first_key = list(dca_results.keys())[0]
    first_df = dca_results[first_key]
    
    plt.plot(first_df["threshold"], first_df["treat_all_net_benefit"], 'k--', label="Treat All")
    plt.plot(first_df["threshold"], first_df["treat_none_net_benefit"], 'k:', label="Treat None")
    
    for model_name, dca_df in dca_results.items():
        plt.plot(dca_df["threshold"], dca_df["model_net_benefit"], label=model_name)
        
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.ylim([-0.05, max(first_df["treat_all_net_benefit"].max(), 0.1) * 1.2])
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()

def subgroup_analysis(y_true: np.ndarray, y_prob: np.ndarray, subgroups: pd.Series, output_path: Optional[str] = None) -> pd.DataFrame:
    """Evaluate model performance across different subgroups."""
    unique_groups = subgroups.unique()
    results = []
    
    for group in unique_groups:
        mask = (subgroups == group)
        if sum(mask) > 0 and sum(y_true[mask]) > 0 and sum(~y_true[mask].astype(bool)) > 0:
            group_metrics = evaluate_model(y_true[mask], y_prob[mask])
            group_metrics["subgroup"] = group
            group_metrics["n_samples"] = sum(mask)
            group_metrics["n_events"] = sum(y_true[mask])
            results.append(group_metrics)
            
    df_results = pd.DataFrame(results)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_results.to_csv(output_path, index=False)
        
    return df_results

def generate_comparison_table(results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Optional[str] = None) -> pd.DataFrame:
    """Generate a table comparing metrics across multiple models."""
    comparison = []
    
    for model_name, (y_true, y_prob) in results_dict.items():
        metrics = evaluate_model(y_true, y_prob)
        thresh_metrics = threshold_metrics(y_true, y_prob, threshold=0.5)
        
        combined = {"Model": model_name, **metrics, **thresh_metrics}
        comparison.append(combined)
        
    df_comparison = pd.DataFrame(comparison)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_comparison.to_csv(output_path, index=False)
        
    return df_comparison

def run_full_evaluation(models_dict: Dict[str, Any], X_test: Union[np.ndarray, pd.DataFrame], y_test: np.ndarray, subgroups: Optional[pd.Series] = None, output_dir: str = "outputs") -> Dict[str, Any]:
    """Run the complete evaluation suite for provided models."""
    figures_dir = os.path.join(output_dir, "figures")
    reports_dir = os.path.join(output_dir, "reports")
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    results_dict = {}
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
        results_dict[name] = (y_test, y_prob)
        
    # Generate Plots
    plot_roc_curves(results_dict, output_path=os.path.join(figures_dir, "roc_curves.png"))
    plot_pr_curves(results_dict, output_path=os.path.join(figures_dir, "pr_curves.png"))
    plot_calibration_curves(results_dict, output_path=os.path.join(figures_dir, "calibration_curves.png"))
    
    # DCA
    dca_results = {name: decision_curve_analysis(y_true, y_prob) for name, (y_true, y_prob) in results_dict.items()}
    plot_decision_curves(dca_results, output_path=os.path.join(figures_dir, "decision_curves.png"))
    
    # Comparison Table
    comp_df = generate_comparison_table(results_dict, output_path=os.path.join(reports_dir, "model_comparison.csv"))
    
    # Subgroup Analysis
    subgroup_results = {}
    if subgroups is not None:
        for name, (y_true, y_prob) in results_dict.items():
            out_path = os.path.join(reports_dir, f"{name}_subgroup_analysis.csv")
            sub_df = subgroup_analysis(y_true, y_prob, subgroups, output_path=out_path)
            subgroup_results[name] = sub_df
            
    return {
        "results": results_dict,
        "dca": dca_results,
        "comparison": comp_df,
        "subgroups": subgroup_results
    }
