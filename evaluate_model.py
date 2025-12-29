"""
Model Evaluation Module for CardioGuardian
Provides comprehensive evaluation metrics including accuracy, precision, recall,
F1-score, and confusion matrix
"""

import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from train_model import load_model
from data_preprocessing import load_scaler


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model with comprehensive metrics.
    
    Args:
        model: Trained model object
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC-AUC if probabilities are available
    roc_auc = None
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            pass
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics


def print_evaluation_report(metrics):
    """
    Print a formatted evaluation report.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 50)
    print("Model Evaluation Report")
    print("=" * 50)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\n" + "-" * 50)
    print("Confusion Matrix:")
    print("-" * 50)
    print(metrics['confusion_matrix'])
    
    print("\n" + "-" * 50)
    print("Classification Report:")
    print("-" * 50)
    # Create a simple classification report
    cm = metrics['confusion_matrix']
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"True Negatives:  {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives:  {tp}")


def plot_confusion_matrix(cm, save_path='results/confusion_matrix.png'):
    """
    Plot and save the confusion matrix.
    
    Args:
        cm (np.array): Confusion matrix
        save_path (str): Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Heart Disease', 'Heart Disease'],
                yticklabels=['No Heart Disease', 'Heart Disease'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix plot saved to {save_path}")
    plt.close()


def plot_roc_curve(y_test, y_pred_proba, save_path='results/roc_curve.png'):
    """
    Plot and save the ROC curve.
    
    Args:
        y_test (pd.Series): True labels
        y_pred_proba (np.array): Prediction probabilities
        save_path (str): Path to save the plot
    """
    if y_pred_proba is None:
        print("ROC curve not available (model doesn't support probabilities)")
        return
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"ROC curve plot saved to {save_path}")
    plt.close()


def evaluate_saved_model(model_path='models/best_model.pkl', 
                        scaler_path='models/scaler.pkl',
                        data_path='heart.csv'):
    """
    Load and evaluate a saved model.
    
    Args:
        model_path (str): Path to the saved model
        scaler_path (str): Path to the saved scaler
        data_path (str): Path to the dataset
        
    Returns:
        dict: Evaluation metrics
    """
    from data_preprocessing import load_data, preprocess_data
    
    print("=" * 50)
    print("CardioGuardian - Model Evaluation")
    print("=" * 50)
    
    # Load model and scaler
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    
    # Load and preprocess data
    df = load_data(data_path)
    _, X_test, _, y_test, _ = preprocess_data(df)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print report
    print_evaluation_report(metrics)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # Plot ROC curve if available
    if metrics['y_pred_proba'] is not None:
        plot_roc_curve(y_test, metrics['y_pred_proba'])
    
    return metrics


if __name__ == "__main__":
    # Evaluate the saved model
    evaluate_saved_model()

