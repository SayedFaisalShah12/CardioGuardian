"""
Model Training Module for CardioGuardian
Handles model comparison, training, and saving the best model
"""

import pickle
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import load_data, preprocess_data, save_scaler


def get_models():
    """
    Get a dictionary of models to compare.
    
    Returns:
        dict: Dictionary of model names and their instances
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    return models


def compare_models(X_train, X_test, y_train, y_test):
    """
    Compare multiple models and return their performance scores.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target
        
    Returns:
        dict: Dictionary of model names and their accuracy scores
    """
    models = get_models()
    results = {}
    
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}")
    
    return results


def select_best_model(results):
    """
    Select the best model based on accuracy.
    
    Args:
        results (dict): Dictionary of model results
        
    Returns:
        tuple: (best_model_name, best_model, best_accuracy)
    """
    best_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_name]['model']
    best_accuracy = results[best_name]['accuracy']
    
    print("\n" + "=" * 50)
    print("Best Model Selection")
    print("=" * 50)
    print(f"Best Model: {best_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    return best_name, best_model, best_accuracy


def save_model(model, model_name, file_path=None):
    """
    Save the trained model to disk using pickle.
    
    Args:
        model: Trained model object
        model_name (str): Name of the model
        file_path (str): Path to save the model (optional)
    """
    if file_path is None:
        # Create a safe filename from model name
        safe_name = model_name.lower().replace(' ', '_')
        file_path = f'models/{safe_name}_model.pkl'
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {file_path}")
    return file_path


def load_model(file_path='models/best_model.pkl'):
    """
    Load a saved model from disk.
    
    Args:
        file_path (str): Path to the saved model
        
    Returns:
        Loaded model object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {file_path}")
    return model


def train_and_save_best_model(data_path='heart.csv', model_save_path='models/best_model.pkl'):
    """
    Complete training pipeline: load data, preprocess, compare models,
    select best, and save.
    
    Args:
        data_path (str): Path to the dataset
        model_save_path (str): Path to save the best model
        
    Returns:
        tuple: (best_model_name, best_model, X_test, y_test, scaler)
    """
    print("=" * 50)
    print("CardioGuardian - Model Training")
    print("=" * 50)
    
    # Load and preprocess data
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Save scaler
    save_scaler(scaler)
    
    # Compare models
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # Select best model
    best_name, best_model, best_accuracy = select_best_model(results)
    
    # Save best model
    save_model(best_model, best_name, model_save_path)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)
    
    return best_name, best_model, X_test, y_test, scaler


if __name__ == "__main__":
    # Train and save the best model
    train_and_save_best_model()

