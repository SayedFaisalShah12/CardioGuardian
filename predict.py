"""
Prediction Module for CardioGuardian
Handles making predictions using the saved model
"""

import pickle
import os
import pandas as pd
import numpy as np
from train_model import load_model
from data_preprocessing import load_scaler


def predict_single(model, scaler, features):
    """
    Make a prediction for a single sample.
    
    Args:
        model: Trained model object
        scaler: Fitted scaler object
        features (dict or pd.DataFrame): Feature values for prediction
        
    Returns:
        dict: Prediction result with class and probability
    """
    # Convert dict to DataFrame if needed
    if isinstance(features, dict):
        # Ensure correct order of features
        feature_order = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        features_df = pd.DataFrame([features], columns=feature_order)
    else:
        features_df = features.copy()
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    features_scaled = pd.DataFrame(features_scaled, columns=features_df.columns)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    # Get prediction probability if available
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(features_scaled)[0]
        probability = {
            'no_disease': float(probability[0]),
            'disease': float(probability[1])
        }
    
    result = {
        'prediction': int(prediction),
        'prediction_label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
        'probability': probability
    }
    
    return result


def predict_batch(model, scaler, features_df):
    """
    Make predictions for multiple samples.
    
    Args:
        model: Trained model object
        scaler: Fitted scaler object
        features_df (pd.DataFrame): DataFrame with feature values
        
    Returns:
        pd.Series: Series of predictions
    """
    # Scale features
    features_scaled = scaler.transform(features_df)
    features_scaled = pd.DataFrame(features_scaled, columns=features_df.columns)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    
    return pd.Series(predictions, name='prediction')


def load_model_and_scaler(model_path='models/best_model.pkl', 
                          scaler_path='models/scaler.pkl'):
    """
    Load both model and scaler from disk.
    
    Args:
        model_path (str): Path to the saved model
        scaler_path (str): Path to the saved scaler
        
    Returns:
        tuple: (model, scaler)
    """
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    return model, scaler


def predict_from_dict(features_dict, 
                     model_path='models/best_model.pkl',
                     scaler_path='models/scaler.pkl'):
    """
    Convenience function to make a prediction from a dictionary of features.
    
    Args:
        features_dict (dict): Dictionary with feature names and values
        model_path (str): Path to the saved model
        scaler_path (str): Path to the saved scaler
        
    Returns:
        dict: Prediction result
    """
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    return predict_single(model, scaler, features_dict)


if __name__ == "__main__":
    # Example usage
    print("=" * 50)
    print("CardioGuardian - Prediction Example")
    print("=" * 50)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Example features (from the dataset)
    example_features = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }
    
    # Make prediction
    result = predict_single(model, scaler, example_features)
    
    print("\nPrediction Result:")
    print(f"Prediction: {result['prediction_label']}")
    if result['probability']:
        print(f"Probability (No Disease): {result['probability']['no_disease']:.4f}")
        print(f"Probability (Disease): {result['probability']['disease']:.4f}")

