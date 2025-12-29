"""
Data Preprocessing Module for CardioGuardian
Handles data loading, cleaning, feature scaling, and train-test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os


def load_data(file_path='heart.csv'):
    """
    Load the heart disease dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    print("\nChecking for missing values...")
    missing_count = df.isnull().sum()
    
    if missing_count.sum() > 0:
        print("Missing values found:")
        print(missing_count[missing_count > 0])
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        print("Missing values handled.")
    else:
        print("No missing values found.")
    
    return df


def preprocess_data(df, target_column='target', test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline: handle missing values, split features/target,
    scale features, and split into train/test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        test_size (float): Proportion of test set (default: 0.2)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Handle missing values
    df = handle_missing_values(df.copy())
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Feature scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("Feature scaling completed using StandardScaler.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_scaler(scaler, file_path='models/scaler.pkl'):
    """
    Save the fitted scaler to disk.
    
    Args:
        scaler: Fitted scaler object
        file_path (str): Path to save the scaler
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {file_path}")


def load_scaler(file_path='models/scaler.pkl'):
    """
    Load a saved scaler from disk.
    
    Args:
        file_path (str): Path to the saved scaler
        
    Returns:
        Loaded scaler object
    """
    with open(file_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {file_path}")
    return scaler


if __name__ == "__main__":
    # Example usage
    print("=" * 50)
    print("CardioGuardian - Data Preprocessing")
    print("=" * 50)
    
    # Load data
    df = load_data('heart.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Save scaler
    save_scaler(scaler)
    
    print("\nPreprocessing completed successfully!")

