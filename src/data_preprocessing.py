"""
Data preprocessing functions for credit card fraud detection.

This module provides reusable functions for scaling, splitting, and handling
class imbalance in the credit card fraud detection dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def scale_features(X, feature_cols=None, scaler=None, fit=True):
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features to scale
    feature_cols : list, optional
        List of column names to scale. If None, scales all columns.
    scaler : StandardScaler, optional
        Pre-fitted scaler. If None, creates a new one.
    fit : bool, default=True
        Whether to fit the scaler (True) or only transform (False)
    
    Returns:
    --------
    X_scaled : pandas.DataFrame or numpy.ndarray
        Scaled features
    scaler : StandardScaler
        Fitted scaler object
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if isinstance(X, pd.DataFrame):
        if feature_cols is None:
            feature_cols = X.columns.tolist()
        
        X_to_scale = X[feature_cols].copy()
        
        if fit:
            X_scaled_array = scaler.fit_transform(X_to_scale)
        else:
            X_scaled_array = scaler.transform(X_to_scale)
        
        X_scaled = X.copy()
        X_scaled[feature_cols] = X_scaled_array
        
        return X_scaled, scaler
    else:
        if fit:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        return X_scaled, scaler


def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets with stratification.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    test_size : float, default=0.15
        Proportion of data for test set
    val_size : float, default=0.15
        Proportion of data for validation set (from remaining data after test split)
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test : tuple
        Split datasets
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    # Adjust val_size to account for the fact that we're splitting from X_temp
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X, y, random_state=42, k_neighbors=5):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to balance classes.
    
    SMOTE creates synthetic samples for the minority class by interpolating
    between existing minority class samples.
    
    Advantages:
    - Increases minority class samples without exact duplicates
    - Can improve model performance on minority class
    - Works well with continuous features
    
    Disadvantages:
    - Can create noisy samples if minority class is too small
    - May overfit if used improperly
    - Computationally expensive for large datasets
    - Should only be applied to training data, not validation/test
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    random_state : int, default=42
        Random seed for reproducibility
    k_neighbors : int, default=5
        Number of nearest neighbors to use for SMOTE
    
    Returns:
    --------
    X_resampled : numpy.ndarray
        Resampled features
    y_resampled : numpy.ndarray
        Resampled target variable
    """
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    X_resampled, y_resampled = smote.fit_resample(X_array, y_array)
    
    return X_resampled, y_resampled


def get_class_weights(y):
    """
    Calculate class weights for imbalanced datasets.
    
    Class weights can be used with models that support the class_weight parameter
    (e.g., RandomForest, XGBoost, LogisticRegression) to give more importance
    to the minority class during training.
    
    Advantages:
    - No need to modify training data
    - Faster than oversampling
    - Works well with tree-based models
    - Preserves original data distribution
    
    Disadvantages:
    - May not work as well as SMOTE for some models
    - Less effective when imbalance is extreme
    - Requires model support for class_weight parameter
    
    Parameters:
    -----------
    y : pandas.Series or numpy.ndarray
        Target variable
    
    Returns:
    --------
    class_weights : dict
        Dictionary mapping class labels to their weights
    """
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    unique_classes = np.unique(y_array)
    n_samples = len(y_array)
    n_classes = len(unique_classes)
    
    # Calculate weights inversely proportional to class frequency
    class_weights = {}
    for cls in unique_classes:
        n_class = np.sum(y_array == cls)
        weight = n_samples / (n_classes * n_class)
        class_weights[cls] = weight
    
    return class_weights



