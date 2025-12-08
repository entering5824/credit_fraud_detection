"""
Model training functions for credit card fraud detection.

This module provides functions to train ensemble learning models:
- Random Forest
- AdaBoost
- XGBoost
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import classification_report


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=15, 
                        class_weight=None, random_state=42, n_jobs=-1, verbose=1):
    """
    Train Random Forest classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    n_estimators : int, default=200
        Number of trees in the forest
    max_depth : int, default=15
        Maximum depth of trees
    class_weight : dict or 'balanced', optional
        Class weights for imbalanced data
    random_state : int, default=42
        Random seed
    n_jobs : int, default=-1
        Number of parallel jobs
    verbose : int, default=1
        Verbosity level
    
    Returns:
    --------
    model : RandomForestClassifier
        Trained Random Forest model
    """
    if verbose >= 1:
        print("🔹 Training Random Forest model...")
        print(f"   Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    model.fit(X_train, y_train)
    
    if verbose >= 1:
        print("✅ Random Forest training completed!")
    
    return model


def train_adaboost(X_train, y_train, n_estimators=50, learning_rate=1.0,
                   base_estimator=None, random_state=42, verbose=1):
    """
    Train AdaBoost classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    n_estimators : int, default=50
        Number of weak learners
    learning_rate : float, default=1.0
        Learning rate
    base_estimator : object, optional
        Base estimator (default: DecisionTreeClassifier with max_depth=1)
    random_state : int, default=42
        Random seed
    verbose : int, default=1
        Verbosity level
    
    Returns:
    --------
    model : AdaBoostClassifier
        Trained AdaBoost model
    """
    if verbose >= 1:
        print("🔹 Training AdaBoost model...")
        print(f"   Parameters: n_estimators={n_estimators}, learning_rate={learning_rate}")
    
    if base_estimator is None:
        base_estimator = DecisionTreeClassifier(max_depth=1, random_state=random_state)
    
    model = AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    if verbose >= 1:
        print("✅ AdaBoost training completed!")
    
    return model


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, 
                  learning_rate=0.1, scale_pos_weight=None, random_state=42, 
                  verbose=1, use_label_encoder=False):
    """
    Train XGBoost classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    n_estimators : int, default=100
        Number of boosting rounds
    max_depth : int, default=6
        Maximum tree depth
    learning_rate : float, default=0.1
        Learning rate
    scale_pos_weight : float, optional
        Weight for positive class (for imbalanced data)
    random_state : int, default=42
        Random seed
    verbose : int, default=1
        Verbosity level
    use_label_encoder : bool, default=False
        Whether to use label encoder (deprecated in newer versions)
    
    Returns:
    --------
    model : XGBClassifier
        Trained XGBoost model
    """
    if verbose >= 1:
        print("🔹 Training XGBoost model...")
        print(f"   Parameters: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=use_label_encoder,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    if verbose >= 1:
        print("✅ XGBoost training completed!")
    
    return model


def save_model(model, model_name, save_dir='models'):
    """
    Save trained model to pickle file.
    
    Parameters:
    -----------
    model : object
        Trained model to save
    model_name : str
        Name of the model (will be used as filename)
    save_dir : str or Path, default='models'
        Directory to save the model
    
    Returns:
    --------
    save_path : Path
        Path where model was saved
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean model name for filename
    filename = model_name.lower().replace(' ', '_') + '.pkl'
    save_path = save_dir / filename
    
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to {save_path}")
    return save_path


def load_model(model_name, model_dir='models'):
    """
    Load trained model from pickle file.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (filename without .pkl)
    model_dir : str or Path, default='models'
        Directory containing the model
    
    Returns:
    --------
    model : object
        Loaded model
    """
    model_dir = Path(model_dir)
    filename = model_name.lower().replace(' ', '_') + '.pkl'
    model_path = model_dir / filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Model loaded from {model_path}")
    return model


def evaluate_model_performance(model, X_test, y_test, model_name='Model'):
    """
    Evaluate model and print classification report.
    
    Parameters:
    -----------
    model : object
        Trained model with predict() method
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    model_name : str, default='Model'
        Name of the model for display
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {model_name}")
    print(f"{'='*60}")
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return y_pred

