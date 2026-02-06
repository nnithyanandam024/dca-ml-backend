"""
Train ML models for debt collection prediction
"""

import sklearn
print(sklearn.__version__)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
import xgboost as xgb
import joblib
from models.feature_engineering import FeatureEngineer

def train_recovery_probability_model(df, feature_engineer):
    """
    Train Gradient Boosting model to predict recovery probability
    Target: recovery_probability (0-1)
    """
    print("\n" + "="*60)
    print("TRAINING RECOVERY PROBABILITY MODEL")
    print("="*60)
    
    # Prepare features
    df_encoded = feature_engineer.encode_categorical(df, fit=True)
    df_features = feature_engineer.create_features(df_encoded)
    
    features = feature_engineer.get_model_features(df_features, 'recovery')
    X = df_features[features]
    y = df_features['recovery_probability']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_scaled = feature_engineer.scale_features(X_train, fit=True)
    X_test_scaled = feature_engineer.scale_features(X_test, fit=False)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\n Model Performance:")
    print(f"   Train R² Score: {train_r2:.4f}")
    print(f"   Test R² Score:  {test_r2:.4f}")
    print(f"   Train RMSE:     {train_rmse:.4f}")
    print(f"   Test RMSE:      {test_rmse:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"   CV R² Score:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s} {row['importance']:.4f}")
    
    return model, feature_importance

def train_priority_classification_model(df, feature_engineer):
    """
    Train Random Forest to classify priority levels
    Target: priority (critical, high, medium, low)
    """
    print("\n" + "="*60)
    print("TRAINING PRIORITY CLASSIFICATION MODEL")
    print("="*60)
    
    # Prepare features
    df_encoded = feature_engineer.encode_categorical(df, fit=False)
    df_features = feature_engineer.create_features(df_encoded)
    
    features = feature_engineer.get_model_features(df_features, 'priority')
    X = df_features[features]
    y = df_features['priority']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\n Model Performance:")
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy:  {test_acc:.4f}")
    
    print(f"\n Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s} {row['importance']:.4f}")
    
    return model, feature_importance

def train_recovery_amount_model(df, feature_engineer):
    """
    Train XGBoost to predict expected recovery amount
    Target: expected_recovery (dollar amount)
    """
    print("\n" + "="*60)
    print("TRAINING EXPECTED RECOVERY AMOUNT MODEL")
    print("="*60)
    
    # Prepare features
    df_encoded = feature_engineer.encode_categorical(df, fit=False)
    df_features = feature_engineer.create_features(df_encoded)
    
    features = feature_engineer.get_model_features(df_features, 'amount')
    X = df_features[features]
    y = df_features['expected_recovery']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\nModel Performance:")
    print(f"   Train R² Score: {train_r2:.4f}")
    print(f"   Test R² Score:  {test_r2:.4f}")
    print(f"   Train RMSE:     ${train_rmse:,.2f}")
    print(f"   Test RMSE:      ${test_rmse:,.2f}")
    
    return model

def main():
    """Main training pipeline"""
    print("\nStarting ML Model Training Pipeline")
    print("="*60)
    
    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv('data/fedex_cases.csv')
    print(f"   Loaded {len(df)} cases")
    print(f"   Features: {df.shape[1]} columns")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Train models
    recovery_model, recovery_importance = train_recovery_probability_model(df, feature_engineer)
    priority_model, priority_importance = train_priority_classification_model(df, feature_engineer)
    amount_model = train_recovery_amount_model(df, feature_engineer)
    
    # Save models
    print("\nSaving models...")
    joblib.dump(recovery_model, 'saved_models/recovery_model.pkl')
    joblib.dump(priority_model, 'saved_models/priority_model.pkl')
    joblib.dump(amount_model, 'saved_models/amount_model.pkl')
    joblib.dump(feature_engineer, 'saved_models/feature_engineer.pkl')
    joblib.dump(recovery_importance, 'saved_models/recovery_importance.pkl')
    joblib.dump(priority_importance, 'saved_models/priority_importance.pkl')

    
    print("   ✅ All models saved successfully!")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nModels saved in ../saved_models/:")
    print("  • recovery_model.pkl")
    print("  • priority_model.pkl")
    print("  • amount_model.pkl")
    print("  • feature_engineer.pkl")
    print("\nReady to deploy! ")

if __name__ == '__main__':
    main()