import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import pickle
import os

def load_and_prepare_data():
    """Load features and prepare data for training."""
    df = pd.read_csv("features_all.csv", index_col=0)
    
    # Separate features and labels
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]
    y = df['label']
    
    # Handle any missing values
    X = X.fillna(0)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts()}")
    
    return X, y, feature_columns

def train_multiclass_models(X_train, X_test, y_train, y_test, label_encoder):
    """Train XGBoost and Random Forest models for multi-class classification."""
    print("\nTraining multi-class models...")
    
    # Get class distribution for weighting
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    
    # More aggressive rebalancing for imbalanced data
    # Use sqrt of inverse frequency to not over-correct
    class_weights_dict = {}
    for i, count in enumerate(class_counts):
        weight = np.sqrt(total_samples / (len(class_counts) * count))
        class_weights_dict[i] = weight
    
    print(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
    print(f"Class weights: {class_weights_dict}")
    
    # Create sample weights for balanced training
    sample_weights = np.array([class_weights_dict[label] for label in y_train])
    
    models = {}
    
    # XGBoost Multi-class with focal loss-like approach
    print("Training XGBoost (multi-class)...")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=300,  # More trees for complex patterns
        max_depth=8,       # Deeper trees
        learning_rate=0.05, # Lower learning rate for stability  
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,     # L1 regularization
        reg_lambda=0.1,    # L2 regularization
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Random Forest Multi-class with balanced subsample
    print("Training Random Forest (multi-class)...")
    rf_model = RandomForestClassifier(
        n_estimators=300,  # More trees
        max_depth=15,      # Deeper trees
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced_subsample',  # More aggressive than 'balanced'
        random_state=42,
        n_jobs=-1  # Use all cores
    )
    rf_model.fit(X_train, y_train)
    
    models['xgb_multiclass'] = xgb_model
    models['rf_multiclass'] = rf_model
    
    # Evaluate models
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    for model_name, model in models.items():
        print(f"\n{model_name.upper()} Results:")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    return models

def main():
    # Load data
    X, y, feature_columns = load_and_prepare_data()
    
    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to maintain feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)
    
    # Train multi-class models
    all_models = train_multiclass_models(X_train_scaled, X_test_scaled, y_train, y_test, label_encoder)
    
    # Save models and preprocessing objects
    os.makedirs('models', exist_ok=True)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature columns
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    # Save label encoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save models
    for model_name, model in all_models.items():
        with open(f'models/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    print(f"\nAll models saved to 'models/' directory")
    print(f"Models trained: {list(all_models.keys())}")
    
    # Feature importance analysis
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Use Random Forest for feature importance (more interpretable)
    rf_model = all_models['rf_multiclass']
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 most important features:")
    print(feature_importance.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
