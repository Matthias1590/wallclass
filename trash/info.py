import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def load_models_and_data():
    """Load multi-class models and prepare test data."""
    # Load data
    df = pd.read_csv("features_all.csv", index_col=0)
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns].fillna(0)
    y = df['label']
    
    # Encode labels (same as training)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Load scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load label encoder from saved models
    with open('models/label_encoder.pkl', 'rb') as f:
        saved_label_encoder = pickle.load(f)
    
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)
    
    # Load models
    models = {}
    model_files = ['xgb_multiclass.pkl', 'rf_multiclass.pkl']
    
    for model_file in model_files:
        model_path = f'models/{model_file}'
        if os.path.exists(model_path):
            model_name = model_file.replace('.pkl', '')
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
    
    return models, X_test_scaled, y_test, saved_label_encoder

def evaluate_multiclass_model(model, X_test, y_test, label_encoder):
    """Evaluate a multi-class model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class metrics
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'classification_report': classification_report(y_test, y_pred, target_names=label_encoder.classes_),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics

def main():
    print("="*60)
    print("MULTI-CLASS MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    # Load models and data
    models, X_test_scaled, y_test, label_encoder = load_models_and_data()
    
    # Get original labels for display
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    print(f"\nTest set size: {len(y_test)} samples")
    print(f"Class distribution:")
    unique, counts = np.unique(y_test_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {cls}: {count} samples ({count/len(y_test_labels)*100:.1f}%)")
    
    print("\n" + "-"*60)
    
    # Evaluate each model
    summary_data = []
    
    for model_name, model in models.items():
        print(f"\n{model_name.upper().replace('_', ' ')} RESULTS:")
        print("-" * 40)
        
        metrics = evaluate_multiclass_model(model, X_test_scaled, y_test, label_encoder)
        
        print(f"Overall Accuracy: {metrics['accuracy']:.3f}")
        print(f"\nMacro-averaged metrics:")
        print(f"  Precision: {metrics['precision_macro']:.3f}")
        print(f"  Recall:    {metrics['recall_macro']:.3f}")
        print(f"  F1-Score:  {metrics['f1_macro']:.3f}")
        
        print(f"\nWeighted-averaged metrics:")
        print(f"  Precision: {metrics['precision_weighted']:.3f}")
        print(f"  Recall:    {metrics['recall_weighted']:.3f}")
        print(f"  F1-Score:  {metrics['f1_weighted']:.3f}")
        
        print(f"\nDetailed Classification Report:")
        print(metrics['classification_report'])
        
        print(f"\nConfusion Matrix:")
        print(f"Classes: {label_encoder.classes_}")
        print(metrics['confusion_matrix'])
        
        # Add to summary
        summary_data.append({
            'Model': model_name.upper().replace('_', ' '),
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision (Macro)': f"{metrics['precision_macro']:.3f}",
            'Recall (Macro)': f"{metrics['recall_macro']:.3f}",
            'F1 (Macro)': f"{metrics['f1_macro']:.3f}",
            'Precision (Weighted)': f"{metrics['precision_weighted']:.3f}",
            'Recall (Weighted)': f"{metrics['recall_weighted']:.3f}",
            'F1 (Weighted)': f"{metrics['f1_weighted']:.3f}"
        })
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("Models loaded from: models/ directory")
    print(f"Total models evaluated: {len(models)}")
    print("Multi-class classification approach used")
    print("="*60)

if __name__ == "__main__":
    main()
