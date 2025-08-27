import pandas as pd
import numpy as np
import pickle
import os

class WalletClassifier:
    """Wallet classifier using a single multi-class model."""
    
    def __init__(self, models_dir='models', model_type='rf'):
        """Initialize the classifier by loading model and scaler."""
        self.models_dir = models_dir
        self.model_type = model_type  # 'rf' for Random Forest, 'xgb' for XGBoost
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.label_encoder = None
        
        self._load_models()
    
    def _load_models(self):
        """Load the trained multi-class model and preprocessing objects."""
        try:
            # Load scaler
            with open(f'{self.models_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature columns
            with open(f'{self.models_dir}/feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            # Load label encoder
            with open(f'{self.models_dir}/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load the multi-class model
            model_file = f'{self.models_dir}/{self.model_type}_multiclass.pkl'
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Loaded {self.model_type.upper()} multi-class model")
            else:
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

    def predict_probabilities(self, features):
        """
        Predict probabilities for each wallet class using multi-class model.
        
        Args:
            features: pandas DataFrame with wallet features, or dict/numpy array
        
        Returns:
            pandas DataFrame with columns: bot_probability, customer_probability, exchange_probability
        """
        # Check if models are loaded
        if self.scaler is None or self.feature_columns is None or self.model is None:
            raise RuntimeError("Models not loaded. Check if model files exist in the models directory.")
        
        # Convert input to DataFrame if necessary
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif isinstance(features, np.ndarray):
            features = pd.DataFrame(features, columns=self.feature_columns)
        elif not isinstance(features, pd.DataFrame):
            raise ValueError("Features must be a DataFrame, dict, or numpy array")
        
        # Ensure we have all required features in correct order
        if not all(col in features.columns for col in self.feature_columns):
            missing_cols = [col for col in self.feature_columns if col not in features.columns]
            raise ValueError(f"Missing required feature columns: {missing_cols}")
        
        # Select and order features correctly, fill missing values
        features_ordered = features[self.feature_columns].fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features_ordered)
        
        # Predict probabilities for all classes
        probabilities = self.model.predict_proba(features_scaled)
        
        # Create results DataFrame with class names
        class_names = self.label_encoder.classes_
        results = pd.DataFrame(
            probabilities, 
            columns=[f'{cls}_probability' for cls in class_names],
            index=features.index
        )
        
        return results
    
    def predict_class(self, features, threshold=0.5):
        """
        Predict the most likely class for each wallet.
        
        Args:
            features: pandas DataFrame with wallet features
            threshold: minimum probability threshold for positive classification
        
        Returns:
            pandas DataFrame with predicted_class and max_probability columns
        """
        probabilities = self.predict_probabilities(features)
        
        # Find class with highest probability
        class_names = self.label_encoder.classes_
        max_prob_idx = probabilities.values.argmax(axis=1)
        
        results = pd.DataFrame(index=features.index)
        results['predicted_class'] = [class_names[i] for i in max_prob_idx]
        results['max_probability'] = probabilities.values.max(axis=1)
        
        # Set to 'unknown' if max probability is below threshold
        results.loc[results['max_probability'] < threshold, 'predicted_class'] = 'unknown'
        
        return results

# Global classifier instance
_classifier = None

def get_classifier():
    """Get or create the global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = WalletClassifier()
    return _classifier

def predict_wallet_probabilities(features):
    """
    Convenience function to predict wallet class probabilities.
    
    Args:
        features: pandas DataFrame, dict, or numpy array with wallet features
    
    Returns:
        pandas DataFrame with bot_probability, customer_probability, exchange_probability columns
    
    Example:
        >>> features = {'tx_count_total': 100, 'txs_per_day': 5.2, ...}
        >>> probabilities = predict_wallet_probabilities(features)
        >>> print(probabilities)
               bot_probability  customer_probability  exchange_probability
        0              0.123                0.567                0.234
    """
    classifier = get_classifier()
    return classifier.predict_probabilities(features)

def predict_wallet_class(features, threshold=0.5):
    """
    Convenience function to predict the most likely wallet class.
    
    Args:
        features: pandas DataFrame, dict, or numpy array with wallet features
        threshold: minimum probability threshold for positive classification
    
    Returns:
        pandas DataFrame with predicted_class and max_probability columns
    """
    classifier = get_classifier()
    return classifier.predict_class(features, threshold)

if __name__ == "__main__":
    # Example usage
    print("Loading wallet classifier...")
    
    # Load a sample from the training data to test
    df = pd.read_csv("features_all.csv", index_col=0)
    feature_columns = [col for col in df.columns if col != 'label']
    
    # Take random samples instead of first few
    sample_size = 5
    sample_indices = np.random.choice(df.index, size=min(sample_size, len(df)), replace=False)
    sample_features = df.loc[sample_indices, feature_columns]
    sample_labels = df.loc[sample_indices, 'label']
    
    print(f"Testing with {len(sample_features)} random sample wallets...")
    print(f"Sample indices: {list(sample_indices)}")
    
    # Test probability prediction
    probabilities = predict_wallet_probabilities(sample_features)
    print("\nPredicted probabilities:")
    print(probabilities.round(3))
    
    # Test class prediction
    predictions = predict_wallet_class(sample_features, threshold=0.4)
    print("\nPredicted classes:")
    print(predictions.round(3))
    
    # Compare with actual labels
    comparison = pd.DataFrame({
        'actual_label': sample_labels,
        'predicted_class': predictions['predicted_class'],
        'max_probability': predictions['max_probability'].round(3)
    })
    print("\nComparison with actual labels:")
    print(comparison)
