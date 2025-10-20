"""
Isolation Forest implementation for anomaly detection
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional
import joblib
from loguru import logger


class IsolationForestDetector:
    """Isolation Forest anomaly detector"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Isolation Forest
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        
        self.model = IsolationForest(
            n_estimators=self.config.get('n_estimators', 100),
            max_samples=self.config.get('max_samples', 'auto'),
            contamination=self.config.get('contamination', 0.05),
            max_features=self.config.get('max_features', 1.0),
            bootstrap=self.config.get('bootstrap', False),
            n_jobs=self.config.get('n_jobs', -1),
            random_state=self.config.get('random_state', 42),
            verbose=0
        )
        
        logger.info(f"Initialized Isolation Forest with config: {self.config}")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the model
        
        Args:
            X: Training data
            y: Not used, kept for API consistency
        """
        logger.info(f"Training Isolation Forest on {X.shape[0]} samples")
        self.model.fit(X)
        logger.info("Training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies
        
        Args:
            X: Data to predict
            
        Returns:
            Binary predictions (1 for anomaly, 0 for normal)
        """
        predictions = self.model.predict(X)
        # Convert -1 (anomaly) to 1, and 1 (normal) to 0
        return (predictions == -1).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores
        
        Args:
            X: Data to predict
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        # Get decision function scores (negative = more anomalous)
        scores = self.model.decision_function(X)
        # Convert to 0-1 range where higher is more anomalous
        scores_normalized = (scores.max() - scores) / (scores.max() - scores.min())
        return scores_normalized
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get raw anomaly scores"""
        return -self.model.score_samples(X)
    
    def save(self, path: str):
        """Save model"""
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return self