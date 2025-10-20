"""
Ensemble model combining multiple anomaly detectors
"""
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
import joblib


class EnsembleDetector:
    """Ensemble of multiple anomaly detection models"""
    
    def __init__(self, models: Optional[List] = None, weights: Optional[List[float]] = None):
        """
        Initialize ensemble
        
        Args:
            models: List of trained models
            weights: Weights for each model's predictions
        """
        self.models = models or []
        self.weights = weights or [1.0] * len(self.models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"Initialized ensemble with {len(self.models)} models")
    
    def add_model(self, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.weights.append(weight)
        
        # Renormalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"Added model to ensemble. Total models: {len(self.models)}")
    
    def predict(self, X: np.ndarray, method: str = 'voting') -> np.ndarray:
        """
        Predict anomalies using ensemble
        
        Args:
            X: Data to predict
            method: 'voting' or 'weighted_average'
            
        Returns:
            Binary predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        if method == 'voting':
            # Weighted majority voting
            predictions = np.zeros(len(X))
            for model, weight in zip(self.models, self.weights):
                predictions += model.predict(X) * weight
            
            return (predictions >= 0.5).astype(int)
        
        elif method == 'weighted_average':
            # Weighted average of anomaly scores
            scores = self.predict_proba(X)
            # Use dynamic threshold
            threshold = np.percentile(scores, 95)
            return (scores > threshold).astype(int)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get weighted average of anomaly scores
        
        Args:
            X: Data to predict
            
        Returns:
            Anomaly probability scores
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        scores = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            model_scores = model.predict_proba(X)
            scores += model_scores * weight
        
        return scores
    
    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get individual predictions from each model"""
        
        predictions = {}
        for i, model in enumerate(self.models):
            model_name = f"{model.__class__.__name__}_{i}"
            predictions[model_name] = {
                'binary': model.predict(X),
                'scores': model.predict_proba(X)
            }
        
        return predictions
    
    def evaluate_agreement(self, X: np.ndarray) -> Dict[str, float]:
        """Evaluate agreement between models"""
        
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        
        predictions = np.array(predictions)
        
        agreement = {
            'full_agreement': np.mean(
                np.all(predictions == predictions[0], axis=0)
            ),
            'majority_agreement': np.mean(
                np.sum(predictions, axis=0) >= len(self.models) / 2
            ),
            'average_pairwise_agreement': 0.0
        }
        
        if len(self.models) > 1:
            pairwise_agreements = []
            for i in range(len(self.models)):
                for j in range(i + 1, len(self.models)):
                    agreement_score = np.mean(
                        predictions[i] == predictions[j]
                    )
                    pairwise_agreements.append(agreement_score)
            
            agreement['average_pairwise_agreement'] = np.mean(pairwise_agreements)
        
        return agreement
    
    def save(self, path: str):
        """Save ensemble"""
        ensemble_data = {
            'models': self.models,
            'weights': self.weights
        }
        joblib.dump(ensemble_data, path)
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: str):
        """Load ensemble"""
        ensemble_data = joblib.load(path)
        self.models = ensemble_data['models']
        self.weights = ensemble_data['weights']
        logger.info(f"Ensemble loaded from {path}")
        return self