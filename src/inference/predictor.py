"""
Real-time and batch prediction service
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from loguru import logger
import time

from src.data.preprocessor import TransactionPreprocessor
from src.models.ensemble import EnsembleDetector


class AnomalyPredictor:
    """
    Production-ready predictor for anomaly detection
    Handles single and batch predictions with caching and monitoring
    """
    
    def __init__(
        self,
        model_path: str = "data/models",
        enable_cache: bool = True,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained models
            enable_cache: Enable prediction caching
            confidence_threshold: Threshold for high-confidence predictions
        """
        self.model_path = Path(model_path)
        self.enable_cache = enable_cache
        self.confidence_threshold = confidence_threshold
        
        # Load models
        self.preprocessor = None
        self.model = None
        self._load_models()
        
        # Prediction cache (in-memory)
        self.prediction_cache = {} if enable_cache else None
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'anomalies_detected': 0,
            'average_inference_time': 0.0,
            'total_inference_time': 0.0
        }
        
        logger.info("Predictor initialized successfully")
    
    def _load_models(self):
        """Load trained models"""
        try:
            # Load preprocessor
            self.preprocessor = TransactionPreprocessor()
            self.preprocessor.load(self.model_path / "preprocessor.joblib")
            
            # Load ensemble model
            self.model = EnsembleDetector()
            self.model.load(self.model_path / "ensemble.joblib")
            
            logger.info(f"Models loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def predict_single(
        self,
        transaction: Dict[str, Any],
        return_explanation: bool = False
    ) -> Dict[str, Any]:
        """
        Predict anomaly for a single transaction
        
        Args:
            transaction: Transaction data dictionary
            return_explanation: Include explanation in response
            
        Returns:
            Prediction result with score and metadata
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._create_cache_key(transaction)
        if self.enable_cache and cache_key in self.prediction_cache:
            self.stats['cache_hits'] += 1
            logger.debug(f"Cache hit for transaction {transaction.get('transaction_id')}")
            return self.prediction_cache[cache_key]
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([transaction])
            
            # Preprocess
            X = self.preprocessor.transform(df)
            
            # Predict
            prediction = self.model.predict(X)[0]
            score = self.model.predict_proba(X)[0]
            
            # Calculate confidence
            confidence = self._calculate_confidence(score)
            
            # Determine risk level
            risk_level = self._determine_risk_level(score)
            
            # Create result
            result = {
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'is_anomaly': bool(prediction),
                'anomaly_score': float(score),
                'confidence': float(confidence),
                'risk_level': risk_level,
                'inference_time_ms': (time.time() - start_time) * 1000,
                'model_version': '1.0.0'
            }
            
            # Add explanation if requested
            if return_explanation:
                result['explanation'] = self._generate_explanation(
                    transaction, score, prediction
                )
            
            # Update statistics
            self._update_stats(prediction, time.time() - start_time)
            
            # Cache result
            if self.enable_cache:
                self.prediction_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(
        self,
        transactions: List[Dict[str, Any]],
        batch_size: int = 1000,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict anomalies for multiple transactions
        
        Args:
            transactions: List of transaction dictionaries
            batch_size: Number of transactions to process at once
            show_progress: Show progress bar
            
        Returns:
            List of prediction results
        """
        logger.info(f"Starting batch prediction for {len(transactions)} transactions")
        start_time = time.time()
        
        results = []
        
        # Process in batches
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            
            # Convert to DataFrame
            df = pd.DataFrame(batch)
            
            # Preprocess
            X = self.preprocessor.transform(df)
            
            # Predict
            predictions = self.model.predict(X)
            scores = self.model.predict_proba(X)
            
            # Create results
            for j, txn in enumerate(batch):
                result = {
                    'transaction_id': txn.get('transaction_id', f'unknown_{i+j}'),
                    'is_anomaly': bool(predictions[j]),
                    'anomaly_score': float(scores[j]),
                    'confidence': float(self._calculate_confidence(scores[j])),
                    'risk_level': self._determine_risk_level(scores[j])
                }
                results.append(result)
            
            if show_progress and i % 1000 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(transactions)} transactions")
        
        total_time = time.time() - start_time
        logger.info(
            f"Batch prediction completed: {len(transactions)} transactions "
            f"in {total_time:.2f}s ({len(transactions)/total_time:.0f} txn/s)"
        )
        
        return results
    
    def predict_stream(
        self,
        transaction_stream: Any,
        window_size: int = 100,
        alert_callback: Optional[callable] = None
    ):
        """
        Predict on streaming data
        
        Args:
            transaction_stream: Iterator of transactions
            window_size: Number of transactions to buffer
            alert_callback: Function to call for alerts
        """
        buffer = []
        
        for transaction in transaction_stream:
            buffer.append(transaction)
            
            if len(buffer) >= window_size:
                # Process buffer
                results = self.predict_batch(buffer, show_progress=False)
                
                # Check for high-risk transactions
                for result in results:
                    if result['risk_level'] in ['high', 'critical']:
                        if alert_callback:
                            alert_callback(result)
                
                buffer.clear()
    
    def _calculate_confidence(self, score: float) -> float:
        """
        Calculate prediction confidence
        Score close to 0 or 1 = high confidence
        Score around 0.5 = low confidence
        """
        return 1 - (2 * abs(score - 0.5))
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from anomaly score"""
        if score >= 0.9:
            return 'critical'
        elif score >= 0.8:
            return 'high'
        elif score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _create_cache_key(self, transaction: Dict[str, Any]) -> str:
        """Create cache key from transaction"""
        # Use transaction_id if available
        if 'transaction_id' in transaction:
            return str(transaction['transaction_id'])
        
        # Otherwise, create hash from relevant fields
        key_fields = ['amount', 'merchant_category', 'location_distance_km', 'hour']
        key_values = [str(transaction.get(field, '')) for field in key_fields]
        return '_'.join(key_values)
    
    def _generate_explanation(
        self,
        transaction: Dict[str, Any],
        score: float,
        prediction: int
    ) -> Dict[str, Any]:
        """
        Generate explanation for prediction
        
        In production, this could use SHAP, LIME, or custom logic
        """
        explanation = {
            'factors': [],
            'reasoning': ''
        }
        
        if prediction == 1:
            # High amount
            if transaction.get('amount', 0) > 1000:
                explanation['factors'].append('Unusually high transaction amount')
            
            # Rapid transactions
            if transaction.get('time_since_last_transaction_minutes', 1000) < 5:
                explanation['factors'].append('Rapid succession of transactions')
            
            # Unusual location
            if transaction.get('location_distance_km', 0) > 500:
                explanation['factors'].append('Transaction far from usual location')
            
            # Unusual time
            if transaction.get('hour', 12) in [2, 3, 4]:
                explanation['factors'].append('Transaction at unusual time')
            
            # High velocity
            if transaction.get('transaction_count_1h', 0) > 10:
                explanation['factors'].append('High transaction velocity')
            
            explanation['reasoning'] = (
                f"Transaction flagged as anomaly (score: {score:.3f}) due to: "
                + ', '.join(explanation['factors'])
            )
        else:
            explanation['reasoning'] = (
                f"Transaction appears normal (score: {score:.3f})"
            )
        
        return explanation
    
    def _update_stats(self, prediction: int, inference_time: float):
        """Update prediction statistics"""
        self.stats['total_predictions'] += 1
        self.stats['anomalies_detected'] += int(prediction)
        self.stats['total_inference_time'] += inference_time
        self.stats['average_inference_time'] = (
            self.stats['total_inference_time'] / self.stats['total_predictions']
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics"""
        stats = self.stats.copy()
        
        if stats['total_predictions'] > 0:
            stats['anomaly_rate'] = (
                stats['anomalies_detected'] / stats['total_predictions']
            )
            stats['cache_hit_rate'] = (
                stats['cache_hits'] / stats['total_predictions']
                if self.enable_cache else 0.0
            )
        
        return stats
    
    def clear_cache(self):
        """Clear prediction cache"""
        if self.enable_cache:
            self.prediction_cache.clear()
            logger.info("Prediction cache cleared")
    
    def reload_models(self):
        """Reload models from disk (for model updates)"""
        logger.info("Reloading models...")
        self._load_models()
        self.clear_cache()
        logger.info("Models reloaded successfully")


class PredictionMonitor:
    """Monitor prediction performance and data drift"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions = []
        self.scores = []
        
    def record(self, prediction: int, score: float):
        """Record a prediction"""
        self.predictions.append(prediction)
        self.scores.append(score)
        
        # Keep only recent predictions
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.scores.pop(0)
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate monitoring metrics"""
        if not self.predictions:
            return {}
        
        return {
            'anomaly_rate': np.mean(self.predictions),
            'average_score': np.mean(self.scores),
            'score_std': np.std(self.scores),
            'high_risk_rate': np.mean([s > 0.8 for s in self.scores])
        }
    
    def detect_drift(self, baseline_anomaly_rate: float = 0.05) -> bool:
        """Detect if anomaly rate has drifted significantly"""
        if len(self.predictions) < 100:
            return False
        
        current_rate = np.mean(self.predictions)
        
        # Alert if anomaly rate is 2x or 0.5x baseline
        if current_rate > baseline_anomaly_rate * 2:
            logger.warning(
                f"Anomaly rate drift detected: {current_rate:.3f} "
                f"(baseline: {baseline_anomaly_rate:.3f})"
            )
            return True
        
        return False