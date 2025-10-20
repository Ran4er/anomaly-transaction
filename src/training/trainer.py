"""
Training pipeline for anomaly detection models
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import yaml
import mlflow

from src.data.preprocessor import TransactionPreprocessor
from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.ensemble import EnsembleDetector
from src.training.evaluator import ModelEvaluator


class AnomalyDetectionTrainer:
    """Training pipeline for anomaly detection"""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """Initialize trainer with configuration"""
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessor = TransactionPreprocessor()
        self.models = {}
        self.evaluator = ModelEvaluator()
        
        logger.info("Trainer initialized")
    
    def load_data(
        self, 
        train_path: str, 
        val_path: Optional[str] = None,
        test_path: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load training data"""
        
        logger.info(f"Loading data from {train_path}")
        
        data = {
            'train': pd.read_csv(train_path)
        }
        
        if val_path:
            data['val'] = pd.read_csv(val_path)
        
        if test_path:
            data['test'] = pd.read_csv(test_path)
        
        logger.info(f"Loaded train: {len(data['train'])} samples")
        if 'val' in data:
            logger.info(f"Loaded val: {len(data['val'])} samples")
        if 'test' in data:
            logger.info(f"Loaded test: {len(data['test'])} samples")
        
        return data
    
    def prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Preprocess and prepare data"""
        
        logger.info("Preprocessing data")
        
        train_features = data['train'].drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        )
        self.preprocessor.fit(train_features)
        
        X_train = self.preprocessor.transform(train_features)
        y_train = data['train']['is_anomaly'].values if 'is_anomaly' in data['train'].columns else None
        
        prepared = {
            'X_train': X_train,
            'y_train': y_train
        }
        
        if 'val' in data:
            val_features = data['val'].drop(
                columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
                errors='ignore'
            )
            prepared['X_val'] = self.preprocessor.transform(val_features)
            prepared['y_val'] = data['val']['is_anomaly'].values if 'is_anomaly' in data['val'].columns else None
        
        if 'test' in data:
            test_features = data['test'].drop(
                columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
                errors='ignore'
            )
            prepared['X_test'] = self.preprocessor.transform(test_features)
            prepared['y_test'] = data['test']['is_anomaly'].values if 'is_anomaly' in data['test'].columns else None
        
        logger.info(f"Data prepared: {X_train.shape}")
        
        return prepared
    
    def train_isolation_forest(self, X_train: np.ndarray) -> IsolationForestDetector:
        """Train Isolation Forest model"""
        
        logger.info("Training Isolation Forest")
        
        config = self.config.get('isolation_forest', {})
        model = IsolationForestDetector(config)
        model.fit(X_train)
        
        self.models['isolation_forest'] = model
        
        return model
    
    def train_autoencoder(
        self, 
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None
    ) -> AutoencoderDetector:
        """Train Autoencoder model"""
        
        logger.info("Training Autoencoder")
        
        config = self.config.get('autoencoder', {})
        model = AutoencoderDetector(config)
        model.fit(X_train, X_val=X_val)
        
        self.models['autoencoder'] = model
        
        return model
    
    def train_ensemble(self) -> EnsembleDetector:
        """Train ensemble model"""
        
        logger.info("Creating ensemble model")
        
        if not self.models:
            raise ValueError("No models trained yet")
        
        models = list(self.models.values())
        weights = self.config.get('ensemble', {}).get('weights', [1.0] * len(models))
        
        ensemble = EnsembleDetector(models, weights)
        self.models['ensemble'] = ensemble
        
        return ensemble
    
    def train_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train all models"""
        
        logger.info("Starting full training pipeline")
        
        prepared_data = self.prepare_data(data)
        
        X_train = prepared_data['X_train']
        X_val = prepared_data.get('X_val')
        
        self.train_isolation_forest(X_train)
        self.train_autoencoder(X_train, X_val)
        self.train_ensemble()
        
        results = {}
        if 'X_val' in prepared_data and 'y_val' in prepared_data:
            logger.info("Evaluating models on validation set")
            
            for model_name, model in self.models.items():
                y_pred = model.predict(prepared_data['X_val'])
                y_scores = model.predict_proba(prepared_data['X_val'])
                
                metrics = self.evaluator.evaluate(
                    prepared_data['y_val'],
                    y_pred,
                    y_scores
                )
                
                results[model_name] = metrics
                logger.info(f"{model_name} - Precision@5%: {metrics['precision_at_5']:.4f}, "
                          f"Recall@5%: {metrics['recall_at_5']:.4f}, "
                          f"F1: {metrics['f1']:.4f}")
        
        logger.info("Training pipeline completed")
        
        return results
    
    def save_models(self, output_dir: str = "data/models"):
        """Save all trained models"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor.save(output_path / "preprocessor.joblib")
        
        for model_name, model in self.models.items():
            if model_name == 'ensemble':
                model.save(output_path / f"{model_name}.joblib")
            elif model_name == 'autoencoder':
                model.save(output_path / f"{model_name}.pth")
            else:
                model.save(output_path / f"{model_name}.joblib")
        
        logger.info(f"Models saved to {output_dir}")
    
    def load_models(self, model_dir: str = "data/models"):
        """Load trained models"""
        
        model_path = Path(model_dir)
        
        self.preprocessor.load(model_path / "preprocessor.joblib")
        
        if (model_path / "isolation_forest.joblib").exists():
            model = IsolationForestDetector()
            model.load(model_path / "isolation_forest.joblib")
            self.models['isolation_forest'] = model
        
        if (model_path / "autoencoder.pth").exists():
            model = AutoencoderDetector(self.config.get('autoencoder', {}))
            model.load(model_path / "autoencoder.pth")
            self.models['autoencoder'] = model
        
        if (model_path / "ensemble.joblib").exists():
            model = EnsembleDetector()
            model.load(model_path / "ensemble.joblib")
            self.models['ensemble'] = model
        
        logger.info(f"Models loaded from {model_dir}")