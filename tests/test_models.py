"""
Unit tests for anomaly detection models
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.ensemble import EnsembleDetector
from src.data.preprocessor import TransactionPreprocessor
from src.data.generator import TransactionGenerator


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    generator = TransactionGenerator(seed=42)
    train_df, val_df, test_df = generator.generate_dataset(
        n_samples=1000,
        anomaly_ratio=0.05
    )
    return train_df, val_df, test_df


@pytest.fixture
def preprocessor(sample_data):
    """Create and fit preprocessor"""
    train_df, _, _ = sample_data
    prep = TransactionPreprocessor()
    train_features = train_df.drop(
        columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
        errors='ignore'
    )
    prep.fit(train_features)
    return prep


class TestIsolationForest:
    """Tests for Isolation Forest model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = IsolationForestDetector()
        assert model is not None
        assert model.model is not None
    
    def test_fit_predict(self, sample_data, preprocessor):
        """Test fitting and prediction"""
        train_df, _, test_df = sample_data
        
        # Prepare data
        X_train = preprocessor.transform(train_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        X_test = preprocessor.transform(test_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        
        # Train
        model = IsolationForestDetector()
        model.fit(X_train)
        
        # Predict
        predictions = model.predict(X_test)
        scores = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert len(scores) == len(X_test)
        assert predictions.dtype == int
        assert np.all((predictions == 0) | (predictions == 1))
        assert np.all((scores >= 0) & (scores <= 1))
    
    def test_anomaly_detection_rate(self, sample_data, preprocessor):
        """Test that model detects reasonable proportion of anomalies"""
        train_df, _, test_df = sample_data
        
        X_train = preprocessor.transform(train_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        X_test = preprocessor.transform(test_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        
        model = IsolationForestDetector({'contamination': 0.05})
        model.fit(X_train)
        predictions = model.predict(X_test)
        
        anomaly_rate = predictions.mean()
        assert 0.01 <= anomaly_rate <= 0.15  # Reasonable range


class TestAutoencoder:
    """Tests for Autoencoder model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = AutoencoderDetector({'hidden_dims': [16, 8], 'epochs': 5})
        assert model is not None
    
    def test_fit_predict(self, sample_data, preprocessor):
        """Test fitting and prediction"""
        train_df, val_df, test_df = sample_data
        
        X_train = preprocessor.transform(train_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        X_val = preprocessor.transform(val_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        X_test = preprocessor.transform(test_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        
        # Train with few epochs for speed
        model = AutoencoderDetector({'hidden_dims': [16, 8], 'epochs': 5, 'batch_size': 64})
        model.fit(X_train, X_val=X_val)
        
        # Predict
        predictions = model.predict(X_test)
        scores = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert len(scores) == len(X_test)
        assert model.threshold is not None
        assert predictions.dtype == int
        assert np.all((predictions == 0) | (predictions == 1))
        assert np.all((scores >= 0) & (scores <= 1))
    
    def test_autoencoder_save_load(self, sample_data, preprocessor, tmp_path):
        """Test autoencoder save and load"""
        train_df, val_df, test_df = sample_data
        
        X_train = preprocessor.transform(train_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        X_val = preprocessor.transform(val_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        X_test = preprocessor.transform(test_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        
        config = {'hidden_dims': [16, 8], 'epochs': 3, 'batch_size': 64}
        model = AutoencoderDetector(config)
        model.fit(X_train, X_val=X_val)
        
        predictions_before = model.predict(X_test)
        
        # Save and load
        model_path = tmp_path / "autoencoder.pt"
        model.save(str(model_path))
        
        loaded_model = AutoencoderDetector(config)
        loaded_model.load(str(model_path))
        
        predictions_after = loaded_model.predict(X_test)
        
        # Predictions should be consistent
        assert np.array_equal(predictions_before, predictions_after)


class TestEnsemble:
    """Tests for Ensemble model"""
    
    def test_initialization(self):
        """Test ensemble initialization"""
        model1 = IsolationForestDetector()
        model2 = IsolationForestDetector()
        
        ensemble = EnsembleDetector([model1, model2])
        assert len(ensemble.models) == 2
        assert len(ensemble.weights) == 2
    
    def test_ensemble_prediction(self, sample_data, preprocessor):
        """Test ensemble predictions"""
        train_df, _, test_df = sample_data
        
        X_train = preprocessor.transform(train_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        X_test = preprocessor.transform(test_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        
        # Train two models
        model1 = IsolationForestDetector()
        model1.fit(X_train)
        
        model2 = IsolationForestDetector({'contamination': 0.1})
        model2.fit(X_train)
        
        # Create ensemble
        ensemble = EnsembleDetector([model1, model2], weights=[0.6, 0.4])
        
        # Predict
        predictions = ensemble.predict(X_test, method='voting')
        scores = ensemble.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert len(scores) == len(X_test)
        assert predictions.dtype == int
        assert np.all((predictions == 0) | (predictions == 1))
        assert np.all((scores >= 0) & (scores <= 1))
    
    def test_ensemble_empty_raises(self):
        """Test that empty ensemble raises error"""
        ensemble = EnsembleDetector()
        X_test = np.random.randn(10, 5)
        
        with pytest.raises(ValueError):
            ensemble.predict(X_test)
    
    def test_ensemble_add_model(self, sample_data, preprocessor):
        """Test adding models to ensemble"""
        train_df, _, test_df = sample_data
        
        X_train = preprocessor.transform(train_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        X_test = preprocessor.transform(test_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        
        model1 = IsolationForestDetector()
        model1.fit(X_train)
        
        ensemble = EnsembleDetector([model1])
        assert len(ensemble.models) == 1
        
        model2 = IsolationForestDetector({'contamination': 0.1})
        model2.fit(X_train)
        ensemble.add_model(model2, weight=0.5)
        
        assert len(ensemble.models) == 2
        assert len(ensemble.weights) == 2
        
        # Test prediction still works
        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_ensemble_agreement_metrics(self, sample_data, preprocessor):
        """Test ensemble agreement metrics"""
        train_df, _, test_df = sample_data
        
        X_train = preprocessor.transform(train_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        X_test = preprocessor.transform(test_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        
        model1 = IsolationForestDetector({'random_state': 42})
        model1.fit(X_train)
        
        model2 = IsolationForestDetector({'random_state': 43})
        model2.fit(X_train)
        
        ensemble = EnsembleDetector([model1, model2])
        agreement = ensemble.evaluate_agreement(X_test)
        
        assert 'full_agreement' in agreement
        assert 'majority_agreement' in agreement
        assert 'average_pairwise_agreement' in agreement
        
        for metric, value in agreement.items():
            assert 0.0 <= value <= 1.0
    
    def test_ensemble_model_predictions(self, sample_data, preprocessor):
        """Test getting individual model predictions"""
        train_df, _, test_df = sample_data
        
        X_train = preprocessor.transform(train_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        X_test = preprocessor.transform(test_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        ))
        
        model1 = IsolationForestDetector()
        model1.fit(X_train)
        
        model2 = IsolationForestDetector({'contamination': 0.1})
        model2.fit(X_train)
        
        ensemble = EnsembleDetector([model1, model2])
        predictions = ensemble.get_model_predictions(X_test)
        
        assert len(predictions) == 2
        for model_name, pred_dict in predictions.items():
            assert 'binary' in pred_dict
            assert 'scores' in pred_dict
            assert len(pred_dict['binary']) == len(X_test)
            assert len(pred_dict['scores']) == len(X_test)


class TestPreprocessor:
    """Tests for data preprocessor"""
    
    def test_fit_transform(self, sample_data):
        """Test fit and transform"""
        train_df, _, _ = sample_data
        
        prep = TransactionPreprocessor()
        train_features = train_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        )
        
        X = prep.fit_transform(train_features)
        
        assert X.shape[0] == len(train_features)
        assert X.shape[1] > 0
        assert not np.isnan(X).any()
    
    def test_transform_consistency(self, sample_data):
        """Test that transform is consistent"""
        train_df, val_df, _ = sample_data
        
        prep = TransactionPreprocessor()
        train_features = train_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        )
        
        prep.fit(train_features)
        
        X1 = prep.transform(train_features.head(10))
        X2 = prep.transform(train_features.head(10))
        
        assert np.allclose(X1, X2)
    
    def test_preprocessor_save_load(self, sample_data, tmp_path):
        """Test preprocessor save and load"""
        train_df, _, _ = sample_data
        
        prep = TransactionPreprocessor()
        train_features = train_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        )
        
        X_before = prep.fit_transform(train_features)
        
        # Save and load
        prep_path = tmp_path / "preprocessor.joblib"
        prep.save(str(prep_path))
        
        loaded_prep = TransactionPreprocessor()
        loaded_prep.load(str(prep_path))
        
        X_after = loaded_prep.transform(train_features)
        
        # Results should be identical
        assert np.allclose(X_before, X_after)
    
    def test_preprocessor_feature_names(self, sample_data):
        """Test that feature names are properly set"""
        train_df, _, _ = sample_data
        
        prep = TransactionPreprocessor()
        train_features = train_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        )
        
        prep.fit(train_features)
        
        assert prep.feature_names is not None
        assert len(prep.feature_names) > 0
        assert len(prep.feature_names) == train_features.shape[1]


class TestDataGenerator:
    """Tests for data generator"""
    
    def test_generate_normal_transactions(self):
        """Test generating normal transactions"""
        generator = TransactionGenerator(seed=42)
        df = generator.generate_normal_transactions(n_samples=100)
        
        assert len(df) == 100
        assert 'transaction_id' in df.columns
        assert 'amount' in df.columns
        assert df['is_anomaly'].sum() == 0
    
    def test_inject_anomalies(self):
        """Test anomaly injection"""
        generator = TransactionGenerator(seed=42)
        df = generator.generate_normal_transactions(n_samples=100)
        df_with_anomalies = generator.inject_anomalies(df, anomaly_ratio=0.1)
        
        n_anomalies = df_with_anomalies['is_anomaly'].sum()
        assert n_anomalies == 10
        assert 'anomaly_type' in df_with_anomalies.columns
    
    def test_dataset_generation(self):
        """Test full dataset generation"""
        generator = TransactionGenerator(seed=42)
        train, val, test = generator.generate_dataset(n_samples=1000)
        
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == 1000
    
    def test_anomaly_types_distribution(self):
        """Test that different anomaly types are created"""
        generator = TransactionGenerator(seed=42)
        df = generator.generate_normal_transactions(n_samples=1000)
        df_with_anomalies = generator.inject_anomalies(df, anomaly_ratio=0.1)
        
        # Check that different anomaly types exist
        anomaly_types = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1]['anomaly_type'].unique()
        assert len(anomaly_types) > 1  # Should have multiple types
        
        # Check that anomaly types are properly assigned
        assert set(anomaly_types).issubset({1, 2, 3, 4})
    
    def test_data_consistency(self):
        """Test that generated data has consistent structure"""
        generator = TransactionGenerator(seed=42)
        train, val, test = generator.generate_dataset(n_samples=500)
        
        # Check that all splits have the same columns
        expected_columns = ['transaction_id', 'amount', 'merchant_category', 'is_anomaly']
        for df in [train, val, test]:
            for col in expected_columns:
                assert col in df.columns
        
        # Check that amounts are positive
        for df in [train, val, test]:
            assert (df['amount'] > 0).all()
        
        # Check that anomaly flags are binary
        for df in [train, val, test]:
            assert df['is_anomaly'].isin([0, 1]).all()
    
    def test_reproducibility(self):
        """Test that generator produces same results with same seed"""
        generator1 = TransactionGenerator(seed=42)
        generator2 = TransactionGenerator(seed=42)
        
        df1 = generator1.generate_normal_transactions(n_samples=100)
        df2 = generator2.generate_normal_transactions(n_samples=100)
        
        # Should be identical
        assert df1.equals(df2)


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_isolation_forest_empty_data(self):
        """Test isolation forest with empty data"""
        model = IsolationForestDetector()
        X_empty = np.array([]).reshape(0, 5)
        
        # Should handle empty data gracefully
        model.fit(X_empty)
        predictions = model.predict(X_empty)
        assert len(predictions) == 0
    
    def test_autoencoder_single_sample(self, preprocessor):
        """Test autoencoder with single sample"""
        # Create minimal data
        X_single = np.random.randn(1, 10)
        
        config = {'hidden_dims': [8, 4], 'epochs': 1, 'batch_size': 1}
        model = AutoencoderDetector(config)
        
        # Should handle single sample
        model.fit(X_single)
        predictions = model.predict(X_single)
        assert len(predictions) == 1
    
    def test_ensemble_weight_mismatch(self):
        """Test ensemble with weight mismatch"""
        model1 = IsolationForestDetector()
        model2 = IsolationForestDetector()
        
        # Should raise error for weight mismatch
        with pytest.raises(ValueError):
            EnsembleDetector([model1, model2], weights=[0.5])  # Only one weight for two models
    
    def test_preprocessor_missing_features(self):
        """Test preprocessor with missing features"""
        prep = TransactionPreprocessor()
        
        # Create data with missing features
        df = pd.DataFrame({
            'amount': [100, 200, 300],
            'merchant_category': ['retail', 'food', 'transport']
        })
        
        # Should handle missing features gracefully
        X = prep.fit_transform(df)
        assert X.shape[0] == 3
        assert X.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])