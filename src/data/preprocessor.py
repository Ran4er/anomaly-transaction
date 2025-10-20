"""
Data preprocessing and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Optional, Tuple
import joblib
from loguru import logger


class TransactionPreprocessor:
    """Preprocess transaction data for anomaly detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = ['merchant_category']
        self.numerical_features = [
            'amount_log', 'location_distance_km', 
            'time_since_last_transaction_minutes',
            'hour', 'day_of_week', 'transaction_count_1h', 
            'total_amount_24h', 'is_online', 'is_weekend'
        ]
        
    def fit(self, df: pd.DataFrame) -> 'TransactionPreprocessor':
        """Fit preprocessor on training data"""
        
        logger.info("Fitting preprocessor")
        
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
        
        X = self._prepare_features(df, fit=True)
        
        self.scaler.fit(X)
        self.feature_names = self.numerical_features.copy()
        
        for col in self.categorical_features:
            if col in df.columns:
                self.feature_names.append(f"{col}_encoded")
        
        logger.info(f"Preprocessor fitted with {len(self.feature_names)} features")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data"""
        
        X = self._prepare_features(df, fit=False)
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
    
    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Prepare feature matrix"""
        
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if 'hour' not in df.columns:
                df['hour'] = df['timestamp'].dt.hour
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        if 'amount_log' not in df.columns and 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
        
        features = []
        for col in self.numerical_features:
            if col in df.columns:
                features.append(df[col].values)
            else:
                logger.warning(f"Feature {col} not found, using zeros")
                features.append(np.zeros(len(df)))
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in df.columns:
                if fit:
                    # This should not happen, encoders are fit separately
                    pass
                
                if col in self.label_encoders:
                    encoded = self.label_encoders[col].transform(
                        df[col].astype(str)
                    )
                    features.append(encoded)
                else:
                    logger.warning(f"Encoder for {col} not found")
                    features.append(np.zeros(len(df)))
        
        X = np.column_stack(features)
        return X
    
    def inverse_transform_numerical(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform for numerical features only"""
        return self.scaler.inverse_transform(X)
    
    def save(self, path: str):
        """Save preprocessor"""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: str):
        """Load preprocessor"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        self.categorical_features = data['categorical_features']
        self.numerical_features = data['numerical_features']
        logger.info(f"Preprocessor loaded from {path}")
        return self


def create_time_windows(
    df: pd.DataFrame, 
    window_size: str = '1H'
) -> pd.DataFrame:
    """Create aggregated features over time windows"""
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    windowed = df.groupby([
        'user_id',
        pd.Grouper(key='timestamp', freq=window_size)
    ]).agg({
        'amount': ['sum', 'mean', 'std', 'count'],
        'transaction_id': 'count'
    }).reset_index()
    
    windowed.columns = ['_'.join(col).strip('_') for col in windowed.columns]
    
    return windowed