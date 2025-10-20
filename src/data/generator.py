"""
Generate synthetic transaction data with anomalies
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
from loguru import logger


class TransactionGenerator:
    """Generate realistic transaction data with injected anomalies"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
    def generate_normal_transactions(
        self, 
        n_samples: int = 10000,
        start_date: str = "2023-01-01"
    ) -> pd.DataFrame:
        """Generate normal transaction patterns"""
        
        logger.info(f"Generating {n_samples} normal transactions")
        
        start = pd.to_datetime(start_date)
        
        # Normal transaction patterns
        data = {
            'transaction_id': [f'TXN_{i:08d}' for i in range(n_samples)],
            'timestamp': [start + timedelta(
                minutes=np.random.randint(0, 525600)  # Random time in year
            ) for _ in range(n_samples)],
            'amount': np.random.lognormal(mean=4.5, sigma=1.2, size=n_samples),
            'merchant_category': np.random.choice(
                ['retail', 'food', 'transport', 'entertainment', 'utilities'],
                size=n_samples,
                p=[0.3, 0.25, 0.2, 0.15, 0.1]
            ),
            'location_distance_km': np.abs(np.random.normal(10, 15, n_samples)),
            'is_online': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
            'time_since_last_transaction_minutes': np.abs(
                np.random.exponential(scale=120, size=n_samples)
            ),
            'is_weekend': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
            'user_id': np.random.randint(1000, 5000, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['amount_log'] = np.log1p(df['amount'])
        
        # Add velocity features
        df['transaction_count_1h'] = np.random.poisson(lam=2, size=n_samples)
        df['total_amount_24h'] = df['amount'] * np.random.uniform(1, 5, n_samples)
        
        df['is_anomaly'] = 0
        
        return df
    
    def inject_anomalies(
        self, 
        df: pd.DataFrame, 
        anomaly_ratio: float = 0.05
    ) -> pd.DataFrame:
        """Inject various types of anomalies"""
        
        n_anomalies = int(len(df) * anomaly_ratio)
        logger.info(f"Injecting {n_anomalies} anomalies ({anomaly_ratio*100:.1f}%)")
        
        anomaly_indices = np.random.choice(
            df.index, 
            size=n_anomalies, 
            replace=False
        )
        
        df_copy = df.copy()
        
        # Type 1: Unusually high amounts (40% of anomalies)
        type1_count = int(n_anomalies * 0.4)
        type1_idx = anomaly_indices[:type1_count]
        df_copy.loc[type1_idx, 'amount'] *= np.random.uniform(10, 50, type1_count)
        
        # Type 2: Rapid succession (30% of anomalies)
        type2_count = int(n_anomalies * 0.3)
        type2_idx = anomaly_indices[type1_count:type1_count + type2_count]
        df_copy.loc[type2_idx, 'time_since_last_transaction_minutes'] = np.random.uniform(
            0.1, 2, type2_count
        )
        df_copy.loc[type2_idx, 'transaction_count_1h'] = np.random.randint(
            10, 30, type2_count
        )
        
        # Type 3: Unusual location (20% of anomalies)
        type3_count = int(n_anomalies * 0.2)
        type3_idx = anomaly_indices[type1_count + type2_count:type1_count + type2_count + type3_count]
        df_copy.loc[type3_idx, 'location_distance_km'] = np.random.uniform(
            500, 5000, type3_count
        )
        
        # Type 4: Unusual time patterns (10% of anomalies)
        type4_idx = anomaly_indices[type1_count + type2_count + type3_count:]
        df_copy.loc[type4_idx, 'hour'] = np.random.choice([2, 3, 4], len(type4_idx))
        df_copy.loc[type4_idx, 'amount'] *= np.random.uniform(3, 10, len(type4_idx))
        
        # Mark all anomalies
        df_copy.loc[anomaly_indices, 'is_anomaly'] = 1
        df_copy['anomaly_type'] = 0
        df_copy.loc[type1_idx, 'anomaly_type'] = 1  # High amount
        df_copy.loc[type2_idx, 'anomaly_type'] = 2  # Rapid succession
        df_copy.loc[type3_idx, 'anomaly_type'] = 3  # Unusual location
        df_copy.loc[type4_idx, 'anomaly_type'] = 4  # Unusual time
        
        # Recalculate log features
        df_copy['amount_log'] = np.log1p(df_copy['amount'])
        
        logger.info(f"Anomaly distribution: High amount={type1_count}, "
                   f"Rapid={type2_count}, Location={type3_count}, Time={len(type4_idx)}")
        
        return df_copy
    
    def generate_dataset(
        self,
        n_samples: int = 10000,
        anomaly_ratio: float = 0.05,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset with train/val/test splits"""
        
        logger.info(f"Generating dataset: {n_samples} samples, {anomaly_ratio*100}% anomalies")
        
        df = self.generate_normal_transactions(n_samples)
        
        df = self.inject_anomalies(df, anomaly_ratio)
        
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        n_train = int(len(df) * train_ratio)
        n_val = int(len(df) * val_ratio)
        
        train_df = df[:n_train].copy()
        val_df = df[n_train:n_train + n_val].copy()
        test_df = df[n_train + n_val:].copy()
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Train anomalies: {train_df['is_anomaly'].sum()} "
                   f"({train_df['is_anomaly'].mean()*100:.2f}%)")
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    generator = TransactionGenerator(seed=42)
    train, val, test = generator.generate_dataset(n_samples=50000)
    
    train.to_csv('data/raw/train.csv', index=False)
    val.to_csv('data/raw/val.csv', index=False)
    test.to_csv('data/raw/test.csv', index=False)
    
    print("Dataset generated successfully!")