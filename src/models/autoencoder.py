"""
PyTorch Autoencoder for anomaly detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Optional, Tuple
from loguru import logger
from tqdm import tqdm


class Autoencoder(nn.Module):
    """Deep Autoencoder architecture"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [32, 16, 8]):
        super(Autoencoder, self).__init__()
        
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i-1]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[i-1]),
                nn.Dropout(0.2)
            ])
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class AutoencoderDetector:
    """Autoencoder-based anomaly detector"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = None
        self.threshold = None
        
        logger.info(f"Using device: {self.device}")
    
    def fit(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None
    ):
        """Train the autoencoder"""
        
        input_dim = X.shape[1]
        hidden_dims = self.config.get('hidden_dims', [32, 16, 8])
        
        self.model = Autoencoder(input_dim, hidden_dims).to(self.device)
        
        # Training parameters
        batch_size = self.config.get('batch_size', 256)
        epochs = self.config.get('epochs', 50)
        learning_rate = self.config.get('learning_rate', 0.001)
        
        # Setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(X)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        if X_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(X_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        logger.info(f"Training autoencoder for {epochs} epochs")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        # Calculate threshold based on training reconstruction errors
        self._calculate_threshold(X)
        
        logger.info("Training completed")
        return self
    
    def _calculate_threshold(self, X: np.ndarray, percentile: float = 95):
        """Calculate anomaly threshold from reconstruction errors"""
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean(
                (X_tensor - reconstructed) ** 2, 
                dim=1
            ).cpu().numpy()
        
        self.threshold = np.percentile(reconstruction_errors, percentile)
        logger.info(f"Anomaly threshold set to {self.threshold:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        scores = self.get_anomaly_scores(X)
        return (scores > self.threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get normalized anomaly scores"""
        scores = self.get_anomaly_scores(X)
        scores_normalized = np.clip(scores / (self.threshold * 2), 0, 1)
        return scores_normalized
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction error scores"""
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean(
                (X_tensor - reconstructed) ** 2, 
                dim=1
            ).cpu().numpy()
        
        return reconstruction_errors
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Reconstruct model
        input_dim = checkpoint['model_state_dict']['encoder.0.weight'].shape[1]
        hidden_dims = self.config.get('hidden_dims', [32, 16, 8])
        self.model = Autoencoder(input_dim, hidden_dims).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        
        logger.info(f"Model loaded from {path}")
        return self