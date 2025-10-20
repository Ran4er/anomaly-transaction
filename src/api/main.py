"""
FastAPI application for anomaly detection
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from pathlib import Path

from src.data.database import Transaction, PredictionResponse, BatchPredictionRequest, HealthResponse
from src.data.preprocessor import TransactionPreprocessor
from src.models.ensemble import EnsembleDetector
from src.inference.alert_service import AlertService

app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="Real-time anomaly detection for financial transactions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

preprocessor = None
model = None
alert_service = None


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global preprocessor, model, alert_service
    
    try:
        model_dir = Path("data/models")
        
        preprocessor = TransactionPreprocessor()
        preprocessor.load(model_dir / "preprocessor.joblib")
        
        model = EnsembleDetector()
        model.load(model_dir / "ensemble.joblib")
        
        alert_service = AlertService()
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        preprocessor = None
        model = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Transaction Anomaly Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_transaction(
    transaction: Transaction,
    background_tasks: BackgroundTasks
):
    """
    Predict if a single transaction is anomalous
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([transaction.dict()])
        
        X = preprocessor.transform(df)
        
        is_anomaly = model.predict(X)[0]
        anomaly_score = model.predict_proba(X)[0]
        
        if anomaly_score > 0.8:
            risk_level = "high"
        elif anomaly_score > 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        response = PredictionResponse(
            transaction_id=transaction.transaction_id,
            is_anomaly=bool(is_anomaly),
            anomaly_score=float(anomaly_score),
            confidence=float(1 - abs(0.5 - anomaly_score) * 2),
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )
        
        if is_anomaly and alert_service:
            background_tasks.add_task(
                alert_service.send_alert,
                transaction.dict(),
                anomaly_score
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict anomalies for multiple transactions
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([t.dict() for t in request.transactions])
        
        X = preprocessor.transform(df)
        
        predictions = model.predict(X)
        scores = model.predict_proba(X)
        
        results = []
        for i, txn in enumerate(request.transactions):
            risk_level = "high" if scores[i] > 0.8 else "medium" if scores[i] > 0.5 else "low"
            
            results.append({
                "transaction_id": txn.transaction_id,
                "is_anomaly": bool(predictions[i]),
                "anomaly_score": float(scores[i]),
                "confidence": float(1 - abs(0.5 - scores[i]) * 2),
                "risk_level": risk_level
            })
        
        return {
            "predictions": results,
            "total_transactions": len(results),
            "anomalies_detected": int(predictions.sum()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/info", tags=["Models"])
async def model_info():
    """Get information about loaded models"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Ensemble",
        "n_models": len(model.models),
        "model_names": [m.__class__.__name__ for m in model.models],
        "feature_count": len(preprocessor.feature_names) if preprocessor else 0,
        "features": preprocessor.feature_names if preprocessor else []
    }


@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """Get prediction statistics"""
    return {
        "total_predictions": 0,
        "anomalies_detected": 0,
        "average_score": 0.0,
        "last_updated": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)