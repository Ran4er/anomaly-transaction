# Quick Start Guide

This guide will help you get the Transaction Anomaly Detection system up and running in 5 minutes.

## Prerequisites

- Python 3.9+ installed
- Git installed
- Docker and Docker Compose (optional, for containerized deployment)
- 4GB+ RAM
- 2GB+ free disk space

## Step 1: Clone Repository

```bash
git clone https://github.com/Ran4er/anomaly-transaction.git
cd anomaly-transaction
```

## Step 2: Setup Environment

### Option A: Using Make (Recommended)

```bash
# Create virtual environment
make setup

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
make install
```

### Option B: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Step 3: Generate Sample Data

```bash
# Generate 50,000 synthetic transactions
make generate-data

# Or with custom parameters
python scripts/generate_data.py --n-samples 100000 --anomaly-ratio 0.05
```

**Expected output:**
```
Generated datasets:
   - data/raw/train.csv (35,000 samples)
   - data/raw/val.csv (7,500 samples)
   - data/raw/test.csv (7,500 samples)
```

## Step 4: Train Models

```bash
# Train all models (Isolation Forest, Autoencoder, Ensemble)
make train

# Training takes approximately 3-5 minutes
```

**Expected output:**
```
Models trained:
   - Isolation Forest: Precision@5%: 0.851
   - Autoencoder: Precision@5%: 0.879
   - Ensemble: Precision@5%: 0.873
   
Models saved to: data/models/
```

## Step 5: Run the Application

### Option A: Streamlit Dashboard

```bash
# Start interactive dashboard
make run-app
```

Open browser: **http://localhost:8501**

### Option B: FastAPI Server

```bash
# Start API server
make run-api
```

Open browser: **http://localhost:8000/docs** for API documentation

### Option C: Both (Recommended)

```bash
# Terminal 1
make run-api

# Terminal 2
make run-app
```

## Step 6: Make Your First Prediction

### Using Streamlit Dashboard

1. Navigate to **http://localhost:8501**
2. Select "Single Prediction" from sidebar
3. Fill in transaction details
4. Click "Analyze Transaction"
5. View anomaly score and risk level

### Using API (curl)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_TEST_001",
    "timestamp": "2023-12-15 14:30:00",
    "amount": 150.50,
    "merchant_category": "retail",
    "location_distance_km": 5.2,
    "is_online": 0,
    "time_since_last_transaction_minutes": 120.0,
    "is_weekend": 0,
    "hour": 14,
    "day_of_week": 3,
    "transaction_count_1h": 2,
    "total_amount_24h": 300.0,
    "user_id": 1234
  }'
```

### Using Python

```python
import requests

transaction = {
    "transaction_id": "TXN_TEST_001",
    "timestamp": "2023-12-15 14:30:00",
    "amount": 150.50,
    "merchant_category": "retail",
    "location_distance_km": 5.2,
    "is_online": 0,
    "time_since_last_transaction_minutes": 120.0,
    "is_weekend": 0,
    "hour": 14,
    "day_of_week": 3,
    "transaction_count_1h": 2,
    "total_amount_24h": 300.0,
    "user_id": 1234
}

response = requests.post(
    "http://localhost:8000/predict",
    json=transaction
)

print(response.json())
```

## Step 7: Test Batch Processing

1. Go to Streamlit dashboard
2. Select "Batch Analysis"
3. Upload CSV file (use `data/raw/test.csv`)
4. Click "Analyze All Transactions"
5. Download results

## Docker Deployment (Alternative)

If you prefer Docker:

```bash
# Build images
make docker-build

# Start all services
make docker-up

# Services will be available at:
# - API: http://localhost:8000
# - Streamlit: http://localhost:8501
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000

# Stop services
make docker-down
```

## Verify Installation

Run tests to verify everything works:

```bash
# Run all tests
make test

# Quick test
make test-fast

# Check code quality
make lint
```

**Expected output:**
```
All tests passed
Code quality checks passed
```

## Common Issues & Solutions

### Issue: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "Port already in use"
**Solution:**
```bash
# Change port in command
uvicorn src.api.main:app --port 8001
streamlit run app/streamlit_app.py --server.port 8502
```

### Issue: "Models not found"
**Solution:**
```bash
# Retrain models
make train
```

### Issue: "CUDA out of memory" (Autoencoder)
**Solution:**
Edit `configs/model_config.yaml`:
```yaml
autoencoder:
  batch_size: 128  # Reduce from 256
  hidden_dims: [16, 8]  # Reduce from [32, 16, 8]
```

## Next Steps

Now that you have the system running:

1. **Explore the Dashboard**
   - Try different transaction patterns
   - Upload your own CSV files
   - View model performance metrics

2. **Customize Configuration**
   - Edit `configs/model_config.yaml`
   - Adjust anomaly thresholds
   - Configure alert settings

3. **Integrate with Your System**
   - Use the REST API
   - Set up webhooks
   - Configure alerts

4. **Deploy to Production**
   - Review `docs/architecture.md`
   - Set up monitoring
   - Configure CI/CD

## Resources

- [Full Documentation](README.md)
- [Architecture Guide](docs/architecture.md)
- [API Documentation](http://localhost:8000/docs)
- [GitHub Issues](https://github.com/Ran4er/anomaly-transaction/issues)

## Helpful Commands

```bash
# Data management
make generate-data          # Generate new dataset
make clean                  # Clean temporary files

# Model training
make train                  # Train all models
make evaluate              # Evaluate models

# Development
make format                # Format code
make lint                  # Check code quality
make test                  # Run tests

# Deployment
make docker-build          # Build Docker images
make docker-up             # Start containers
make docker-down           # Stop containers
make docker-logs           # View logs
```

## Getting Help

If you encounter any issues:

1. Check the [README](README.md)
2. Review [Architecture docs](docs/architecture.md)
3. Search [GitHub Issues](https://github.com/Ran4er/anomaly-transaction/issues)
4. Open a new issue with details

---

**Congratulations!** You now have a production-ready anomaly detection system running locally.