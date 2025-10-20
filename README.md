# Transaction Anomaly Detection System

[![CI Pipeline](https://github.com/Ran4er/anomaly-transaction/workflows/CI%20Pipeline/badge.svg)](https://github.com/Ran4er/anomaly-transaction/actions)
[![codecov](https://codecov.io/gh/Ran4er/anomaly-transaction/branch/main/graph/badge.svg)](https://codecov.io/gh/Ran4er/anomaly-transaction)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready unsupervised anomaly detection system for financial transactions, featuring Isolation Forest, PyTorch Autoencoder, and Ensemble methods.

## Features

- **Unsupervised Learning**: Detects anomalies without labeled data
- **Multiple Models**: Isolation Forest, Deep Autoencoder, and Ensemble
- **Production-Ready**: Complete CI/CD, Docker, monitoring, and alerting
- **Interactive UI**: Streamlit dashboard for real-time analysis
- **REST API**: FastAPI for seamless integration
- **Human-in-the-Loop**: Review queue for model feedback

## Performance Metrics

| Model | Precision@5% | Recall@5% | F1 Score | ROC AUC |
|-------|--------------|-----------|----------|---------|
| Isolation Forest | 0.851 | 0.812 | 0.831 | 0.924 |
| Autoencoder | 0.879 | 0.765 | 0.818 | 0.938 |
| **Ensemble** | **0.873** | **0.791** | **0.830** | **0.947** |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Ran4er/anomaly-transaction.git
cd anomaly-transaction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
```

### Generate Data & Train Models

```bash
# Generate synthetic transaction data
make generate-data

# Train all models
make train
```

### Run Applications

```bash
# Start FastAPI server
make run-api
# Access at http://localhost:8000

# Start Streamlit dashboard
make run-app
# Access at http://localhost:8501
```

### Docker Deployment

```bash
# Build and start all services
make docker-build
make docker-up

# Services:
# - API: http://localhost:8000
# - Streamlit: http://localhost:8501
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

## Project Structure

```
anomaly-transaction/
├── .github/workflows/      # CI/CD pipelines
├── src/
│   ├── data/              # Data generation and preprocessing
│   ├── models/            # ML models (IF, Autoencoder, Ensemble)
│   ├── training/          # Training and evaluation
│   ├── inference/         # Prediction and alerting
│   └── api/               # FastAPI application
├── app/                   # Streamlit dashboard
├── tests/                 # Unit and integration tests
├── docker/                # Docker configuration
├── configs/               # Model and app configs
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

## Configuration

Edit `configs/model_config.yaml` to customize:

```yaml
isolation_forest:
  n_estimators: 100
  contamination: 0.05

autoencoder:
  hidden_dims: [32, 16, 8]
  epochs: 50
  batch_size: 256

ensemble:
  weights: [0.5, 0.5]
```

## Testing

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Lint and format
make lint
make format
```

## API Usage

### Single Prediction

```python
import requests

transaction = {
    "transaction_id": "TXN_00000001",
    "timestamp": "2023-06-15 14:30:00",
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
# {
#   "transaction_id": "TXN_00000001",
#   "is_anomaly": false,
#   "anomaly_score": 0.234,
#   "risk_level": "low"
# }
```

### Batch Prediction

```python
transactions = [transaction1, transaction2, ...]
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"transactions": transactions}
)
```

## Dashboard Features

### Single Prediction
- Real-time anomaly detection
- Interactive risk gauge
- Detailed transaction analysis

### Batch Analysis
- Upload CSV files
- Bulk prediction
- Visual analytics
- Export results

### Model Performance
- Precision/Recall curves
- Confusion matrices
- Model comparison

## Production Integration

### Alerting System

```python
from src.inference.alert_service import AlertService

alert_service = AlertService()

# Automatically triggered for high-risk transactions
# Integrates with:
# - Email (SendGrid, AWS SES)
# - SMS (Twilio)
# - Slack/Teams webhooks
# - PagerDuty, Opsgenie
```

### Human-in-the-Loop

```python
from src.inference.alert_service import HumanInTheLoopService

hitl = HumanInTheLoopService()

# Add to review queue
hitl.add_to_review_queue(
    transaction_id="TXN_123",
    transaction_data=data,
    anomaly_score=0.85,
    model_prediction=1
)

# Submit human review
hitl.submit_review(
    transaction_id="TXN_123",
    human_label=1,  # Confirmed anomaly
    reviewer_id="analyst_001",
    notes="Suspected card fraud"
)

# Get feedback metrics
metrics = hitl.get_review_metrics()
```

## Monitoring

- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **MLflow**: Experiment tracking
- **Logs**: Structured logging with loguru

### Key Metrics
- Prediction latency
- Anomaly detection rate
- Model accuracy over time
- False positive rate
- API throughput

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│  FastAPI    │────▶│  PostgreSQL  │
│     API     │      │   Database   │
└──────┬──────┘      └──────────────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│   Models    │      │   Alert      │
│  Ensemble   │────▶│   Service    │
└─────────────┘      └──────┬───────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Slack/Email │
                    │    /SMS      │
                    └──────────────┘
```

## Security

- Input validation with Pydantic
- SQL injection prevention
- Rate limiting
- API authentication (add JWT)
- Secure secrets management

## Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/new-model
   ```

2. **Develop with tests**
   ```bash
   make format
   make lint
   make test
   ```

3. **Push and create PR**
   - CI pipeline runs automatically
   - Code coverage checked
   - Security scan performed

4. **Merge to main**
   - CD pipeline deploys to production
   - Model versioning with MLflow

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file

## Authors

- Your Name - [@Ran4er](https://github.com/Ran4er)

## Documentation

- [Architecture Guide](docs/architecture.md)
- [Model Card](docs/model_card.md)
- [API Documentation](docs/api_documentation.md)
- [Deployment Guide](docs/deployment.md)

## Known Issues

- None currently

## Roadmap

- [ ] Add LSTM-based sequence model
- [ ] Implement online learning
- [ ] Add more alert channels
- [ ] GraphQL API support
- [ ] Real-time streaming with Kafka

## Support

- Issues: [GitHub Issues](https://github.com/Ran4er/anomaly-transaction/issues)
- Discussions: [GitHub Discussions](https://github.com/Ran4er/anomaly-transaction/discussions)

---