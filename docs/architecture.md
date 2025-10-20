# System Architecture

## Overview

The Transaction Anomaly Detection System is designed as a microservices-based architecture with clear separation of concerns, enabling scalability, maintainability, and production deployment.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  ┌─────────────────┐              ┌──────────────────────┐   │
│  │  Web Dashboard  │              │   Mobile/External    │   │
│  │   (Streamlit)   │              │      Clients         │   │
│  └────────┬────────┘              └──────────┬───────────┘   │
└───────────┼──────────────────────────────────┼───────────────┘
            │                                  │
            ▼                                  ▼
┌──────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                       │
│                  ┌──────────────────────┐                    │
│                  │   FastAPI Backend    │                    │
│                  │  - REST Endpoints    │                    │
│                  │  - Input Validation  │                    │
│                  │  - Authentication    │                    │
│                  └──────────┬───────────┘                    │
└─────────────────────────────┼────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Preprocessing   │ │    Model     │ │     Alert        │
│     Service      │ │  Inference   │ │    Service       │
│                  │ │   Service    │ │                  │
│ - Feature Eng.   │ │ - Isolation  │ │ - Notifications  │
│ - Normalization  │ │   Forest     │ │ - Human Review   │
│ - Validation     │ │ - Autoencoder│ │ - Ticketing      │
│                  │ │ - Ensemble   │ │                  │
└──────────────────┘ └──────┬───────┘ └──────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │  Data Storage  │
                   │                │
                   │ - PostgreSQL   │
                   │ - Model Store  │
                   │ - Alert Logs   │
                   └────────────────┘
```

## Component Details

### 1. Data Layer

#### Data Generation (`src/data/generator.py`)
- Generates synthetic transaction data
- Injects various anomaly types:
  - High amount anomalies (40%)
  - Rapid succession (30%)
  - Unusual location (20%)
  - Unusual time patterns (10%)

#### Data Preprocessing (`src/data/preprocessor.py`)
- Feature engineering
- Normalization with StandardScaler
- Label encoding for categorical features
- Handles missing values

### 2. Model Layer

#### Isolation Forest (`src/models/isolation_forest.py`)
- **Algorithm**: Ensemble of isolation trees
- **Strengths**: Fast, interpretable, works well with high-dimensional data
- **Contamination**: 5% (configurable)
- **Use Case**: Initial screening, real-time detection

#### Autoencoder (`src/models/autoencoder.py`)
- **Architecture**: Deep neural network (input → 32 → 16 → 8 → 16 → 32 → input)
- **Training**: Unsupervised reconstruction error minimization
- **Strengths**: Captures complex non-linear patterns
- **Use Case**: Deep analysis, batch processing

#### Ensemble (`src/models/ensemble.py`)
- **Method**: Weighted voting
- **Weights**: [0.5, 0.5] for IF and AE (configurable)
- **Benefit**: Combines strengths of both models
- **Use Case**: Production deployment

### 3. Training Pipeline (`src/training/`)

```python
Pipeline Flow:
1. Load Data → 2. Preprocess → 3. Train Models → 4. Evaluate → 5. Save
```

#### Trainer (`trainer.py`)
- Orchestrates full training pipeline
- Manages model lifecycle
- Integrates with MLflow for experiment tracking

#### Evaluator (`evaluator.py`)
- Computes comprehensive metrics:
  - Precision, Recall, F1
  - Precision@K, Recall@K
  - ROC AUC, PR AUC
  - Confusion Matrix

### 4. Inference Layer (`src/inference/`)

#### Predictor
- Real-time prediction
- Batch prediction support
- Anomaly score calibration

#### Alert Service (`alert_service.py`)
- Risk-based alerting (low/medium/high/critical)
- Multi-channel notifications:
  - Email (SendGrid, AWS SES)
  - SMS (Twilio)
  - Slack/Teams webhooks
  - PagerDuty integration
- Alert logging and tracking

#### Human-in-the-Loop Service
- Review queue management
- Feedback collection
- Model performance tracking
- Continuous improvement loop

### 5. API Layer (`src/api/main.py`)

#### Endpoints

**Health & Status**
- `GET /` - Service information
- `GET /health` - Health check

**Predictions**
- `POST /predict` - Single transaction prediction
- `POST /predict/batch` - Batch predictions

**Model Management**
- `GET /models/info` - Model information
- `GET /stats` - Prediction statistics

### 6. Application Layer (`app/`)

#### Streamlit Dashboard
- **Single Prediction**: Interactive form with real-time analysis
- **Batch Analysis**: CSV upload and bulk processing
- **Model Performance**: Metrics visualization
- **Data Explorer**: Sample data generation and EDA

## Data Flow

### Real-Time Prediction Flow

```
1. Transaction arrives → API endpoint
2. Input validation → Pydantic models
3. Feature extraction → Preprocessor
4. Model inference → Ensemble
5. Risk assessment → Scoring
6. Alert generation → Alert Service (if needed)
7. Response returned → Client
8. Logging → Database
```

### Batch Processing Flow

```
1. CSV upload → Streamlit/API
2. Data validation → Batch preprocessor
3. Parallel inference → Model ensemble
4. Results aggregation → Statistics
5. Alert filtering → High-risk only
6. Report generation → CSV/Dashboard
7. Review queue → HITL service
```

## Production Deployment Strategy

### Infrastructure

```yaml
Production Stack:
  - Load Balancer: Nginx/AWS ALB
  - API Servers: 3+ FastAPI instances
  - Database: PostgreSQL (primary + replica)
  - Cache: Redis for predictions
  - Message Queue: RabbitMQ/Kafka for async processing
  - Model Store: S3/MinIO
  - Monitoring: Prometheus + Grafana
```

### Scalability Considerations

1. **Horizontal Scaling**
   - Stateless API design
   - Docker containers orchestrated by Kubernetes
   - Auto-scaling based on CPU/memory

2. **Model Serving**
   - Pre-loaded models in memory
   - Model versioning with MLflow
   - A/B testing capability
   - Canary deployments

3. **Database Optimization**
   - Connection pooling
   - Read replicas for analytics
   - Partitioning for large tables
   - Indexes on frequently queried columns

4. **Caching Strategy**
   - Redis for recent predictions
   - Feature cache for repeated users
   - Model cache for fast loading

### Deployment Process

```
1. Code Push → GitHub
2. CI Pipeline → Tests, Linting, Security Scan
3. Build Docker Images → Container Registry
4. Deploy to Staging → Integration Tests
5. Manual Approval → Production Gate
6. Blue-Green Deployment → Zero Downtime
7. Health Checks → Validation
8. Traffic Switch → Gradual Rollout
9. Monitoring → Alert on Issues
```

## Security Architecture

### Authentication & Authorization
- JWT tokens for API access
- API keys for service-to-service
- Role-based access control (RBAC)

### Data Protection
- Encryption at rest (database)
- Encryption in transit (TLS/SSL)
- PII masking in logs
- Secure credential management (AWS Secrets Manager)

### API Security
- Rate limiting (per user/IP)
- Input validation
- SQL injection prevention
- XSS protection

## Monitoring & Observability

### Metrics Collection

**System Metrics**
- CPU, Memory, Disk usage
- Network I/O
- Container health

**Application Metrics**
- Request rate, latency
- Error rate
- Prediction throughput
- Model inference time

**Business Metrics**
- Anomaly detection rate
- False positive rate
- Alert volume
- Review queue size

### Logging

```python
Log Levels:
- ERROR: System failures, critical issues
- WARNING: Anomalies detected, degraded performance
- INFO: Normal operations, predictions
- DEBUG: Detailed debugging information
```

**Log Aggregation**: ELK Stack (Elasticsearch, Logstash, Kibana)

### Alerting

**Critical Alerts**
- Service down
- Database connection failure
- Model loading error
- High error rate (>5%)

**Warning Alerts**
- High latency (>500ms)
- Unusual anomaly rate
- Queue backlog
- Memory pressure

## Model Updates & Retraining

### Continuous Learning Pipeline

```
1. Collect Feedback → Human reviews
2. Aggregate Data → Training dataset
3. Retrain Models → Scheduled/Triggered
4. Evaluate Performance → Validation set
5. Version Control → MLflow registry
6. Staging Deployment → Shadow mode
7. A/B Testing → Compare versions
8. Production Rollout → Gradual deployment
```

### Model Versioning

```
models/
├── isolation_forest_v1.0.0.joblib
├── isolation_forest_v1.1.0.joblib
├── autoencoder_v1.0.0.pth
├── autoencoder_v1.1.0.pth
└── ensemble_v1.0.0.joblib
```

## Disaster Recovery

### Backup Strategy
- Database: Daily backups, 30-day retention
- Models: Versioned in S3, immutable
- Logs: Retained for 90 days
- Configuration: Git-tracked

### Recovery Procedures
1. **Service Failure**: Auto-restart with Kubernetes
2. **Database Failure**: Failover to replica
3. **Model Corruption**: Rollback to previous version
4. **Complete Outage**: Multi-region deployment

## Performance Optimization

### Inference Optimization
- Model quantization for faster inference
- Batch prediction for efficiency
- Feature caching
- Model warm-up on startup

### Database Optimization
- Connection pooling
- Query optimization
- Indexing strategy
- Partitioning for large tables

### Network Optimization
- Content delivery network (CDN)
- API response compression
- Connection keep-alive
- Request batching

## Cost Optimization

### Compute
- Auto-scaling based on load
- Spot instances for batch jobs
- Reserved instances for base load

### Storage
- S3 lifecycle policies
- Compressed logs
- Archival storage for old data

### Monitoring
- Metric sampling for high-volume data
- Log filtering
- Dashboard query optimization

## Future Enhancements

1. **Real-time Streaming**: Kafka integration for event streaming
2. **Graph Analysis**: Network analysis for fraud rings
3. **Explainability**: SHAP values for predictions
4. **Online Learning**: Incremental model updates
5. **Multi-model Serving**: TensorFlow Serving, TorchServe
6. **Feature Store**: Centralized feature management
7. **AutoML**: Automated hyperparameter tuning