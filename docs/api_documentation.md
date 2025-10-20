# API Documentation

## Base URL

```
http://localhost:8000
```

Production: `https://api.yourdomain.com`

## Authentication

Currently, the API is open. For production deployment, implement JWT authentication:

```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  http://localhost:8000/predict
```

## Endpoints

### Health & Status

#### GET `/`

Get API information.

**Response:**
```json
{
  "message": "Transaction Anomaly Detection API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

#### GET `/health`

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2023-12-15T14:30:00"
}
```

**Status Codes:**
- `200`: Service is healthy
- `503`: Service is unhealthy (model not loaded)

---

### Predictions

#### POST `/predict`

Predict if a single transaction is anomalous.

**Request Body:**
```json
{
  "transaction_id": "TXN_00000001",
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
```

**Field Descriptions:**

| Field | Type | Description | Required | Constraints |
|-------|------|-------------|----------|-------------|
| `transaction_id` | string | Unique transaction identifier | Yes | - |
| `timestamp` | string | Transaction timestamp | Yes | ISO 8601 format |
| `amount` | float | Transaction amount | Yes | > 0 |
| `merchant_category` | string | Category of merchant | Yes | retail, food, transport, entertainment, utilities |
| `location_distance_km` | float | Distance from usual location (km) | Yes | >= 0 |
| `is_online` | int | 1 if online, 0 if in-person | Yes | 0 or 1 |
| `time_since_last_transaction_minutes` | float | Minutes since last transaction | Yes | >= 0 |
| `is_weekend` | int | 1 if weekend, 0 if weekday | Yes | 0 or 1 |
| `hour` | int | Hour of day (0-23) | Yes | 0-23 |
| `day_of_week` | int | Day of week (0=Monday) | Yes | 0-6 |
| `transaction_count_1h` | int | Number of transactions in last hour | Yes | >= 0 |
| `total_amount_24h` | float | Total amount spent in last 24h | Yes | >= 0 |
| `user_id` | int | User identifier | Yes | > 0 |

**Response:**
```json
{
  "transaction_id": "TXN_00000001",
  "is_anomaly": false,
  "anomaly_score": 0.234,
  "confidence": 0.532,
  "risk_level": "low",
  "timestamp": "2023-12-15T14:30:05"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `transaction_id` | string | Transaction identifier |
| `is_anomaly` | boolean | True if flagged as anomaly |
| `anomaly_score` | float | Anomaly probability (0-1) |
| `confidence` | float | Model confidence (0-1) |
| `risk_level` | string | low, medium, high, critical |
| `timestamp` | string | Prediction timestamp |

**Risk Levels:**
- `low`: score < 0.5
- `medium`: 0.5 <= score < 0.8
- `high`: 0.8 <= score < 0.9
- `critical`: score >= 0.9

**Status Codes:**
- `200`: Success
- `422`: Validation error
- `500`: Server error
- `503`: Model not loaded

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_00000001",
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

**Example (Python):**
```python
import requests

transaction = {
    "transaction_id": "TXN_00000001",
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

---

#### POST `/predict/batch`

Predict anomalies for multiple transactions in batch.

**Request Body:**
```json
{
  "transactions": [
    {
      "transaction_id": "TXN_00000001",
      "timestamp": "2023-12-15 14:30:00",
      "amount": 150.50,
      // ... other fields
    },
    {
      "transaction_id": "TXN_00000002",
      "timestamp": "2023-12-15 14:31:00",
      "amount": 75.25,
      // ... other fields
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "transaction_id": "TXN_00000001",
      "is_anomaly": false,
      "anomaly_score": 0.234,
      "confidence": 0.532,
      "risk_level": "low"
    },
    {
      "transaction_id": "TXN_00000002",
      "is_anomaly": true,
      "anomaly_score": 0.856,
      "confidence": 0.712,
      "risk_level": "high"
    }
  ],
  "total_transactions": 2,
  "anomalies_detected": 1,
  "timestamp": "2023-12-15T14:30:05"
}
```

**Batch Limits:**
- Maximum 1000 transactions per request
- Use pagination for larger datasets

**Example (Python):**
```python
import requests

batch_request = {
    "transactions": [
        {
            "transaction_id": f"TXN_{i:08d}",
            "timestamp": "2023-12-15 14:30:00",
            "amount": 100.0 + i,
            "merchant_category": "retail",
            "location_distance_km": 5.0,
            "is_online": 0,
            "time_since_last_transaction_minutes": 120.0,
            "is_weekend": 0,
            "hour": 14,
            "day_of_week": 3,
            "transaction_count_1h": 2,
            "total_amount_24h": 300.0,
            "user_id": 1234
        }
        for i in range(100)
    ]
}

response = requests.post(
    "http://localhost:8000/predict/batch",
    json=batch_request
)

result = response.json()
print(f"Processed {result['total_transactions']} transactions")
print(f"Found {result['anomalies_detected']} anomalies")
```

---

### Model Management

#### GET `/models/info`

Get information about loaded models.

**Response:**
```json
{
  "model_type": "Ensemble",
  "n_models": 2,
  "model_names": ["IsolationForestDetector", "AutoencoderDetector"],
  "feature_count": 10,
  "features": [
    "amount_log",
    "location_distance_km",
    "time_since_last_transaction_minutes",
    "hour",
    "day_of_week",
    "transaction_count_1h",
    "total_amount_24h",
    "is_online",
    "is_weekend",
    "merchant_category_encoded"
  ]
}
```

---

### Statistics

#### GET `/stats`

Get prediction statistics.

**Response:**
```json
{
  "total_predictions": 15234,
  "anomalies_detected": 762,
  "average_score": 0.124,
  "last_updated": "2023-12-15T14:30:00"
}
```

---

## Error Handling

All errors follow this format:

```json
{
  "detail": "Error message description"
}
```

### Common Error Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input data |
| 422 | Validation Error - Data doesn't match schema |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not loaded |

### Validation Error Example

**Request with invalid data:**
```json
{
  "transaction_id": "TXN_001",
  "amount": -50.0,  // Invalid: must be positive
  "hour": 25        // Invalid: must be 0-23
}
```

**Response (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "amount"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    },
    {
      "loc": ["body", "hour"],
      "msg": "ensure this value is less than or equal to 23",
      "type": "value_error.number.not_le"
    }
  ]
}
```

---

## Rate Limiting

Production API implements rate limiting:

- **Free Tier**: 100 requests/hour
- **Standard**: 1000 requests/hour
- **Enterprise**: Unlimited

Rate limit headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1609459200
```

---

## Webhooks

Configure webhooks to receive alerts for high-risk transactions.

**Webhook Payload:**
```json
{
  "event": "anomaly_detected",
  "timestamp": "2023-12-15T14:30:00",
  "transaction": {
    "transaction_id": "TXN_00000001",
    "is_anomaly": true,
    "anomaly_score": 0.923,
    "risk_level": "critical"
  }
}
```

**Configuration:**
```python
webhook_config = {
    "url": "https://your-webhook-endpoint.com/alerts",
    "events": ["anomaly_detected"],
    "risk_levels": ["high", "critical"]
}
```

---

## SDKs and Client Libraries

### Python SDK

```python
from anomaly_detection_client import AnomalyDetectionClient

client = AnomalyDetectionClient(
    base_url="http://localhost:8000",
    api_key="your_api_key"
)

# Single prediction
result = client.predict(transaction)

# Batch prediction
results = client.predict_batch(transactions)

# Stream prediction
for result in client.predict_stream(transaction_stream):
    if result.is_anomaly:
        print(f"Alert: {result.transaction_id}")
```

### JavaScript/Node.js

```javascript
const AnomalyDetectionClient = require('anomaly-detection-client');

const client = new AnomalyDetectionClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your_api_key'
});

// Single prediction
const result = await client.predict(transaction);

// Batch prediction
const results = await client.predictBatch(transactions);
```

---

## Best Practices

### 1. Batch Requests

For multiple transactions, use batch endpoint:
```python
# Good - Batch request
results = client.predict_batch(transactions)

# Bad - Multiple single requests
results = [client.predict(txn) for txn in transactions]
```

### 2. Error Handling

Always handle errors gracefully:
```python
try:
    result = client.predict(transaction)
except ValidationError as e:
    print(f"Validation error: {e}")
except ServiceUnavailable:
    print("Service temporarily unavailable")
    # Implement retry logic
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 3. Caching

Cache predictions for repeated transactions:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_prediction(transaction_id):
    return client.predict(transaction)
```

### 4. Timeouts

Set appropriate timeouts:
```python
client = AnomalyDetectionClient(
    base_url="http://localhost:8000",
    timeout=30  # 30 seconds
)
```

### 5. Monitoring

Monitor API usage:
```python
import time

start = time.time()
result = client.predict(transaction)
latency = time.time() - start

if latency > 1.0:
    logger.warning(f"High latency: {latency}s")
```

---

## Performance

### Response Times

| Endpoint | Average | P95 | P99 |
|----------|---------|-----|-----|
| `/predict` | 45ms | 120ms | 200ms |
| `/predict/batch` (100 txn) | 250ms | 500ms | 800ms |
| `/health` | 5ms | 10ms | 15ms |

### Throughput

- Single predictions: ~1000 requests/second
- Batch predictions: ~10,000 transactions/second

### Optimization Tips

1. **Use batch endpoint** for multiple predictions
2. **Enable caching** for repeated transactions
3. **Use connection pooling** for multiple requests
4. **Implement retry logic** with exponential backoff
5. **Monitor response times** and set alerts

---

## Migration Guide

### From v0.9 to v1.0

**Breaking Changes:**
- `anomaly_probability` renamed to `anomaly_score`
- `category` renamed to `merchant_category`
- New required field: `transaction_count_1h`

**Migration:**
```python
# Old (v0.9)
transaction = {
    "category": "retail",
    "anomaly_probability": 0.5
}

# New (v1.0)
transaction = {
    "merchant_category": "retail",
    "anomaly_score": 0.5,
    "transaction_count_1h": 2  # New field
}
```

---

## Support

- **Documentation**: https://docs.yourdomain.com
- **API Status**: https://status.yourdomain.com
- **Support Email**: support@yourdomain.com
- **GitHub Issues**: https://github.com/username/transaction-anomaly-detection/issues

---

## Changelog

### v1.0.0 (2023-12-15)
- Initial release
- Isolation Forest and Autoencoder models
- Ensemble predictions
- Batch processing support

### v1.1.0 (TBD)
- Streaming predictions
- Webhook support
- Enhanced explanations
- Model versioning API
---

**This is a test document for production.**