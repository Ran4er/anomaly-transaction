"""
Alert service for anomaly notifications
"""
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger
import json
from pathlib import Path


class AlertService:
    """Service for sending alerts when anomalies are detected"""
    
    def __init__(self, config_path: Optional[str] = "configs/alert_config.yaml"):
        """Initialize alert service"""
        self.alerts_log = []
        self.alert_threshold = 0.7
        self.alert_dir = Path("data/alerts")
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Alert service initialized")
    
    async def send_alert(
        self,
        transaction: Dict[str, Any],
        anomaly_score: float
    ):
        """
        Send alert for anomalous transaction
        
        Args:
            transaction: Transaction data
            anomaly_score: Anomaly score (0-1)
        """
        if anomaly_score < self.alert_threshold:
            return
        
        alert = {
            "timestamp": datetime.now().isoformat(),
            "transaction_id": transaction.get("transaction_id"),
            "user_id": transaction.get("user_id"),
            "amount": transaction.get("amount"),
            "anomaly_score": float(anomaly_score),
            "risk_level": self._get_risk_level(anomaly_score),
            "merchant_category": transaction.get("merchant_category"),
            "location_distance_km": transaction.get("location_distance_km"),
            "action_required": True
        }
        
        self.alerts_log.append(alert)
        logger.warning(
            f"ALERT: Anomaly detected - Transaction {alert['transaction_id']}, "
            f"Score: {anomaly_score:.3f}, Risk: {alert['risk_level']}"
        )
        
        self._save_alert(alert)
        
        # In production, integrate with:
        # - Email service (SendGrid, AWS SES)
        # - SMS service (Twilio)
        # - Slack/Teams webhooks
        # - Incident management (PagerDuty, Opsgenie)
        # - Ticketing system (Jira, ServiceNow)
        
        return alert
    
    def _get_risk_level(self, score: float) -> str:
        """Determine risk level from score"""
        if score >= 0.9:
            return "critical"
        elif score >= 0.8:
            return "high"
        elif score >= 0.7:
            return "medium"
        else:
            return "low"
    
    def _save_alert(self, alert: Dict[str, Any]):
        """Save alert to file"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        alert_file = self.alert_dir / f"alerts_{date_str}.jsonl"
        
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert) + '\n')
    
    def get_recent_alerts(self, limit: int = 10) -> list:
        """Get recent alerts"""
        return self.alerts_log[-limit:]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        if not self.alerts_log:
            return {
                "total_alerts": 0,
                "by_risk_level": {},
                "average_score": 0.0
            }
        
        risk_levels = {}
        total_score = 0
        
        for alert in self.alerts_log:
            risk = alert['risk_level']
            risk_levels[risk] = risk_levels.get(risk, 0) + 1
            total_score += alert['anomaly_score']
        
        return {
            "total_alerts": len(self.alerts_log),
            "by_risk_level": risk_levels,
            "average_score": total_score / len(self.alerts_log)
        }


class HumanInTheLoopService:
    """Service for human review of flagged transactions"""
    
    def __init__(self):
        self.review_queue = []
        self.reviewed_transactions = {}
        
    def add_to_review_queue(
        self,
        transaction_id: str,
        transaction_data: Dict[str, Any],
        anomaly_score: float,
        model_prediction: int
    ):
        """Add transaction to human review queue"""
        
        review_item = {
            "transaction_id": transaction_id,
            "added_at": datetime.now().isoformat(),
            "transaction_data": transaction_data,
            "anomaly_score": anomaly_score,
            "model_prediction": model_prediction,
            "status": "pending",
            "reviewed_by": None,
            "reviewed_at": None,
            "human_label": None,
            "notes": ""
        }
        
        self.review_queue.append(review_item)
        logger.info(f"Added transaction {transaction_id} to review queue")
        
        return review_item
    
    def submit_review(
        self,
        transaction_id: str,
        human_label: int,
        reviewer_id: str,
        notes: str = ""
    ):
        """Submit human review for a transaction"""
        
        for item in self.review_queue:
            if item["transaction_id"] == transaction_id:
                item["status"] = "reviewed"
                item["reviewed_by"] = reviewer_id
                item["reviewed_at"] = datetime.now().isoformat()
                item["human_label"] = human_label
                item["notes"] = notes
                
                self.reviewed_transactions[transaction_id] = item
                
                logger.info(
                    f"Transaction {transaction_id} reviewed by {reviewer_id}. "
                    f"Label: {human_label}"
                )
                
                return item
        
        raise ValueError(f"Transaction {transaction_id} not found in review queue")
    
    def get_review_metrics(self) -> Dict[str, Any]:
        """Calculate metrics on reviewed transactions"""
        
        if not self.reviewed_transactions:
            return {
                "total_reviewed": 0,
                "model_accuracy": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0
            }
        
        correct = 0
        false_positives = 0
        false_negatives = 0
        
        for txn in self.reviewed_transactions.values():
            model_pred = txn["model_prediction"]
            human_label = txn["human_label"]
            
            if model_pred == human_label:
                correct += 1
            elif model_pred == 1 and human_label == 0:
                false_positives += 1
            elif model_pred == 0 and human_label == 1:
                false_negatives += 1
        
        total = len(self.reviewed_transactions)
        
        return {
            "total_reviewed": total,
            "model_accuracy": correct / total if total > 0 else 0,
            "false_positive_rate": false_positives / total if total > 0 else 0,
            "false_negative_rate": false_negatives / total if total > 0 else 0,
            "agreement_rate": correct / total if total > 0 else 0
        }
    
    def get_pending_reviews(self, limit: Optional[int] = None) -> list:
        """Get pending reviews"""
        pending = [item for item in self.review_queue if item["status"] == "pending"]
        
        if limit:
            return pending[:limit]
        
        return pending