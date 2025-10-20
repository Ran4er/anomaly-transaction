"""
Database module for storing transactions and predictions
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional
from loguru import logger

Base = declarative_base()


class Transaction(Base):
    """Transaction model"""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    amount = Column(Float)
    merchant_category = Column(String)
    location_distance_km = Column(Float)
    is_online = Column(Boolean)
    user_id = Column(Integer, index=True)
    is_anomaly = Column(Boolean, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    risk_level = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Database connection manager"""

    def __init__(self, database_url: str = "sqlite:///./transactions.db"):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Database initialized: {database_url}")

    def get_session(self):
        """Get database session"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def save_prediction(self, transaction_data: dict, prediction_result: dict):
        """Save transaction and prediction to database"""
        db = self.SessionLocal()
        try:
            transaction = Transaction(
                transaction_id=transaction_data.get("transaction_id"),
                amount=transaction_data.get("amount"),
                merchant_category=transaction_data.get("merchant_category"),
                user_id=transaction_data.get("user_id"),
                is_anomaly=prediction_result.get("is_anomaly"),
                anomaly_score=prediction_result.get("anomaly_score"),
                risk_level=prediction_result.get("risk_level"),
            )
            db.add(transaction)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            db.rollback()
        finally:
            db.close()