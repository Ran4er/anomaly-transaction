"""
Utility functions for Streamlit app
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level"""
    colors = {"low": "green", "medium": "orange", "high": "red", "critical": "darkred"}
    return colors.get(risk_level, "gray")


def create_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Create summary statistics from dataframe"""
    stats = {
        "total_transactions": len(df),
        "total_amount": df["amount"].sum() if "amount" in df.columns else 0,
        "avg_amount": df["amount"].mean() if "amount" in df.columns else 0,
        "anomalies": df["is_anomaly"].sum() if "is_anomaly" in df.columns else 0,
    }
    return stats


def prepare_chart_data(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Prepare data for charts"""
    return df.groupby(group_by).size().reset_index(name="count")