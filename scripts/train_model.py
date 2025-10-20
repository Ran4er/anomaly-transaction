"""
Script to train anomaly detection models
"""
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.training.trainer import AnomalyDetectionTrainer


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/raw/train.csv",
        help="Path to training data"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/raw/val.csv",
        help="Path to validation data"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/raw/test.csv",
        help="Path to test data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Directory to save trained models"
    )
    
    args = parser.parse_args()
    
    logger.add(
        "logs/training_{time}.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("Starting model training pipeline")
    logger.info("=" * 80)
    
    try:
        trainer = AnomalyDetectionTrainer(config_path=args.config)
        
        data = trainer.load_data(
            train_path=args.train_data,
            val_path=args.val_data,
            test_path=args.test_data
        )
        
        results = trainer.train_all(data)
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 80)
        
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
            logger.info(f"  F1 Score: {metrics.get('f1', 0):.4f}")
            logger.info(f"  Precision@5%: {metrics.get('precision_at_5', 0):.4f}")
            logger.info(f"  Recall@5%: {metrics.get('recall_at_5', 0):.4f}")
        
        trainer.save_models(output_dir=args.output_dir)
        
        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Models saved to: {args.output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()