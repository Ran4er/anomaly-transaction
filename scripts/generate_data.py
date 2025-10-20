"""
Script to generate synthetic transaction data
"""
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.data.generator import TransactionGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50000,
        help="Total number of transactions to generate"
    )
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=0.05,
        help="Ratio of anomalies (0-1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    logger.add(
        "logs/data_generation_{time}.log",
        rotation="100 MB",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("Starting data generation")
    logger.info("=" * 80)
    logger.info(f"Total samples: {args.n_samples}")
    logger.info(f"Anomaly ratio: {args.anomaly_ratio * 100}%")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generator = TransactionGenerator(seed=args.seed)
        
        train_df, val_df, test_df = generator.generate_dataset(
            n_samples=args.n_samples,
            anomaly_ratio=args.anomaly_ratio
        )
        
        train_df.to_csv(output_path / "train.csv", index=False)
        val_df.to_csv(output_path / "val.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        
        logger.info("\n" + "=" * 80)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 80)
        logger.info(f"\nTraining set:")
        logger.info(f"  Total: {len(train_df)}")
        logger.info(f"  Anomalies: {train_df['is_anomaly'].sum()} ({train_df['is_anomaly'].mean()*100:.2f}%)")
        
        logger.info(f"\nValidation set:")
        logger.info(f"  Total: {len(val_df)}")
        logger.info(f"  Anomalies: {val_df['is_anomaly'].sum()} ({val_df['is_anomaly'].mean()*100:.2f}%)")
        
        logger.info(f"\nTest set:")
        logger.info(f"  Total: {len(test_df)}")
        logger.info(f"  Anomalies: {test_df['is_anomaly'].sum()} ({test_df['is_anomaly'].mean()*100:.2f}%)")
        
        logger.info("\n" + "=" * 80)
        logger.info("Data generation completed successfully!")
        logger.info(f"Files saved to: {output_path}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        raise


if __name__ == "__main__":
    main()