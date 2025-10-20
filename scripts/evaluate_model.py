"""
Script to evaluate trained models on test set
"""
import sys
from pathlib import Path
import argparse
import json

sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt

from src.training.trainer import AnomalyDetectionTrainer
from src.training.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection models")
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/raw/test.csv",
        help="Path to test data"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="data/models",
        help="Directory with trained models"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save evaluation plots"
    )
    
    args = parser.parse_args()
    
    logger.add(
        "logs/evaluation_{time}.log",
        rotation="100 MB",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("Starting model evaluation")
    logger.info("=" * 80)
    
    try:
        logger.info(f"Loading test data from {args.test_data}")
        test_df = pd.read_csv(args.test_data)
        
        trainer = AnomalyDetectionTrainer()
        trainer.load_models(args.model_dir)
        
        test_features = test_df.drop(
            columns=['is_anomaly', 'anomaly_type', 'transaction_id', 'timestamp', 'user_id'],
            errors='ignore'
        )
        X_test = trainer.preprocessor.transform(test_features)
        y_test = test_df['is_anomaly'].values
        
        evaluator = ModelEvaluator()
        
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        
        for model_name, model in trainer.models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {model_name.upper()}")
            logger.info(f"{'='*60}")
            
            y_pred = model.predict(X_test)
            y_scores = model.predict_proba(X_test)
            
            metrics = evaluator.evaluate(y_test, y_pred, y_scores)
            all_results[model_name] = metrics
            
            evaluator.print_evaluation_report(metrics)
            
            if args.save_plots:
                fig = evaluator.plot_confusion_matrix(
                    y_test, y_pred,
                    title=f"Confusion Matrix - {model_name}"
                )
                fig.savefig(output_path / f"{model_name}_confusion_matrix.png")
                plt.close()
                
                fig = evaluator.plot_precision_recall_curve(
                    y_test, y_scores,
                    title=f"Precision-Recall Curve - {model_name}"
                )
                fig.savefig(output_path / f"{model_name}_pr_curve.png")
                plt.close()
        
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 80)
        
        comparison_df = pd.DataFrame({
            model_name: {
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'Precision@5%': metrics.get('precision_at_5', 0),
                'Recall@5%': metrics.get('recall_at_5', 0),
                'ROC AUC': metrics.get('roc_auc', 0),
                'PR AUC': metrics.get('pr_auc', 0)
            }
            for model_name, metrics in all_results.items()
        }).T
        
        logger.info("\n" + comparison_df.to_string())
        
        best_model = comparison_df['F1'].idxmax()
        logger.info(f"\nBest model: {best_model} (F1: {comparison_df.loc[best_model, 'F1']:.4f})")
        
        results_file = output_path / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        comparison_df.to_csv(output_path / "model_comparison.csv")
        
        logger.info(f"\nResults saved to {output_path}")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("SUMMARY FOR CI/CD")
        print("=" * 80)
        print(f"Best Model: {best_model}")
        print(f"F1 Score: {comparison_df.loc[best_model, 'F1']:.4f}")
        print(f"Precision@5%: {comparison_df.loc[best_model, 'Precision@5%']:.4f}")
        print(f"Recall@5%: {comparison_df.loc[best_model, 'Recall@5%']:.4f}")
        print(f"ROC AUC: {comparison_df.loc[best_model, 'ROC AUC']:.4f}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()