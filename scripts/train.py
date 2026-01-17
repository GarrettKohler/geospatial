#!/usr/bin/env python3
"""CLI entry point for model training."""

import argparse
import sys
from datetime import datetime

from dooh_ml.config import Config, load_config_from_env
from dooh_ml.training.pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train DOOH site optimization models"
    )
    parser.add_argument(
        "--train-end",
        required=True,
        help="Last date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--validation-end",
        required=True,
        help="Last date for validation data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--test-start",
        help="First date for test data (YYYY-MM-DD). Default: validation-end + gap",
    )
    parser.add_argument(
        "--experiment",
        default="dooh-site-optimization",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--gap-days",
        type=int,
        default=14,
        help="Gap days between train/val and val/test splits",
    )

    args = parser.parse_args()

    # Load configuration from environment
    config = load_config_from_env()
    config.gap_days = args.gap_days

    print(f"Starting training pipeline...")
    print(f"  Train end: {args.train_end}")
    print(f"  Validation end: {args.validation_end}")
    print(f"  Test start: {args.test_start or 'auto'}")
    print(f"  Gap days: {args.gap_days}")
    print()

    # Run pipeline
    pipeline = TrainingPipeline(config)

    try:
        result = pipeline.run(
            train_end=args.train_end,
            validation_end=args.validation_end,
            test_start=args.test_start,
            experiment_name=args.experiment,
        )

        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print(f"Run ID: {result.run_id}")
        print()
        print("Metrics:")
        for model_name, metrics in result.metrics.items():
            print(f"\n  {model_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {metric_name}: {value:.4f}")
                else:
                    print(f"    {metric_name}: {value}")

    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
