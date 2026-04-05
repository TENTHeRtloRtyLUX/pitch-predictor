import argparse
import json
from pathlib import Path

from training_data_pipeline import prepare_retraining_artifacts


def parse_seasons(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="Build shared retraining artifacts for parallel model jobs.")
    parser.add_argument("--seasons", default="2023,2024,2025", help="Comma-separated seasons.")
    parser.add_argument("--output-dir", default="output/retrain_shared")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-sequence-length", type=int, default=12)
    args = parser.parse_args()

    summary = prepare_retraining_artifacts(
        seasons=parse_seasons(args.seasons),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
