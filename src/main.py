"""Entry point for the full GAVELbench pipeline.

Runs the full pipeline:
  0. Fetch Bob Q&A data from BigQuery
  1. Generate answers for all models defined in models.yaml
  2. Evaluate generated answers and write a report

Usage::

    uv run python src/main.py

Optional flags:
    --skip-fetch        Skip BQ fetch (reuse existing bob_data.json).
    --skip-generation   Skip answer generation and only run evaluation.
    --only-eval         Alias for --skip-generation (implies --skip-fetch).
"""

import argparse
import glob
import sys

from evaluate import evaluate
from gen_answers_from_llm import run_pipeline
from load_data_from_BQ import fetch_bob_data

BOB_PATH = "data/bob_data.json"
MODELS_YAML = "src/models.yaml"
GENERATED_DIR = "data/generated"
REPORT_PATH = "data/results/evaluation_report.md"


def main(skip_fetch: bool = False, skip_generation: bool = False) -> None:
    """Runs the full pipeline: fetch → generation → evaluation → report.

    Args:
        skip_fetch: If True, skips the BigQuery fetch and reuses the existing
            bob_data.json file.
        skip_generation: If True, skips answer generation and goes straight
            to evaluation using whatever generated files already exist.
            Implies skip_fetch.
    """
    if skip_generation:
        skip_fetch = True

    if not skip_fetch:
        print("=== Step 0: Fetching Bob data from BigQuery ===")
        fetch_bob_data(BOB_PATH)
    else:
        print("=== Step 0: Skipping BQ fetch ===")

    if not skip_generation:
        print("\n=== Step 1: Generating answers ===")
        run_pipeline(
            bob_path=BOB_PATH,
            models_yaml=MODELS_YAML,
            output_dir=GENERATED_DIR,
        )
    else:
        print("=== Step 1: Skipping generation ===")

    generated_files = sorted(glob.glob(f"{GENERATED_DIR}/generated_answers_*.json"))
    if not generated_files:
        print(f"No generated answer files found in {GENERATED_DIR}/. Aborting.")
        sys.exit(1)

    print("\n=== Step 2: Evaluating answers ===")
    evaluate(
        bob_path=BOB_PATH,
        generated_paths=generated_files,
        output_path=REPORT_PATH,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GAVELbench pipeline.")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip BigQuery fetch and reuse existing bob_data.json.",
    )
    parser.add_argument(
        "--skip-generation",
        "--only-eval",
        action="store_true",
        help="Skip answer generation and only run evaluation (implies --skip-fetch).",
    )
    args = parser.parse_args()
    main(skip_fetch=args.skip_fetch, skip_generation=args.skip_generation)
