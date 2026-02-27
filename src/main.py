"""Entry point for the full GAVELbench pipeline.

Runs answer generation for all models defined in models.yaml, then evaluates
all generated answers against the Bob ground-truth data and writes a report.

Usage::

    uv run python src/main.py

Optional flags:
    --skip-generation   Skip answer generation and only run evaluation.
    --only-eval         Alias for --skip-generation.
"""

import argparse
import glob
import sys

from evaluate import evaluate
from gen_answers_from_llm import run_pipeline

BOB_PATH = "data/bob_data.jsonl"
MODELS_YAML = "src/models.yaml"
GENERATED_DIR = "data/generated"
REPORT_PATH = "data/results/evaluation_report.md"


def main(skip_generation: bool = False) -> None:
    """Runs the full pipeline: generation → evaluation → report.

    Args:
        skip_generation: If True, skips answer generation and goes straight
            to evaluation using whatever generated files already exist.
    """
    if not skip_generation:
        print("=== Step 1: Generating answers ===")
        run_pipeline(
            bob_path=BOB_PATH,
            models_yaml=MODELS_YAML,
            output_dir=GENERATED_DIR,
        )
    else:
        print("=== Step 1: Skipping generation ===")

    generated_files = sorted(glob.glob(f"{GENERATED_DIR}/generated_answers_*.jsonl"))
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
        "--skip-generation",
        "--only-eval",
        action="store_true",
        help="Skip answer generation and only run evaluation.",
    )
    args = parser.parse_args()
    main(skip_generation=args.skip_generation)
