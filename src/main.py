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
import asyncio
import glob
import sys

from evaluate import evaluate
from gen_answers_from_llm import run_pipeline
from utils.get_kunnskapsbank_data import fetch_bob_data, sample_by_overkategori

BOB_PATH = "data/bob_data.json"
KUNNSKAPSBASE_KATEGORIER_PATH = "data/kunnskapsbase_kategorier.json"
MODELS_YAML = "src/models.yaml"
GENERATED_DIR = "data/generated"
REPORT_PATH = "data/results/evaluation_report.md"
KATEGORI_MAPPING_PATH = "data/kategorier_mapping.json"
N_PER_OVERKATEGORI = 10


async def main(
    skip_bq_fetch: bool = False,
    skip_generation: bool = False,
    n_per_overkategori: int = N_PER_OVERKATEGORI,
) -> None:
    """Runs the full pipeline: fetch → sample → generation → evaluation → report.

    Args:
        skip_bq_fetch: If True, skips the BigQuery fetch but still re-samples
            from the existing ``kunnskapsbase_kategorier.json``.
        skip_generation: If True, skips answer generation and goes straight
            to evaluation using whatever generated files already exist.
            Implies skip_bq_fetch and skips re-sampling (reuses bob_data.json).
        n_per_overkategori: Number of Q&A pairs to sample per overarching category.
    """
    if skip_generation:
        skip_bq_fetch = True

    if not skip_generation:
        if not skip_bq_fetch:
            print("=== Step 0: Fetching Bob data from BigQuery ===")
            fetch_bob_data(KUNNSKAPSBASE_KATEGORIER_PATH)
        else:
            print(
                "=== Step 0: Skipping BQ fetch, using existing kunnskapsbase_kategorier.json ==="
            )

        print(f"\n=== Step 0b: Sampling {n_per_overkategori} Q&As per overkategori ===")
        sampled = sample_by_overkategori(
            KUNNSKAPSBASE_KATEGORIER_PATH,
            KATEGORI_MAPPING_PATH,
            n_per_overkategori,
        )
        import json as _json

        with open(BOB_PATH, "w", encoding="utf-8") as f:
            _json.dump(sampled, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {len(sampled)} sampled Q&A pairs to {BOB_PATH}")
    else:
        print(
            "=== Step 0: Skipping fetch and sampling, reusing existing bob_data.json ==="
        )

    if not skip_generation:
        print("\n=== Step 1: Generating answers ===")
        await run_pipeline(
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

    # Filter to only files for models currently active in models.yaml
    import yaml as _yaml

    with open(MODELS_YAML, encoding="utf-8") as _f:
        _spec = _yaml.safe_load(_f)
    _active_ids = {m["id"].replace("/", "__") for m in _spec.get("models", [])}
    generated_files = [
        p
        for p in generated_files
        if any(p.endswith(f"generated_answers_{mid}.json") for mid in _active_ids)
    ]
    if not generated_files:
        print(
            f"No generated answer files match active models in {MODELS_YAML}. Aborting."
        )
        sys.exit(1)
    print(
        f"Evaluating {len(generated_files)} model(s): {[p.split('/')[-1] for p in generated_files]}"
    )

    print("\n=== Step 2: Evaluating answers ===")
    evaluate(
        bob_path=BOB_PATH,
        generated_paths=generated_files,
        output_path=REPORT_PATH,
        kategori_mapping_path=KATEGORI_MAPPING_PATH,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GAVELbench pipeline.")
    parser.add_argument(
        "--skip-bq-fetch",
        action="store_true",
        help=(
            "Skip BigQuery fetch but still re-sample from the existing"
            " kunnskapsbase_kategorier.json."
        ),
    )
    parser.add_argument(
        "--skip-generation",
        "--only-eval",
        action="store_true",
        help=(
            "Skip fetch, sampling, and generation — only re-evaluate"
            " existing generated answers against existing bob_data.json."
        ),
    )
    parser.add_argument(
        "--n-per-overkategori",
        type=int,
        default=N_PER_OVERKATEGORI,
        help=(
            f"Number of Q&A pairs to sample per overarching category"
            f" (default: {N_PER_OVERKATEGORI})."
        ),
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            skip_bq_fetch=args.skip_bq_fetch,
            skip_generation=args.skip_generation,
            n_per_overkategori=args.n_per_overkategori,
        )
    )
