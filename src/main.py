"""Entry point for the full GAVELbench pipeline.

Runs the full pipeline:
  0. Fetch Bob Q&A data from BigQuery
  1. Generate answers for all models defined in models.yaml
  2. Evaluate generated answers
  3. Write report artifacts (JSON, PNGs, static QMD files)

Usage::

    uv run python src/main.py

Optional flags:
    --skip-fetch        Skip BQ fetch (reuse existing kunnskapsbase_kategorier.json).
    --skip-generation   Skip answer generation and only run evaluation + report.
    --only-eval         Alias for --skip-generation (implies --skip-fetch).
    --only-report       Skip all pipeline steps; only regenerate report artifacts
                        from existing JSON results in data/results/.
"""

import argparse
import asyncio
import glob
import json as _json
import os
import sys

from evaluate import evaluate
from fetch_data import fetch_bob_data, sample_by_overkategori
from generate import run_pipeline
from models import active_model_ids
from report import print_report, regenerate_report, write_report

BOB_PATH = "data/bob_data.json"
KUNNSKAPSBASE_KATEGORIER_PATH = "data/kunnskapsbase_kategorier.json"
MODELS_YAML = "src/models.yaml"
GENERATED_DIR = "data/generated"
REPORT_PATH = "data/results/evaluation_report.json"
RESULTS_JSON_PATH = "data/results/evaluation_report.json"
OVK_JSON_PATH = "data/results/evaluation_report_scores_per_overkategori.json"
KATEGORI_MAPPING_PATH = "data/kategorier_mapping.json"
N_PER_OVERKATEGORI = 10


async def main(
    skip_bq_fetch: bool = False,
    skip_generation: bool = False,
    only_report: bool = False,
    n_per_overkategori: int = N_PER_OVERKATEGORI,
) -> None:
    """Runs the full pipeline: fetch → sample → generate → evaluate → report.

    Args:
        skip_bq_fetch: If True, skips the BigQuery fetch but still re-samples
            from the existing ``kunnskapsbase_kategorier.json``.
        skip_generation: If True, skips answer generation and goes straight
            to evaluation using whatever generated files already exist.
            Implies skip_bq_fetch and skips re-sampling (reuses bob_data.json).
        only_report: If True, skips all pipeline steps and only regenerates
            report artifacts from the existing JSON result files.
        n_per_overkategori: Number of Q&A pairs to sample per overarching category.
    """
    if only_report:
        print("=== Only regenerating report from existing results ===")
        if not os.path.exists(RESULTS_JSON_PATH):
            print(
                f"No results file found at {RESULTS_JSON_PATH}. Run the full pipeline first."
            )
            sys.exit(1)
        regenerate_report(
            results_json_path=RESULTS_JSON_PATH,
            output_path=REPORT_PATH,
            ovk_json_path=OVK_JSON_PATH,
            models_yaml_path=MODELS_YAML,
        )
        return

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
    _active_ids = active_model_ids(MODELS_YAML)
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
    results, n_pairs, scores_ovk, n_pairs_ovk = evaluate(
        bob_path=BOB_PATH,
        generated_paths=generated_files,
    )

    print("\n=== Step 3: Writing report ===")
    print_report(results, n_pairs)
    write_report(
        results,
        n_pairs,
        REPORT_PATH,
        scores_by_overkategori=scores_ovk or None,
        n_pairs_by_overkategori=n_pairs_ovk or None,
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
        "--only-report",
        action="store_true",
        help=(
            "Skip all pipeline steps — only regenerate report artifacts"
            " from the existing JSON results in data/results/."
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
            only_report=args.only_report,
            n_per_overkategori=args.n_per_overkategori,
        )
    )
