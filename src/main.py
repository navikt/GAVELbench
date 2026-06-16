"""Entry point for the full GAVELbench pipeline.

Runs the full pipeline:
  0. Fetch Bob Q&A data from BigQuery
  1. Generate answers for all models defined in models.yaml
  2. Evaluate generated answers
  3. Write report artifacts (JSON, PNGs, static QMD files)

Bucket sync (enabled by default, opt-out with --skip-bucket):
  After each stage, the produced JSON files are uploaded to the configured
  GCS bucket.  When skipping an upstream stage, the required files are
  downloaded from the bucket so that a local copy is not required.

Usage::

    uv run python src/main.py

Optional flags:
    --skip-bq-fetch     Skip BQ fetch (reuse existing / bucket copy of
                        kunnskapsbase_kategorier.json).
    --skip-generation   Skip answer generation and only run evaluation + report.
    --only-eval         Alias for --skip-generation (implies --skip-fetch).
    --only-report       Skip all pipeline steps; only regenerate report artifacts
                        from existing JSON results in data/results/.
    --skip-bucket       Disable all GCS bucket upload/download steps.
"""

import argparse
import asyncio
import glob
import json as _json
import os
import sys

from evaluate import evaluate
from fetch_data import fetch_bob_data, sample_by_overkategori
from generate import answered_questions, run_pipeline
from models import active_model_ids
from report import print_report, regenerate_report, write_report
from storage import download_dir, download_file, upload_dir, upload_file

BOB_PATH = "data/bob_data.json"
KUNNSKAPSBASE_KATEGORIER_PATH = "data/kunnskapsbase_kategorier.json"
MODELS_YAML = "src/models.yaml"
GENERATED_DIR = "data/generated"
REPORT_PATH = "data/results/evaluation_report.json"
RESULTS_JSON_PATH = "data/results/evaluation_report.json"
OVK_JSON_PATH = "data/results/evaluation_report_scores_per_overkategori.json"
KATEGORI_MAPPING_PATH = "data/kategorier_mapping.json"
N_PER_OVERKATEGORI = 10


def _sync_generated_from_bucket() -> None:
    """Downloads each active model's generated-answers file from the bucket.

    The generated-answer files are the source of truth for which questions have
    already been answered, so they are pulled before sampling to keep the local
    copy in sync with the bucket.
    """
    for safe_id in active_model_ids(MODELS_YAML):
        blob = f"{GENERATED_DIR}/generated_answers_{safe_id}.json"
        download_file(blob, blob)


async def main(
    skip_bq_fetch: bool = False,
    skip_generation: bool = False,
    only_report: bool = False,
    skip_bucket: bool = False,
    n_per_overkategori: int = N_PER_OVERKATEGORI,
) -> None:
    """Runs the full pipeline: fetch → sample → generate → evaluate → report.

    Args:
        skip_bq_fetch: If True, skips the BigQuery fetch; tries to pull
            ``kunnskapsbase_kategorier.json`` from the bucket before falling
            back to the existing local copy.
        skip_generation: If True, skips answer generation and goes straight
            to evaluation using whatever generated files already exist (pulled
            from the bucket if not present locally).
            Implies skip_bq_fetch and skips re-sampling (reuses bob_data.json).
        only_report: If True, skips all pipeline steps and only regenerates
            report artifacts from the existing JSON result files.
        skip_bucket: If True, disables all GCS bucket upload/download steps.
        n_per_overkategori: Number of Q&A pairs to sample per overarching category.
    """
    if only_report:
        print("=== Only regenerating report from existing results ===")
        if not os.path.exists(RESULTS_JSON_PATH):
            if not skip_bucket:
                print("  Trying to fetch results from bucket …")
                download_file(RESULTS_JSON_PATH, RESULTS_JSON_PATH)
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
            if not skip_bucket:
                print("  Uploading to bucket …")
                upload_file(
                    KUNNSKAPSBASE_KATEGORIER_PATH, KUNNSKAPSBASE_KATEGORIER_PATH
                )
        else:
            print("=== Step 0: Skipping BQ fetch ===")
            if not skip_bucket and not os.path.exists(KUNNSKAPSBASE_KATEGORIER_PATH):
                print("  Local file missing — downloading from bucket …")
                if not download_file(
                    KUNNSKAPSBASE_KATEGORIER_PATH, KUNNSKAPSBASE_KATEGORIER_PATH
                ):
                    print(
                        f"  Not found in bucket either: {KUNNSKAPSBASE_KATEGORIER_PATH}"
                    )
                    sys.exit(1)

        print(f"\n=== Step 0b: Sampling {n_per_overkategori} Q&As per overkategori ===")
        # The generated-answer files are the source of truth for already-answered
        # questions, so sync them from the bucket before computing the exclusion
        # set. The local copy is assumed to mirror the bucket.
        if not skip_bucket:
            print("  Syncing existing answers from bucket (source of truth) …")
            _sync_generated_from_bucket()
        answered = answered_questions(GENERATED_DIR, active_model_ids(MODELS_YAML))
        print(
            f"  {len(answered)} question(s) already answered by all active models"
            " — excluding them from sampling."
        )
        sampled = sample_by_overkategori(
            KUNNSKAPSBASE_KATEGORIER_PATH,
            KATEGORI_MAPPING_PATH,
            n_per_overkategori,
            exclude_questions=answered,
        )
        if not sampled:
            print(
                "  No unanswered questions left to sample — the pool is exhausted."
                " Nothing to generate."
            )
            sys.exit(0)
        with open(BOB_PATH, "w", encoding="utf-8") as f:
            _json.dump(sampled, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {len(sampled)} sampled Q&A pairs to {BOB_PATH}")
        if not skip_bucket:
            print("  Uploading to bucket …")
            upload_file(BOB_PATH, BOB_PATH)
    else:
        print("=== Step 0: Skipping fetch and sampling ===")
        if not skip_bucket and not os.path.exists(BOB_PATH):
            print("  Local bob_data.json missing — downloading from bucket …")
            if not download_file(BOB_PATH, BOB_PATH):
                print(f"  Not found in bucket either: {BOB_PATH}")
                sys.exit(1)

    if not skip_generation:
        print("\n=== Step 1: Generating answers ===")
        # Generated files were already synced from the bucket before sampling.
        await run_pipeline(
            bob_path=BOB_PATH,
            models_yaml=MODELS_YAML,
            output_dir=GENERATED_DIR,
        )
        if not skip_bucket:
            print("  Uploading generated answers to bucket …")
            upload_dir(GENERATED_DIR, GENERATED_DIR)
    else:
        print("=== Step 1: Skipping generation ===")
        if not skip_bucket:
            print("  Downloading generated answers from bucket …")
            download_dir(GENERATED_DIR, GENERATED_DIR)
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
    if not skip_bucket:
        print("  Uploading results to bucket …")
        upload_dir("data/results", "data/results")


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
        "--skip-bucket",
        action="store_true",
        help="Disable all GCS bucket upload/download steps.",
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
            skip_bucket=args.skip_bucket,
            n_per_overkategori=args.n_per_overkategori,
        )
    )
