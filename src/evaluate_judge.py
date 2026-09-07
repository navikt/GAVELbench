"""evaluate_judge: run judge-based metric evaluations.

This module is responsible for evaluating the performance of a judge model
using a specified evaluation dataset. It includes functions to load the
evaluation dataset, compute mean results from evaluation metrics, and run
the factual correction evaluation process asynchronously.

Key functionalities:
- Load evaluation datasets from various sources.
- Compute mean results for each metric overall and per overkategori.
- Run evaluations using the Google GenAI client.
- Save detailed results grouped by overkategori.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import pandas
from datasets import Dataset
from google import genai

# Local imports
from storage import _load_cfg

try:
    from eval_judge import claim_decom_fc_bidirectional
    from eval_judge.DataFetcher import DataFetcher
    from eval_judge.DataManager import DataSourceManager


except ImportError:
    print("Warning: Local modules not found.")

# ==================== Configuration ====================
CONFIG_PATH = "src/config.yml"
cfg = _load_cfg(CONFIG_PATH)
RESULTS_DIR = "./data/results_judge"
UPLOAD_TO_STORAGE = True  # upload results to cloud storage

_PROJECT_ID = cfg["project"]
_LOCATION = cfg["location"]
_JUDGE_MODEL = "gemini-2.5-pro"
_EVAL_DATASET = "claude-sonnet-4-6"
_SAMPLE_SIZE = 100  # Increased default for stratification
_BATCH_SIZE = 10
_MIN_SAMPLES_PER_OVERKATEGORI = (
    50  # Minimum samples required to compute metrics for an overkategori
)


def load_evaluation_dataset(
    model: str = _EVAL_DATASET,
    sample_size: int = _SAMPLE_SIZE,
    min_per_overkategori: int = _MIN_SAMPLES_PER_OVERKATEGORI,
    categories_key: str = "categories",
    mapping_key: str = "category_mapping",
) -> Dataset:
    """Loads and prepares the evaluation dataset with stratified overkategori sampling.

    Ensures each overkategori has at least min_per_overategori samples before
    filling the remainder randomly.

    Args:
        model (str): The name of the model to fetch the dataset for.
        sample_size (int): Total number of samples to load.
        min_per_overkategori (int): Minimum samples per overkategori label.
                                   If -1, no minimum enforced (random sampling).
        categories_key (str): Key for the categories file in datasources.json.
        mapping_key (str): Key for the category mapping file.

    Returns:
        Dataset: A Dataset object containing evaluation data including overkategori,
                 with stratified sampling applied.
    """
    try:
        raw_data = DataFetcher(
            DataSourceManager("config/data_sources.json")
        ).get_dataset(key=model, categories_key=categories_key, mapping_key=mapping_key)

        print(f"Loaded {len(raw_data)} total rows from dataset '{model}'")

        # Check if we have overkategori column
        if "overkategori" not in raw_data.column_names:
            print("⚠️ Warning: 'overkategori' column not found. Using random sampling.")
            if sample_size == -1 or sample_size is None:
                return raw_data
            elif len(raw_data) > sample_size:
                raw_data = raw_data.shuffle(seed=42).select(range(sample_size))
            return raw_data

        # If no minimum required, use simple random sampling
        if min_per_overkategori <= 0 or min_per_overkategori is None:
            if sample_size == -1 or sample_size is None:
                return raw_data
            elif len(raw_data) > sample_size:
                raw_data = raw_data.shuffle(seed=42).select(range(sample_size))
            return raw_data

        # === STRATIFIED SAMPLING BY OVERKATEGORI ===
        # Step 1: Index all rows by their overkategori values
        category_indices: Dict[str, List[int]] = {}
        for idx, row in enumerate(raw_data):
            cats = row.get("overkategori", [])
            if isinstance(cats, list):
                for cat in cats:
                    if cat not in category_indices:
                        category_indices[cat] = []
                    category_indices[cat].append(idx)
            elif cats and isinstance(cats, str):
                # Handle single string category
                if cats not in category_indices:
                    category_indices[cats] = []
                category_indices[cats].append(idx)

        num_categories = len(category_indices)
        print(f"🏷️ Found {num_categories} unique overkategorier")

        # Calculate distribution stats
        for cat, indices in sorted(category_indices.items()):
            pct = len(indices) / len(raw_data) * 100 if raw_data else 0
            print(
                f"   ├─ {cat}: {len(indices)} ({pct:.1f}%)\n       Available: {len(indices)}, Min requested: {min_per_overkategori}"
            )

        # Step 2: Select minimum samples per category
        selected_indices: set[int] = set()
        deficit = 0  # Track shortfall

        for cat, indices in category_indices.items():
            available = len(indices)
            needed = min(min_per_overkategori, available)

            # Shuffle to randomize within category
            shuffled_idx = indices.copy()
            import random

            random.shuffle(shuffled_idx)

            # Take minimum
            for i in shuffled_idx[:needed]:
                selected_indices.add(i)

            if available < min_per_overkategori:
                deficit += min_per_overkategori - available
                print(
                    f"   ⚠️ Category '{cat}' undersampled: {available}/{min_per_overkategori}"
                )

        sampled_count = len(selected_indices)
        remaining_slots = max(0, sample_size - sampled_count)

        # Step 3: Fill remaining slots from unselected rows (across all categories)
        unselected_indices = [
            i for i in range(len(raw_data)) if i not in selected_indices
        ]

        if remaining_slots > 0 and unselected_indices:
            import random

            random.shuffle(unselected_indices)
            additional = unselected_indices[:remaining_slots]
            selected_indices.update(additional)

            if len(additional) < remaining_slots:
                print(
                    f"⚠️ Not enough data to fill quota. Got {len(selected_indices)}/{sample_size}"
                )

        # Final validation
        final_count = len(selected_indices)
        print(f"\n✅ Selected {final_count} samples (target: {sample_size})")

        # Sort indices to preserve order stability
        sorted_selected = sorted(list(selected_indices))

        # Return subset
        if final_count > 0:
            result = raw_data.select(sorted_selected)
            print(
                f"📂 Sampled dataset shape: {result.num_rows} rows, {len(result.column_names)} columns"
            )
            return result
        else:
            print("❌ No samples selected. Returning empty fallback.")
            return Dataset.from_list(
                [
                    {
                        "user_input": ["Test input"],
                        "response": ["Test response"],
                        "reference": ["Test reference"],
                        "data_categories": [],
                        "overkategori": [],
                    }
                ]
            )

    except Exception as e:
        print(f"⚠️ Error loading dataset: {e}. Using fallback.")
        fallback = [
            {
                "user_input": ["Test input"],
                "response": ["Test response"],
                "reference": ["Test reference"],
                "data_categories": [],
                "overkategori": [],
            }
        ]
        return Dataset.from_list([fallback * min(sample_size, 5)])


def compute_mean_results(results: List[Any]) -> Dict[str, float]:
    """Computes mean values for each metric from the results list.

    Args:
        results (list): A list of dictionaries containing metric results.

    Returns:
        dict: A dictionary with mean values for each metric.
    """
    if not results:
        return {}

    # Initialize a dictionary to hold sums and counts
    metric_sums: Dict[str, int | float] = {}
    metric_counts: Dict[str, int | float] = {}

    for result in results:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                metric_sums[key] = metric_sums.get(key, 0) + value
                metric_counts[key] = metric_counts.get(key, 0) + 1

    # Compute mean values
    mean_results = {key: (metric_sums[key] / metric_counts[key]) for key in metric_sums}

    return mean_results


def compute_per_overkategorier_results(
    results: List[Any],
    df_results: pandas.DataFrame,
    overkategorier_column: str = "overkategori",
) -> Dict[str, Dict[str, float]]:
    """Computes mean results grouped by overkategori.

    Args:
        results (list): A list of evaluation results.
        df_results: Pandas DataFrame with results and overkategorier column.
        overkategorier_column: Name of the column containing overkategorier lists.

    Returns:
        dict: Dictionary with per-overkategori mean results.
    """
    if df_results.empty or overkategorier_column not in df_results.columns:
        print(
            f"⚠️ No '{overkategorier_column}' column found, skipping per-category analysis."
        )
        return {}

    # Get unique overkategorier from all rows
    unique_overkategorier = set()
    for cat_list in df_results[overkategorier_column]:
        if isinstance(cat_list, list):
            unique_overkategorier.update(cat_list)

    per_category_results: Dict[str, Dict[str, float]] = {}

    for overkat in unique_overkategorier:
        # Filter rows that have this overkategori
        mask = df_results[overkategorier_column].apply(
            lambda x: overkat in x if isinstance(x, list) else False
        )
        filtered_df = df_results[mask]

        if filtered_df.empty:
            continue

        # Calculate metrics for this category
        category_metrics: Dict[str, float] = {}

        # Extract numeric columns (excluding text fields like user_input, response)
        numeric_cols = filtered_df.select_dtypes(include=["number"]).columns.tolist()

        for col in numeric_cols:
            if col != "index":  # Skip index column
                valid_values = filtered_df[col].dropna()
                if len(valid_values) > 0:
                    category_metrics[col] = float(valid_values.mean())

        # Also use the original results if available
        if len(filtered_df) <= len(results):
            indices = filtered_df.index.tolist()
            category_results = [results[i] for i in indices if i < len(results)]
            if category_results:
                category_metrics = compute_mean_results(category_results)

        per_category_results[overkat] = category_metrics

    return per_category_results


# ==================== Client Setup ====================
def get_genai_client() -> genai.Client:
    """Initialize Google GenAI client with Vertex AI.

    Returns:
        genai.Client: An instance of the Google GenAI client.
    """
    return genai.Client(vertexai=True, project=_PROJECT_ID, location=_LOCATION)


async def run_judge_eval() -> None:
    """Runs the factual correction evaluation process asynchronously.

    This function retrieves the evaluation dataset(s), runs the evaluation,
    and saves the results to JSON files. Supports single dataset or all datasets
    from datasources.json when "all" is provided. Includes overkategori grouping.
    """
    client = get_genai_client()  # Initialize the GenAI client

    # Determine which datasets to process: list of (name, path) tuples
    datasets_to_process: List[tuple[str, str]] = []

    if len(sys.argv) > 1:
        cmd_arg = sys.argv[1]
        if cmd_arg == "all":
            # Load dataset map from datasources.json
            try:
                with open("config/data_sources.json", "r", encoding="utf-8") as f:
                    datasource_config = json.load(f)

                if not isinstance(datasource_config, dict):
                    raise ValueError(
                        "datasources.json must contain a JSON object (dictionary)"
                    )

                if not datasource_config:
                    print(
                        "Warning: 'all' specified but no datasets found in datasources.json"
                    )
                    return

                datasets_to_process = list(datasource_config.items())

            except FileNotFoundError:
                print(
                    "Error: config/data_sources.json not found. Please provide a specific dataset name or create config/data_sources.json"
                )
                return
            except json.JSONDecodeError:
                print("Error: Invalid JSON in config/data_sources.json")
                return
            except ValueError as e:
                print(f"Error: {e}")
                return
        else:
            # Single dataset specified directly
            datasets_to_process = [(cmd_arg, cmd_arg)]
    else:
        # No argument provided - use default dataset defined in global scope
        if "_EVAL_DATASET" in globals():
            datasets_to_process = [(_EVAL_DATASET, _EVAL_DATASET)]
        else:
            print("Error: No command line argument provided and _EVAL_DATASET not set.")
            return

    # Create results directory if it doesn't exist
    # results_dir = "./data/results_judge"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Track aggregate results across multiple datasets
    all_mean_results: Dict[str, Any] = {}
    all_overkategori_results: Dict[str, Any] = {}

    for dataset_name, dataset_path in datasets_to_process:
        if dataset_name == "bob_answers":
            print(
                "Skipping dataset 'bob_answers' as it is not intended for evaluation."
            )
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"Source path: {dataset_path}")
        print("=" * 60)

        try:
            # Load the evaluation dataset with overcategories
            dataSet = load_evaluation_dataset(
                model=dataset_name,
                sample_size=_SAMPLE_SIZE,
                categories_key="categories",
                mapping_key="category_mapping",
            )

            # Check if overkategori column exists
            has_overkategori = "overkategori" in dataSet.column_names
            if has_overkategori:
                print(
                    f"📋 Detected {len(set([cat for row in dataSet['overkategori'] for cat in row]))} unique overkategorier in dataset: {(set([cat for row in dataSet['overkategori'] for cat in row]))}"
                )
            else:
                print("⚠️ Warning: 'overkategori' column not found in dataset.")

            # Run evaluation with specified batch size
            # async def _factual_correctness_eval(Dataset, client, model=_JUDGE_MODEL, batch_size=_BATCH_SIZE)-> List[Dict[str, Any]]:
            #    results = await claim_decom_fc_bidirectional.evaluate_dataset_batch(
            #        Dataset, client, model=_JUDGE_MODEL, batch_size=_BATCH_SIZE
            #    )

            #    # Create result dataframe with claims included
            #    df_results = claim_decom_fc_bidirectional.create_result_dataframe(
            #        results, original_dataset=dataSet, include_claims=True
            #    )
            #    return results, df_results
            results = await claim_decom_fc_bidirectional.evaluate_dataset_batch(
                dataSet, client, model=_JUDGE_MODEL, batch_size=_BATCH_SIZE
            )

            # Create result dataframe with claims included
            df_results = claim_decom_fc_bidirectional.create_result_dataframe(
                results, original_dataset=dataSet, include_claims=True
            )
            # results, df_results = _factual_correctness_eval(dataSet, client, model=_JUDGE_MODEL, batch_size=_BATCH_SIZE)
            # Print summary statistics
            print(df_results.head())

            # Compute mean results for each metric (overall)
            mean_results = compute_mean_results(results)
            print(f"\n📊 Mean Results for {dataset_name}:", mean_results)
            print(df_results.describe())
            mean_results_dict = df_results.describe().to_dict()

            # Compute per-overkategori results if column exists
            per_overkategori_summary: Dict[str, Dict[str, float]] = {}
            if has_overkategori:
                per_overkategori_summary = compute_per_overkategorier_results(
                    results, df_results, overkategorier_column="overkategori"
                )

                if per_overkategori_summary:
                    print("\n🏷️ Per-overkategori Summary:")
                    for overkat, metrics in sorted(per_overkategori_summary.items()):
                        print(f"   └─ {overkat}: {metrics}")
                else:
                    print(
                        "\n⚠️ No per-overkategori results generated (no matches found)."
                    )

            # Save detailed results to JSON files
            full_results = {
                "timestamp": datetime.now().isoformat(),
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "total_rows": len(df_results),
                "has_overkategori": has_overkategori,
                "results": results,
            }

            # Save main results
            output_filename = f"fc_eval_results.{dataset_name}.json"
            output_path = os.path.join(RESULTS_DIR, output_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(full_results, f, indent=4, ensure_ascii=False, default=str)
            print(f"\n✓ Detailed results saved to {output_path}")

            # Save mean results (overall)
            output_filename = f"fc_eval_mean_results.{dataset_name}.json"
            output_path = os.path.join(RESULTS_DIR, output_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(mean_results_dict, f, indent=4, ensure_ascii=False)
            print(f"✓ Mean results saved to {output_path}")

            # Save per-overkategori summary if available
            if per_overkategori_summary:
                output_filename = f"fc_eval_overkategori_summary.{dataset_name}.json"
                output_path = os.path.join(RESULTS_DIR, output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "dataset_name": dataset_name,
                            "overkategorier_count": len(per_overkategori_summary),
                            "per_overkategori_metrics": per_overkategori_summary,
                        },
                        f,
                        indent=4,
                        ensure_ascii=False,
                    )
                print(f"✓ Overkategori summary saved to {output_path}")

                # Also add to aggregated tracking
                all_overkategori_results[dataset_name] = per_overkategori_summary

            print(f"\n✅ Completed processing {dataset_name}")

            # Aggregate mean results if processing multiple datasets
            if len(datasets_to_process) > 1:
                all_mean_results[dataset_name] = mean_results_dict

        except Exception as e:
            print(f"❌ Error processing dataset {dataset_name}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    # If multiple datasets were processed, save combined summaries
    if len(datasets_to_process) > 1:
        combined_summary = {
            "timestamp": datetime.now().isoformat(),
            "datasets_processed": [name for name, _ in datasets_to_process],
            "individual_results": all_mean_results,
        }
        combined_path = os.path.join(RESULTS_DIR, "combined_eval_summary.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined_summary, f, indent=4, ensure_ascii=False)
        print(f"\n📊 Combined summary saved to {combined_path}")

        # Save combined overkategori results
        if all_overkategori_results:
            combined_overkat_path = os.path.join(
                RESULTS_DIR, "combined_overkategori_summary.json"
            )
            with open(combined_overkat_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "datasets_processed": list(all_overkategori_results.keys()),
                        "per_dataset_overkategorier": all_overkategori_results,
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            print(f"🏷️ Combined overkategori summary saved to {combined_overkat_path}")

    # Print final summary
    print(f"\n{'=' * 80}")
    print("🎉 Evaluation completed!")
    print(f"Results stored in: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    # Run the evaluation asynchronously
    asyncio.run(run_judge_eval())

    if UPLOAD_TO_STORAGE:
        from storage import upload_dir

        print("Uploading results to storage...")
        upload_dir(RESULTS_DIR, RESULTS_DIR, CONFIG_PATH)
