"""Module for creating a leaderboard for model evaluation.

This module provides functionalities to evaluate models based on various metrics,
normalize values, and create a scoring pipeline configuration. It includes
default settings for metrics, weights, and thresholds, and allows for the
evaluation of individual models against these criteria.

Key functionalities include:
- Mapping raw metrics to canonical categories.
- Normalizing metric values.
- Evaluating models based on defined thresholds and weights.

Usage:
=====
# Quick run (skip evaluation if results exist)
python eval_judge/run_evaluation.py all

# Force full recomputation
python eval_judge/run_evaluation.py all --force

# With short flag
python eval_judge/run_evaluation.py all -f
"""

import asyncio
import glob
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from google import genai

# Local imports
from storage import _load_cfg

try:
    from eval_judge import claim_decom_fc_bidirectional
    from eval_judge.DataFetcher import DataFetcher
    from eval_judge.DataManager import DataSourceManager
except ImportError:
    print("Warning: Local modules (eval_judge.*) not found.")


# ==================== Configuration ====================
CONFIG_PATH = "src/config.yml"
cfg = _load_cfg(CONFIG_PATH)

_PROJECT_ID = cfg.get("project", "your-project-id")
_LOCATION = cfg.get("location", "europe-north1")
_JUDGE_MODEL = "gemini-2.5-pro"
_EVAL_DATASET = "claude-sonnet-4-6"
_SAMPLE_SIZE = -1
_BATCH_SIZE = 10

RESULTS_DIR = "./data/"

# Skip evaluation if results already exist (default=True)
SKIP_RECOMPUTE = True

# ==================== MODULAR SCORING ENGINE ====================


def get_default_metric_map() -> Dict[str, List[str]]:
    """Maps raw metric keys to canonical categories."""
    return {
        "rouge_l": ["rouge_l"],
        "semantic_sim": ["bertscore_f1", "cosine_similarity"],
        "factual_corr": ["nli_entailment", "f1_score"],
        "answer_quality": ["answer_quality"],
        "jsd_divergence": ["jsd"],
    }


def get_default_weights() -> Dict[str, float]:
    """Default weights for composite score calculation."""
    return {
        "rouge_l": 0.15,
        "semantic_sim": 0.25,
        "factual_corr": 0.40,
        "answer_quality": 0.20,
    }


def get_default_thresholds() -> Dict[str, float]:
    """Safety thresholds."""
    return {
        "factual_corr": 0.10,
        "answer_quality": 0.10,
        "jsd_divergence": 0.90,
    }


def create_pipeline_config(
    metric_map: Optional[Dict[str, List[str]]] = None,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    scoring_method: str = "geometric_mean",
) -> Dict[str, Any]:
    """Creates a complete pipeline configuration dictionary.

    Args:
        metric_map (Optional[Dict[str, List[str]]]): A mapping of metrics to their keys.
        weights (Optional[Dict[str, float]]): Weights for each metric.
        thresholds (Optional[Dict[str, float]]): Safety thresholds for metrics.
        scoring_method (str): The method used for scoring (default is "geometric_mean").

    Returns:
        Dict[str, Any]: A dictionary containing the pipeline configuration.
    """
    return {
        "metric_map": metric_map if metric_map else get_default_metric_map(),
        "weights": weights if weights else get_default_weights(),
        "thresholds": thresholds if thresholds else get_default_thresholds(),
        "scoring_method": scoring_method,
    }


def normalize_value(
    value: Optional[float], min_val: float = 0.0, max_val: float = 1.0
) -> float:
    """Safely normalizes a value to [min, max].

    Args:
        value (Optional[float]): The value to normalize.
        min_val (float): The minimum value of the range (default is 0.0).
        max_val (float): The maximum value of the range (default is 1.0).

    Returns:
        float: The normalized value.
    """
    if value is None:
        return 0.0
    try:
        v = float(value)
        return float(np.clip(v, min_val, max_val))
    except ValueError, TypeError:
        return 0.0


def extract_metric(
    data: Dict[str, Any], possible_keys: List[str], default: float = 0.0
) -> float:
    """Finds the first matching key in the data dict.

    Args:
        data (Dict[str, Any]): The data dictionary to search.
        possible_keys (list): A list of possible keys to look for.
        default (float): The default value to return if no keys are found (default is 0.0).

    Returns:
        float: The extracted metric value or the default value.
    """
    for key in possible_keys:
        if key in data:
            val = data[key]
            if isinstance(val, (list, tuple)):
                val = val[0] if val else 0.0
            return normalize_value(val)
    return default


def evaluate_single_model(
    raw_data: Dict[str, Any], config: Dict[str, Any]
) -> Tuple[Dict[str, float], str]:
    """Maps raw data to canonical metrics, checks safety gates, and calculates composite score.

    Args:
        raw_data (Dict[str, Any]): The raw data to evaluate.
        config (Dict[str, Any]): The configuration dictionary containing metrics, weights, and thresholds.

    Returns:
        Tuple[Dict[str, float], str]: A tuple containing the mapped metrics and the evaluation status.

    FIX: Always calculates composite_score before returning status to ensure the key exists.
    """
    mm = config["metric_map"]
    thresholds = config["thresholds"]
    weights = config["weights"]

    # 1. Map Metrics
    mapped = {
        "rouge_l": extract_metric(raw_data, mm.get("rouge_l", [])),
        "semantic_sim": extract_metric(raw_data, mm.get("semantic_sim", [])),
        "factual_corr": extract_metric(raw_data, mm.get("factual_corr", [])),
        "answer_quality": extract_metric(raw_data, mm.get("answer_quality", [])),
        "jsd_divergence": extract_metric(raw_data, mm.get("jsd_divergence", [])),
    }

    # 2. Check Safety Gates FIRST to determine status
    status = "PASS"

    if mapped["factual_corr"] < thresholds.get("factual_corr", 0.0):
        status = "FAIL_FACTUALITY"
    elif mapped["answer_quality"] < thresholds.get("answer_quality", 0.0):
        status = "FAIL_QUALITY"
    elif mapped["jsd_divergence"] > thresholds.get("jsd_divergence", 1.0):
        status = "FAIL_DIVERGENCE"

    # 3. Calculate Composite Score (ALWAYS calculate, even if failed)
    total_weight = sum(weights.values())
    score = 0.0

    if total_weight > 0:
        if config["scoring_method"] == "weighted_avg":
            weighted_sum = 0.0
            for m_name, weight in weights.items():
                if weight <= 0:
                    continue
                val = mapped.get(m_name, 0)
                if m_name == "jsd_divergence":
                    val = 1.0 - val
                weighted_sum += val * weight
            score = weighted_sum / total_weight

        elif config["scoring_method"] == "geometric_mean":
            weighted_log_sum = 0.0
            count = 0
            for m_name, weight in weights.items():
                if weight <= 0:
                    continue
                val = mapped.get(m_name, 0)
                if m_name == "jsd_divergence":
                    val = 1.0 - val

                # Numerical stability
                safe_val = max(val, 1e-9)
                weighted_log_sum += weight * math.log(safe_val)
                count += 1

            score = math.exp(weighted_log_sum / count) if count > 0 else 0.0

    # Add composite_score to the mapped dict
    mapped["composite_score"] = score

    return mapped, status


def process_leaderboard_data(
    models_data: Dict[str, Dict[str, Any]], config: Dict[str, Any]
) -> pd.DataFrame:
    """Iterates over models and builds the dataframe."""
    results = []

    for model_id, raw_data in models_data.items():
        metrics, status = evaluate_single_model(raw_data, config)

        results.append(
            {
                "model_id": model_id,
                **metrics,
                "status": status,
                "raw_snippet": json.dumps(raw_data)[:200],
            }
        )

    df = pd.DataFrame(results)

    # SAFETY CHECK: Ensure column exists before sorting
    if not df.empty and "composite_score" in df.columns:
        df = df.sort_values(by="composite_score", ascending=False).reset_index(
            drop=True
        )
    elif not df.empty:
        print(
            f"⚠️ Warning: No 'composite_score' column found for {len(df)} models. Sorting disabled."
        )

    return df


def format_table(df: pd.DataFrame) -> str:
    """Formats the dataframe into a clean string table."""
    if df.empty:
        return "\nNo models to display."

    cols = [
        "model_id",
        "rouge_l",
        "semantic_sim",
        "factual_corr",
        "answer_quality",
        "jsd_divergence",
        "composite_score",
        "status",
    ]
    available_cols = [c for c in cols if c in df.columns]
    display_df = df[available_cols].copy()

    numeric_cols = [
        c for c in available_cols if c not in ("model_id", "status", "raw_snippet")
    ]
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )

    header = "=" * 120
    footer = "=" * 120

    output = [header, "LLM LEADERBOARD EVALUATION RESULTS", header]
    output.append(display_df.to_string(index=False))
    output.append(footer)

    return "\n".join(output)


# ==================== END MODULAR SCORING ENGINE ====================


def load_evaluation_dataset(
    model: str,
    sample_size: int = _SAMPLE_SIZE,
    categories_key: str = "categories",
    mapping_key: str = "category_mapping",
) -> Dataset:
    """Load the evaluation dataset for a given model.

    This function fetches the dataset from a specified data source and returns
    a subset of the data based on the provided sample size. If the sample size
    is -1 or exceeds the length of the dataset, the entire dataset is returned.
    In case of an error during data fetching, a fallback dataset is returned.

    Args:
        model (str): The key of the model for which the dataset is to be fetched.
        sample_size (int, optional): The number of samples to return from the dataset.
                                      Defaults to _SAMPLE_SIZE.
        categories_key (str, optional): The key to access categories in the dataset.
                                         Defaults to "categories".
        mapping_key (str, optional): The key to access category mapping in the dataset.
                                      Defaults to "category_mapping".

    Returns:
        Dataset: A dataset containing the requested samples or a fallback dataset
                  in case of an error.
    """
    try:
        fetcher = DataFetcher(DataSourceManager("config/data_sources.json"))
        raw_data = fetcher.get_dataset(
            key=model, categories_key=categories_key, mapping_key=mapping_key
        )
        if sample_size == -1 or sample_size is None or len(raw_data) <= sample_size:
            return raw_data
        return raw_data.select(range(sample_size))
    except Exception as e:
        print(f"Warning: Failed to load dataset ({e}). Using fallback.")
        fallback = {
            "user_input": ["Test"],
            "response": ["Test"],
            "reference": ["Test"],
            "data_categories": [],
            "overkategori": [],
        }
        return Dataset.from_list(
            [fallback] * min(sample_size if sample_size != -1 else 5, 5)
        )


def generate_category_leaderboards(
    results_dir: str, scoring_method: str = "geometric_mean"
) -> None:
    """Generate one comprehensive leaderboard per category using the Modular Scoring Engine.

    Args:
        results_dir (str): The directory where the results are stored.
        scoring_method (str, optional): The scoring method to use. Defaults to "geometric_mean".

    Returns:
        None
    """
    combined_f1_path = os.path.join(
        results_dir, "results_judge/combined_overkategori_summary.json"
    )
    combined_semantic_path = os.path.join(
        results_dir, "results/evaluation_report_scores_per_overkategori.json"
    )

    if not os.path.exists(combined_f1_path):
        print(f"⚠️ Missing {combined_f1_path}. Cannot generate leaderboards.")
        return

    # Load F1 Data
    try:
        with open(combined_f1_path, "r", encoding="utf-8") as f:
            f1_data = json.load(f).get("summary", {})
    except Exception as e:
        print(f"❌ Error loading F1 summary: {e}")
        return

    # Load Semantic Data
    semantic_data = {}
    if os.path.exists(combined_semantic_path):
        try:
            with open(combined_semantic_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                semantic_data = content.get("scores", content)
        except Exception as e:
            print(f"⚠️ Could not load semantic scores: {e}")
    else:
        print("⚠️ No semantic scores file found. Using F1 only.")

    # Identify all unique categories
    all_categories = set()
    for model_metrics in f1_data.values():
        all_categories.update(model_metrics.keys())
    for model_cat in semantic_data.values():
        all_categories.update(model_cat.keys())

    sorted_categories = sorted(list(all_categories))
    print(
        f"\n🏆 Generating Leaderboards ({scoring_method}) for {len(sorted_categories)} categories..."
    )

    # Define Pipeline Config
    custom_metric_map = {
        "rouge_l": ["rouge_l"],
        "semantic_sim": ["bertscore_f1", "cosine_similarity", "nli_entailment"],
        "factual_corr": ["f1_score", "nli_entailment"],
        "answer_quality": ["answer_quality"],
        "jsd_divergence": ["jsd"],
    }

    config = create_pipeline_config(
        metric_map=custom_metric_map,
        weights=get_default_weights(),
        thresholds=get_default_thresholds(),
        scoring_method=scoring_method,
    )

    for cat in sorted_categories:
        models_raw_data = {}

        # Aggregate data for this category
        for model in set(list(f1_data.keys()) + list(semantic_data.keys())):
            raw_entry = {}

            # Extract values from Source 1 (F1)
            if model in f1_data and cat in f1_data[model]:
                entry = f1_data[model][cat]
                if "f1_score" in entry:
                    raw_entry["f1_score"] = entry["f1_score"]
                if "nli_entailment" in entry:
                    raw_entry["nli_entailment"] = entry["nli_entailment"]
                if "rouge_l" in entry:
                    raw_entry["rouge_l"] = entry["rouge_l"]

            # Extract values from Source 2 (Semantic)
            if model in semantic_data and cat in semantic_data[model]:
                entry = semantic_data[model][cat]
                if "bertscore_f1" in entry:
                    raw_entry["bertscore_f1"] = entry["bertscore_f1"]
                if "cosine_similarity" in entry:
                    raw_entry["cosine_similarity"] = entry["cosine_similarity"]
                if "jsd" in entry:
                    raw_entry["jsd"] = entry["jsd"]
                if "rouge_l" in entry:
                    raw_entry["rouge_l"] = entry["rouge_l"]

            if raw_entry:
                models_raw_data[model] = raw_entry

        if not models_raw_data:
            continue

        # Run Pipeline
        df_results = process_leaderboard_data(models_raw_data, config)

        if df_results.empty:
            continue

        # Save CSV
        safe_cat_name = cat.replace(" ", "_").replace("/", "_").lower()
        csv_filename = f"leaderboard_{safe_cat_name}_{scoring_method}.csv"
        leaderboard_dir = os.path.join(results_dir, "leaderboards")
        os.makedirs(leaderboard_dir, exist_ok=True)
        csv_path = os.path.join(leaderboard_dir, csv_filename)

        export_df = df_results.drop(columns=["raw_snippet"])
        export_df.to_csv(csv_path, index=False)

        # Print Top 5
        print(f"\n🥇 Category: {cat}")
        print("-" * 80)
        top_5 = export_df.head(5)[["model_id", "composite_score", "status"]]
        print(top_5.to_string(index=False))
        print(f"✅ Saved: {csv_filename} (Total: {len(export_df)} models)")

    print("\n✅ All category leaderboards generated successfully.")


def generate_master_leaderboard(
    results_dir: str, scoring_method: str = "geometric_mean"
) -> None:
    """Creates a MASTER LEADERBOARD by merging all category leaderboards.

    Features:
    - Loads composite scores from each category leaderboard CSV
    - Merges by model_id
    - Calculates geometric average across all categories as Global_Composite_Score
    - Adds placeholder columns AP1-AP6 with N/A values
    - Sorts by Global_Composite_Score descending
    """
    leaderboard_dir = os.path.join(results_dir, "leaderboards")

    if not os.path.exists(leaderboard_dir):
        print(f"⚠️ Leaderboard directory not found: {leaderboard_dir}")
        return

    # Find all category leaderboard CSV files
    csv_files = glob.glob(
        os.path.join(leaderboard_dir, f"leaderboard_*_{scoring_method}.csv")
    )

    if not csv_files:
        print(f"⚠️ No category leaderboards found in {leaderboard_dir}")
        return

    print(
        f"\n📊 Building Master Leaderboard from {len(csv_files)} category leaderboards..."
    )

    # Dictionary to hold model -> {category: composite_score}
    master_data: Dict[str, Dict[str, float]] = defaultdict(dict)
    all_models: Set[str] = set()
    all_categories: List[str] = []

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"filename: {filename}")
        # Extract category name from filename: leaderboard_familie_og_barn_geometric.csv
        parts = filename.replace(f"_{scoring_method}.csv", "").replace(
            "leaderboard_", ""
        )  # .split("_")
        # print(f"parts: {parts}")
        # if len(parts) > 1:
        #    category = " ".join(parts[:-1])
        # else:
        #    category = parts[0] if parts else "unknown"
        ##category = " ".join(parts[:-1]) if len(parts) > 1 else "unknown"
        category = parts  # .replace("_", " ")
        all_categories.append(category)
        print(f"categories: {all_categories}")

        try:
            df = pd.read_csv(csv_file)
            if "model_id" not in df.columns or "composite_score" not in df.columns:
                print(f"⚠️ Skipping {filename}: missing required columns")
                continue

            for _, row in df.iterrows():
                model_id = row["model_id"]
                score = row["composite_score"]
                master_data[model_id][category] = score
                all_models.add(model_id)
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
            continue

    if not master_data:
        print("⚠️ No valid data found in category leaderboards.")
        return

    # Build master dataframe
    rows = []
    for model_id in sorted(all_models):
        row_data = {"model_id": model_id}

        # Add composite score for each category
        for category in sorted(all_categories):
            score = master_data[model_id].get(category, np.nan)
            row_data[f"Composite_{category.replace(' ', '_')}"] = (
                round(score, 4) if not pd.isna(score) else np.nan
            )

        # Calculate geometric mean across all available categories
        available_scores = [v for v in master_data[model_id].values() if not pd.isna(v)]
        if available_scores:
            # Geometric mean: exp(sum(ln(x))/n)
            log_sum = sum(math.log(max(s, 1e-9)) for s in available_scores)
            geo_avg = math.exp(log_sum / len(available_scores))
            row_data["Global_Composite_Score"] = str(round(geo_avg, 4))
        else:
            row_data["Global_Composite_Score"] = np.nan

        # Add placeholder columns AP1-AP6
        for i in range(1, 7):
            row_data[f"AP{i}"] = "N/A"

        rows.append(row_data)

    # Create DataFrame
    df_master = pd.DataFrame(rows)

    # Sort by Global_Composite_Score descending (NaN values at bottom)
    if "Global_Composite_Score" in df_master.columns:
        df_master = df_master.sort_values(
            by="Global_Composite_Score", ascending=False, na_position="last"
        )

    # Save Master Leaderboard
    master_filename = f"master_leaderboard_{scoring_method}.csv"
    master_path = os.path.join(leaderboard_dir, master_filename)
    df_master.to_csv(master_path, index=False)

    # Also save JSON version
    master_json_path = os.path.join(
        leaderboard_dir, f"master_leaderboard_{scoring_method}.json"
    )
    df_master_json = df_master.copy()
    # Convert NaN to None for JSON serialization
    df_master_json = df_master_json.where(pd.notnull(df_master_json), None)
    df_master_json.to_json(
        master_json_path, orient="records", indent=2, force_ascii=False
    )

    print("✅ Master Leaderboard saved to:")
    print(f"   📄 CSV: {master_filename}")
    print(f"   📋 JSON: master_leaderboard_{scoring_method}.json")

    # Print summary
    print(f"\n{'=' * 80}")
    print("🏆 MASTER LEADERBOARD SUMMARY")
    print("=" * 80)
    print(f"Total Models: {len(df_master)}")
    print(f"Categories Covered: {len(all_categories)}")

    # Show top 5
    if len(df_master) > 0:
        print("\n🥇 TOP 5 MODELS:")
        top_cols = ["model_id", "Global_Composite_Score"]
        # Add first 3 category columns for context
        cat_cols = [col for col in df_master.columns if col.startswith("Composite_")][
            : len(all_categories)
        ]
        print(df_master[top_cols + cat_cols].head(5).to_string(index=False))

    print("=" * 80)


def compute_mean_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute the mean of numerical results from a list of dictionaries.

    Args:
        results (List[Dict[str, Any]]): A list of dictionaries containing
        numerical results. Each dictionary represents a single result.

    Returns:
        Dict[str, float]: A dictionary where the keys are the metric names
        and the values are the mean of the corresponding metrics from the
        input results. If the input list is empty, an empty dictionary is returned.
    """
    if not results:
        return {}
    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        for key, value in result.items():
            if isinstance(value, (int, float)):
                metric_sums[key] = metric_sums.get(key, 0.0) + value
                metric_counts[key] = metric_counts.get(key, 0) + 1
    return {key: val / metric_counts[key] for key, val in metric_sums.items()}


def compute_per_overkategorier_results(
    df_results: pd.DataFrame, overkategorier_column: str = "overkategori"
) -> Dict[str, Dict[str, float]]:
    """Computes the average results per category from a DataFrame.

    This function takes a DataFrame containing results and a specified column
    that categorizes the results into different groups. It calculates the
    average of numeric columns for each unique category found in the specified
    column.

    Args:
        df_results (pd.DataFrame): The DataFrame containing results and categories.
        overkategorier_column (str): The name of the column containing category
                                      information. Defaults to "overkategori".

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where keys are unique categories
                                      and values are dictionaries of metric names
                                      and their corresponding average values.
    """
    if df_results.empty or overkategorier_column not in df_results.columns:
        return {}
    unique_cats: Set[str] = set()
    for item in df_results[overkategorier_column]:
        if isinstance(item, list):
            unique_cats.update(item)
    if not unique_cats:
        return {}
    category_buckets: Dict[str, Dict[str, List[float]]] = {
        cat: {} for cat in unique_cats
    }
    exclude_cols = {
        "user_input",
        "response",
        "reference",
        overkategorier_column,
        "index",
    }
    numeric_cols = [
        col
        for col in df_results.columns
        if col not in exclude_cols
        and df_results[col].dtype in ["float64", "int64", "int32", "float32"]
    ]
    if not numeric_cols:
        return {}
    for _, row in df_results.iterrows():
        cats = row.get(overkategorier_column, [])
        if not isinstance(cats, list):
            continue
        row_metrics = {}
        for col in numeric_cols:
            val = row[col]
            if not pd.isna(val):
                row_metrics[col] = float(val)
        if not row_metrics:
            continue
        for cat in cats:
            if cat not in category_buckets:
                continue
            for metric_name, value in row_metrics.items():
                if metric_name not in category_buckets[cat]:
                    category_buckets[cat][metric_name] = []
                category_buckets[cat][metric_name].append(value)
    final_summary = {}
    for cat, metrics_map in category_buckets.items():
        calculated = {}
        for metric, values in metrics_map.items():
            if values:
                calculated[metric] = round(sum(values) / len(values), 4)
        if calculated:
            final_summary[cat] = calculated
    return final_summary


async def run_fc_eval(force_recompute: bool = False) -> None:
    """Main entry point for running evaluations.

    This function evaluates datasets and generates leaderboards based on the results.
    If cached results exist and the force_recompute flag is not set, it will skip the evaluation
    and regenerate leaderboards from the cached data.

    Args:
        force_recompute (bool): If True, ignore cached results and recompute everything.
                                Default is False, which skips evaluation if results exist.

    Returns:
        None
    """
    if _LOCATION != "europe-north1":
        print(
            f"⚠️ Warning: Current location is {_LOCATION}. Recommended: 'europe-north1'. Returning."
        )
        return

    # Check if we should skip full evaluation
    combined_f1_path = os.path.join(
        RESULTS_DIR, "results_judge/combined_overkategori_summary.json"
    )

    if SKIP_RECOMPUTE and not force_recompute and os.path.exists(combined_f1_path):
        print("🔄 Found existing results. Skipping evaluation phase...")
        print(
            "   Use --force flag to recompute: python eval_judge/run_evaluation.py all --force"
        )

        # Still generate leaderboards from existing data
        generate_category_leaderboards(RESULTS_DIR, scoring_method="geometric_mean")
        generate_master_leaderboard(RESULTS_DIR, scoring_method="geometric_mean")

        print(f"\n{'=' * 80}")
        print("✅ Leaderboards regenerated from cached results.")
        print(f"{'=' * 80}")
        return

    client = genai.Client(vertexai=True, project=_PROJECT_ID, location=_LOCATION)

    datasets_to_process: List[tuple[str, str]] = []
    # force_flag = "--force" in sys.argv or "-f" in sys.argv

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "all":
            try:
                with open("config/data_sources.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                if isinstance(config, dict):
                    datasets_to_process = [
                        (k, v) for k, v in config.items() if k != "bob_answers"
                    ]
                else:
                    raise ValueError("data_sources.json must be a JSON object.")
            except Exception as e:
                print(f"Error loading data_sources.json: {e}")
                return
        else:
            datasets_to_process = [(arg, arg)]
    else:
        datasets_to_process = [(_EVAL_DATASET, _EVAL_DATASET)]

    if not datasets_to_process:
        print("No datasets to process.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_overkat_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dataset_name, dataset_path in datasets_to_process:
        if dataset_name == "bob_answers":
            print(f"Skipping {dataset_name} (reference only).")
            continue

        print(f"\n{'=' * 80}\nProcessing: {dataset_name}\n{'=' * 80}")
        try:
            ds = load_evaluation_dataset(
                model=dataset_name,
                sample_size=_SAMPLE_SIZE,
                categories_key="categories",
                mapping_key="category_mapping",
            )
            has_overkat = "overkategori" in ds.column_names

            # Compute factual correction
            results = await claim_decom_fc_bidirectional.evaluate_dataset_batch(
                ds, client, model=_JUDGE_MODEL, batch_size=_BATCH_SIZE
            )
            df_res = claim_decom_fc_bidirectional.create_result_dataframe(
                results, original_dataset=ds, include_claims=True
            )
            if "overkategori" not in df_res.columns and has_overkat:
                df_res["overkategori"] = ds["overkategori"]

            overall_means = compute_mean_results(results)
            per_cat_summary = {}
            if has_overkat and not df_res.empty:
                per_cat_summary = compute_per_overkategorier_results(
                    df_res, "overkategori"
                )

            timestamp = datetime.now().isoformat()
            full_res = {
                "timestamp": timestamp,
                "dataset": dataset_name,
                "total_rows": len(df_res),
                "results": results,
                "df_stats": df_res.describe().to_dict(),
            }
            save_file(
                os.path.join(
                    RESULTS_DIR, f"results_judge/fc_eval_results.{dataset_name}.json"
                ),
                full_res,
            )
            save_file(
                os.path.join(
                    RESULTS_DIR,
                    f"results_judge/fc_eval_mean_results.{dataset_name}.json",
                ),
                overall_means,
            )

            if per_cat_summary:
                save_file(
                    os.path.join(
                        RESULTS_DIR,
                        f"results_judge/fc_eval_overkategori_summary.{dataset_name}.json",
                    ),
                    {
                        "timestamp": timestamp,
                        "dataset": dataset_name,
                        "categories_count": len(per_cat_summary),
                        "metrics_by_category": per_cat_summary,
                    },
                )
                all_overkat_summaries[dataset_name] = per_cat_summary

        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            continue

    # Combine Summary
    if len(datasets_to_process) > 1 and all_overkat_summaries:
        save_file(
            os.path.join(
                RESULTS_DIR, "results_judge/combined_overkategori_summary.json"
            ),
            {
                "timestamp": datetime.now().isoformat(),
                "datasets_processed": list(all_overkat_summaries.keys()),
                "summary": all_overkat_summaries,
            },
        )

    # Generate Category Leaderboards
    generate_category_leaderboards(RESULTS_DIR, scoring_method="geometric_mean")

    # Generate Master Leaderboard
    generate_master_leaderboard(RESULTS_DIR, scoring_method="geometric_mean")

    print(f"\n{'=' * 80}")
    print("🎉 All evaluations finished.")
    print(f"Results directory: {os.path.abspath(RESULTS_DIR)}")
    print(f"{'=' * 80}")


def save_file(path: str, data: Any) -> None:
    """Save data to a specified file path in JSON format.

    This function creates the necessary directories if they do not exist,
    and writes the provided data to a file at the specified path. If the
    operation is successful, a confirmation message is printed. If it fails,
    an error message is displayed.

    Args:
        path (str): The file path where the data should be saved.
        data (Any): The data to be saved in JSON format.

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False, default=str)
        print(f"✓ Saved: {path}")
    except Exception as e:
        print(f"❌ Failed to save {path}: {e}")


if __name__ == "__main__":
    # Check for force flag
    force_recompute = "--force" in sys.argv or "-f" in sys.argv
    asyncio.run(run_fc_eval(force_recompute=force_recompute))
