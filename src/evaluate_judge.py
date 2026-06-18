"""evaluate_judge: run judge-based metric evaluations.

This module is responsible for evaluating the performance of a judge model
using a specified evaluation dataset. It includes functions to load the
evaluation dataset, compute mean results from evaluation metrics, and run
the factual correction evaluation process asynchronously.

Key functionalities:
- Load evaluation datasets from various sources.
- Compute mean results for evaluation metrics.
- Run evaluations using the Google GenAI client.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from datasets import Dataset
from google import genai

from eval_judge.Datafetcher import DataFetcher

# Local imports
from storage import _load_cfg

try:
    from eval_judge import claim_decom_fc_bidirectional
    from eval_judge.DataFetcher import DataFetcher
    from eval_judge.DataManager import DataSourceManager

except ImportError:
    print(
        "Warning: Local modules not found. Ensure that eval_judge/datafetch.py and eval_judge/claim_decom_fc_bidirectional.py are present."
    )


# ==================== Configuration ====================
CONFIG_PATH = "src/config.yml"
cfg = _load_cfg(CONFIG_PATH)

_PROJECT_ID = cfg["project"]
_LOCATION = cfg["location"]
_JUDGE_MODEL = "gemini-2.5-pro"
_EVAL_DATASET = "claude-sonnet-4-6"  # Default dataset to evaluate
_SAMPLE_SIZE = 20
_BATCH_SIZE = 5  # Number of rows to process in parallel per step


def load_evaluation_dataset(
    model: str = _EVAL_DATASET, sample_size: int = _SAMPLE_SIZE
) -> Dataset:
    """Loads and prepares the evaluation dataset.

    Args:
        model (str): The name of the model to fetch the dataset for.
        sample_size (int): The number of samples to load from the dataset.
                           If -1 or None, the entire dataset is loaded.

    Returns:
        Dataset: A Dataset object containing the evaluation data.
    """
    try:
        # raw_data = datafetch.get_dataset(gen_model=model)
        raw_data = DataFetcher(
            DataSourceManager("config/data_sources.json")
        ).get_dataset(model)
        # Check if we need to limit the sample size
        if sample_size == -1 or sample_size is None:
            return raw_data
        elif len(raw_data) > sample_size:
            raw_data = raw_data.select(range(sample_size))
        return raw_data

    except Exception as e:
        print(f"Warning: datafetch failed ({e}). Using fallback.")

    # Fallback data in case datafetch fails
    fallback = {
        "user_input": ["Test input"],
        "response": ["Test response"],
        "reference": ["Test reference"],
    }
    return Dataset.from_list([fallback] * min(sample_size, 5))


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


# ==================== Client Setup ====================
def get_genai_client() -> genai.Client:
    """Initialize Google GenAI client with Vertex AI.

    Returns:
        genai.Client: An instance of the Google GenAI client.
    """
    return genai.Client(vertexai=True, project=_PROJECT_ID, location=_LOCATION)


async def run_fc_eval() -> None:
    """Runs the factual correction evaluation process asynchronously.

    This function retrieves the evaluation dataset, runs the evaluation,
    and saves the results to a JSON file.
    """
    client = get_genai_client()  # Initialize the GenAI client

    # Check if _EVAL_DATASET is provided on the command line
    if len(sys.argv) > 1:
        _EVAL_DATASET = sys.argv[1]

    # Load the evaluation dataset
    dataSet = load_evaluation_dataset(model=_EVAL_DATASET, sample_size=_SAMPLE_SIZE)

    # Run evaluation with a specified batch size
    results = await claim_decom_fc_bidirectional.evaluate_dataset_batch(
        dataSet, client, model=_JUDGE_MODEL, batch_size=_BATCH_SIZE
    )

    # Create a result dataframe with claims included
    df_results = claim_decom_fc_bidirectional.create_result_dataframe(
        results, original_dataset=dataSet, include_claims=True
    )

    # Print the first few rows of the results dataframe
    print(df_results.head())

    # Compute mean results for each metric
    mean_results = compute_mean_results(results)
    print("Mean Results:", mean_results)
    print(df_results.describe())
    mean_results_dict = df_results.describe().to_dict()

    # Save detailed results to a JSON file
    full_results = {
        "timestamp": datetime.now().isoformat(),
        "total_rows": len(df_results),
        "results": results,
    }
    # Create the results directory if it does not exist
    results_dir = "./data/results_judge"
    os.makedirs(results_dir, exist_ok=True)

    # Save results to JSON files
    for result_type, data in [
        ("fc_eval_results", full_results),
        ("fc_eval_mean_results", mean_results_dict),
    ]:
        with open(
            f"{results_dir}/{result_type}.{_EVAL_DATASET}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


# Run the evaluation asynchronously
asyncio.run(run_fc_eval())
