"""Leaderboard Scoring.

A modular, configuration-driven system for evaluating Large Language Models
(LLMs) across multiple performance dimensions with safety gates and composite
scoring.

Core Features
-------------
- Multi-metric evaluation (ROUGE, BERTScore, semantic similarity, JSD, NLI, etc.)
- Configurable safety thresholds (hard gates for factuality, quality, divergence)
- Multiple scoring methods (geometric mean, weighted average)
- Support for both dictionary inputs and file-based workflows
- Clean separation of concerns (config → mapping → scoring → output)

How To Use
----------

Extending Metrics
^^^^^^^^^^^^^^^^^
To add a new metric (e.g., `perplexity`), simply:
    1. Add an entry in `get_default_metric_map()` under the appropriate key
    2. Update `evaluate_single_model()` to call `extract_metric()` for the new metric
    No need to rewrite the entire loop or modify scoring logic.

Changing Logic
^^^^^^^^^^^^^^
Switch scoring methods by updating the `scoring_method` parameter in
`create_pipeline_config()`:

    # Geometric mean (default, more conservative)
    config = create_pipeline_config(scoring_method="geometric_mean")

    # Weighted arithmetic average (more linear weighting)
    config = create_pipeline_config(scoring_method="weighted_avg")

External Configuration
^^^^^^^^^^^^^^^^^^^^^^
Replace manual config creation with external JSON/YAML files for production:

    with open('config.json', 'r') as f:
        config_dict = json.load(f)

    config = create_pipeline_config(
        weights=config_dict['weights'],
        thresholds=config_dict['thresholds'],
        metric_map=config_dict.get('metric_map'),
        scoring_method=config_dict.get('scoring_method', 'geometric_mean')
    )

Testing Guidance
^^^^^^^^^^^^^^^^
Since `evaluate_single_model()` is a pure function, you can unit test it
directly without file setup or complex pipeline initialization:

    def test_factual_failure():
        raw_data = {'nli_entailment': 0.3}  # Below threshold
        metrics, status = evaluate_single_model(raw_data, config)
        assert status == "FAIL_FACTUALITY"

Input Data Format
^^^^^^^^^^^^^^^^^
The main function expects a JSON structure with a 'models' key:

    {
      "models": {
        "model-name": {
          "rouge_l": 0.85,
          "bertscore_f1": 0.72,
          "cosine_similarity": 0.68,
          "jsd": 0.45,
          "nli_entailment": 0.78
        },
        ...
      }
    }

Pass this dictionary directly to `process_leaderboard_data(models, config)`
or load via `json.loads()` / `Path('file.json').read_text()`.

Key Architectural Benefits
^^^^^^^^^^^^^^^^^^^^^^^^^^
| Aspect              | Benefit                                            |
|---------------------|----------------------------------------------------|
| Config-Driven       | Change metrics/thresholds without modifying code   |
| Pure Functions      | Easy to test, reason about, and reuse              |
| Flexible Mapping    | Map multiple source keys to same canonical metric  |
| Modular Output      | Swap display/export formats independently          |
| Safety First        | Hard failure gates prevent unsafe models scoring   |

Export Options
^^^^^^^^^^^^^^
Uncomment export line at end of script to save results:

    df_result.drop(columns=['raw_snippet']).to_csv("leaderboard.csv", index=False)

Dependencies
------------
- pandas >= 1.3.0
- numpy >= 1.20.0
"""

import json
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def get_default_metric_map() -> Dict[str, list[str]]:
    """Returns default metric mapping configuration."""
    return {
        "rouge_l": ["rouge_l", "rouge_score"],
        "semantic_sim": ["bertscore_f1", "cosine_similarity", "semantic_score"],
        "factual_corr": ["nli_entailment", "factual_correctness", "fc_high"],
        "answer_quality": ["answer_quality", "quality_score", "nli_entailment"],
        "jsd_divergence": ["jsd", "kl_divergence", "distribution_diff"],
    }


def get_default_weights() -> Dict[str, float]:
    """Returns default weighting configuration."""
    return {
        "rouge_l": 0.15,
        "semantic_sim": 0.25,
        "factual_corr": 0.40,
        "answer_quality": 0.40,
    }


def get_default_thresholds() -> Dict[str, float]:
    """Returns default safety threshold configuration."""
    return {
        "factual_corr": 0.20,  # Minimum required
        "answer_quality": 0.20,  # Minimum required
        "jsd_divergence": 0.90,  # Maximum allowed (lower is better)
    }


def create_pipeline_config(
    metric_map: Optional[Dict[str, list[Any]]] = None,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    scoring_method: str = "geometric_mean",
) -> Dict[str, Any]:
    """Creates a complete pipeline configuration dictionary."""
    return {
        "metric_map": metric_map if metric_map else get_default_metric_map(),
        "weights": weights if weights else get_default_weights(),
        "thresholds": thresholds if thresholds else get_default_thresholds(),
        "scoring_method": scoring_method,
    }


def normalize_value(
    value: Optional[float], min_val: float = 0.0, max_val: float = 1.0
) -> float:
    """Safely normalizes a value to [min, max]."""
    if value is None:
        return 0.0
    try:
        v = float(value)
        return float(np.clip(v, min_val, max_val))
    except ValueError, TypeError:
        return 0.0


def extract_metric(
    data: Dict[str, Any], possible_keys: list[str], default: float = 0.0
) -> float:
    """Finds the first matching key in the data dict."""
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
    """Maps raw data to canonical metrics and calculates score."""
    # 1. Map Metrics
    mapped = {}
    mm = config["metric_map"]

    mapped["rouge_l"] = extract_metric(raw_data, mm.get("rouge_l", []))
    mapped["semantic_sim"] = extract_metric(raw_data, mm.get("semantic_sim", []))
    mapped["factual_corr"] = extract_metric(raw_data, mm.get("factual_corr", []))
    mapped["answer_quality"] = extract_metric(raw_data, mm.get("answer_quality", []))
    mapped["jsd_divergence"] = extract_metric(raw_data, mm.get("jsd_divergence", []))

    # 2. Check Safety Gates
    thresholds = config["thresholds"]

    # Factual/Quality Minimums
    if mapped["factual_corr"] < thresholds.get("factual_corr", 0.0):
        return mapped, "FAIL_FACTUALITY"
    if mapped["answer_quality"] < thresholds.get("answer_quality", 0.0):
        return mapped, "FAIL_QUALITY"

    # Divergence Maximums (JSD: High is bad)
    if mapped["jsd_divergence"] > thresholds.get("jsd_divergence", 1.0):
        return mapped, "FAIL_DIVERGENCE"

    # 3. Calculate Composite Score
    weights = config["weights"]
    total_weight = sum(weights.values())

    if total_weight == 0:
        return mapped, "ERROR_NO_WEIGHTS"

    score = 0.0

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
        product_term = 1.0
        count = 0
        for m_name, weight in weights.items():
            if weight <= 0:
                continue
            val = mapped.get(m_name, 0)
            if m_name == "jsd_divergence":
                val = 1.0 - val

            safe_val = max(val, 1e-9)
            product_term *= safe_val**weight
            count += 1

        score = product_term ** (1.0 / count) if count > 0 else 0.0

    mapped["composite_score"] = score
    return mapped, "PASS"


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
    if not df.empty and "composite_score" in df.columns:
        df = df.sort_values(by="composite_score", ascending=False).reset_index(
            drop=True
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


def print_analysis_summary(df: pd.DataFrame) -> None:
    """Prints summary statistics."""
    if df.empty:
        print("\n⚠️  No data to analyze.")
        return

    total = len(df)
    passed = (df["status"] == "PASS").sum()
    failed = total - passed

    print("\n📈 ANALYSIS SUMMARY:")
    print(f"   Total Models:          {total}")
    print(f"   Passed Safety Gates:   {passed} ({passed / total * 100:.1f}%)")
    print(f"   Failed Safety Gates:   {failed} ({failed / total * 100:.1f}%)")

    if df[df["status"] == "PASS"].empty:
        print("   ⚠️  WARNING: No models qualified!")
    else:
        top = df[df["status"] == "PASS"].iloc[0]
        print(
            f"   🏆 Top Model:          {top['model_id']} (Score: {top['composite_score']:.4f})"
        )


if __name__ == "__main__":
    input_json_str = """
    {
      "models": {
        "claude-sonnet-4-6": {
          "rouge_l": 0.22275420744483693,
          "bertscore_f1": 0.700186014175415,
          "cosine_similarity": 0.3403572994488996,
          "jsd": 0.6687984612483253,
          "nli_entailment": 0.34285914041101934
        },
        "gemini-2.0-flash": {
          "rouge_l": 0.2260329115930793,
          "bertscore_f1": 0.7045748233795166,
          "cosine_similarity": 0.3335971053590159,
          "jsd": 0.6475931296194044,
          "nli_entailment": 0.34024275706421275
        },
        "gemini-2.5-flash": {
          "rouge_l": 0.24935984848586362,
          "bertscore_f1": 0.7104960083961487,
          "cosine_similarity": 0.3573922653998569,
          "jsd": 0.638843254061178,
          "nli_entailment": 0.34120670539559794
        }
      }
    }
    """

    filename = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        if filename:
            with open(filename, "r") as f:
                json_data = json.load(f)
        else:
            json_data = json.loads(input_json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON input: {e}")
        exit(1)

    config = create_pipeline_config(
        weights={
            "rouge_l": 0.15,
            "semantic_sim": 0.25,
            "factual_corr": 0.40,
            "answer_quality": 0.20,
        },
        thresholds={
            "factual_corr": 0.10,
            "answer_quality": 0.10,
            "jsd_divergence": 0.90,
        },
        scoring_method="geometric_mean",
    )

    models = json_data.get("models", {})
    df_result = process_leaderboard_data(models, config)

    print(format_table(df_result))
    print_analysis_summary(df_result)

    df_result.drop(columns=["raw_snippet"]).to_csv(
        "leaderboard_export.csv", index=False
    )
