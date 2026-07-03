"""tabulate_category_results_fc.py.

This module provides functionality to extract F1 scores from JSON files
and display them in a formatted table.

It includes the following functions:
- extract_f1_scores: Reads JSON files matching a specified pattern and
  extracts F1 scores per category.
- print_table: Prints the extracted F1 scores in a formatted table,
  organized by categories and datasets.

Usage:
    If run as a script, it will extract F1 scores from JSON files in the
    specified directory and print them in a table format.
"""

import glob
import json
import os
from typing import Dict


def extract_f1_scores(
    pattern: str = "*eval_overkategori*.json",
) -> Dict[str, Dict[str, float]]:
    """Reads all JSON files matching pattern and extracts f1_score per category.

    Returns:
        { "Category Name": { "dataset_name": f1_score } }

    Example output:
        {
            "Familie og barn": {"norallm__normistral": 0.5036},
            "Pensjon": {"norallm__normistral": 0.5128}
        }
    """
    # Organized as: Category -> { Dataset: f1_score }
    category_table: Dict[str, Dict[str, float]] = {}

    files = glob.glob(os.path.join(os.getcwd(), pattern))

    if not files:
        print(f"⚠️ No files found matching '{pattern}'")
        return category_table

    print(f"📂 Found {len(files)} files:")

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)

            dataset = content.get(
                "dataset", os.path.basename(file_path).replace(".json", "")
            )
            metrics = content.get("metrics_by_category", {})

            for category, cat_metrics in metrics.items():
                f1_score = cat_metrics.get("f1_score")

                if isinstance(f1_score, (int, float)):
                    # Initialize category dict if first time seeing it
                    if category not in category_table:
                        category_table[category] = {}

                    category_table[category][dataset] = round(f1_score, 4)

            print(f"   ✓ {os.path.basename(file_path)} ({dataset})")

        except json.JSONDecodeError:
            print(f"❌ Invalid JSON: {file_path}")
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    return category_table


def print_table(category_data: Dict[str, Dict[str, float]], precision: int = 4) -> None:
    """Prints the data as a formatted table.

    Parameters
    ----------
    category_data : Dict[str, Dict[str, float]]
        A dictionary where keys are category names and values are dictionaries
        containing dataset names as keys and their corresponding F1-scores as values.

    precision : int, optional
        The number of decimal places to display for the F1-scores (default is 4).

    Returns:
    -------
    None
        This function prints the formatted table directly to the console.

    Notes:
    -----
    If `category_data` is empty, a message indicating no data will be displayed.
    """
    if not category_data:
        print("No data to display.")
        return

    categories = list(category_data.keys())
    datasets: set[str] = set()
    for cat_dict in category_data.values():
        datasets.update(cat_dict.keys())

    sorted_datasets = sorted(datasets)

    # Print header
    print("\n" + "=" * 80)
    print("F1-Score by Category")
    print("=" * 80)

    # Header row
    header = "Category".ljust(30)
    for ds in sorted_datasets:
        short_ds = ds[:15] + "..." if len(ds) > 15 else ds
        header += f"{short_ds:>12}"
    print(header)
    print("-" * 80)

    # Data rows
    for cat in sorted(categories):
        row = cat.ljust(30)
        for ds in sorted_datasets:
            score = category_data[cat].get(ds)
            if score is not None:
                row += f"{score:12.{precision}f}"
            else:
                row += f"{'N/A':>12}"
        print(row)

    print("=" * 80)


# ==================== Usage ====================

if __name__ == "__main__":
    # 1. Extract scores
    data = extract_f1_scores("data/results_judge/*eval_overkategori*.json")

    # 2. Print formatted table
    print_table(data)

    # 3. Optional: Access programmatically
    # Example: Get all F1 scores for "Pensjon"
    pensjon_scores = data.get("Pensjon", {})
    print(f"\n💡 Pensjon scores: {pensjon_scores}")

    # Example: Average F1 across all categories
    avg_per_cat = {
        cat: sum(scores.values()) / len(scores)
        for cat, scores in data.items()
        if scores
    }
    print("\n📊 Average F1 per category:")
    for cat, avg in sorted(avg_per_cat.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat}: {avg:.4f}")
