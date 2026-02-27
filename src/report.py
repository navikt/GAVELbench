"""Formats and writes evaluation reports."""

import json
import os
import re
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

# Metrics where a lower value is better.
LOWER_IS_BETTER: set[str] = {"jsd"}


def build_commentary(results_by_model: dict[str, dict[str, float]]) -> list[str]:
    """Generates human-readable commentary comparing models across metrics.

    Args:
        results_by_model: ``{model_name: {score_name: value}}``.

    Returns:
        List of comment strings, one per metric plus a summary sentence.
        Empty list when fewer than two models are present.
    """
    if len(results_by_model) < 2:
        return []

    models = list(results_by_model.keys())
    all_score_names: list[str] = list(
        dict.fromkeys(s for scores in results_by_model.values() for s in scores)
    )

    lines: list[str] = []
    best_counts: dict[str, int] = {m: 0 for m in models}

    for score_name in all_score_names:
        lower_is_better = score_name in LOWER_IS_BETTER
        values = {
            m: results_by_model[m][score_name]
            for m in models
            if score_name in results_by_model[m]
        }
        if not values:
            continue

        best_model = (
            min(values, key=values.__getitem__)
            if lower_is_better
            else max(values, key=values.__getitem__)
        )
        worst_model = (
            max(values, key=values.__getitem__)
            if lower_is_better
            else min(values, key=values.__getitem__)
        )
        best_counts[best_model] += 1

        direction = "lower" if lower_is_better else "higher"
        delta = abs(values[best_model] - values[worst_model])
        lines.append(
            f"- **{score_name}**: `{best_model}` scores best ({values[best_model]:.4f}),"
            f" `{worst_model}` scores worst ({values[worst_model]:.4f})."
            f" Δ={delta:.4f} ({direction} is better)."
        )

    overall_winner = max(best_counts, key=best_counts.__getitem__)
    lines.append(
        f"\n**Overall**: `{overall_winner}` leads on {best_counts[overall_winner]}"
        f" of {len(all_score_names)} metric(s)."
    )
    return lines


def format_table(
    results_by_model: dict[str, dict[str, float]],
    n_pairs_by_model: dict[str, int],
) -> list[str]:
    """Formats a comparison table (metrics as rows, models as columns).

    Returns a list of strings, one per table row, suitable for both
    terminal output and Markdown files.
    """
    models = list(results_by_model.keys())
    all_score_names: list[str] = list(
        dict.fromkeys(s for scores in results_by_model.values() for s in scores)
    )

    col_metric = max(len(s) for s in all_score_names + ["Metric"]) + 2
    col_model = max((len(m) for m in models), default=8) + 2
    col_model = max(col_model, 8)

    def sep_row() -> str:
        return (
            "+"
            + "-" * (col_metric + 2)
            + "+"
            + ("+".join(["-" * (col_model + 2)] * len(models)))
            + "+"
        )

    def header_row() -> str:
        cells = "".join(f" {m:<{col_model}} |" for m in models)
        return f"| {'Metric':<{col_metric}} |{cells}"

    def pairs_row() -> str:
        cells = "".join(
            f" {'n=' + str(n_pairs_by_model.get(m, '?')):<{col_model}} |"
            for m in models
        )
        return f"| {'pairs evaluated':<{col_metric}} |{cells}"

    lines = [sep_row(), header_row(), sep_row(), pairs_row(), sep_row()]

    for score_name in all_score_names:
        lower_is_better = score_name in LOWER_IS_BETTER
        values = {m: results_by_model[m].get(score_name) for m in models}

        present = {m: v for m, v in values.items() if v is not None}
        best_val = (
            (min(present.values()) if lower_is_better else max(present.values()))
            if present
            else None
        )

        cells = ""
        for m in models:
            v = values[m]
            cell = "N/A" if v is None else f"{v:.4f}" + (" ★" if v == best_val else "")
            cells += f" {cell:<{col_model}} |"

        label = f"{score_name}{' ↓' if lower_is_better else ''}"
        lines.append(f"| {label:<{col_metric}} |{cells}")

    lines.append(sep_row())
    return lines


def print_report(
    results_by_model: dict[str, dict[str, float]],
    n_pairs_by_model: dict[str, int],
) -> None:
    """Prints the comparison table and commentary to stdout."""
    print("\n=== GAVELbench Evaluation Report ===\n")
    for line in format_table(results_by_model, n_pairs_by_model):
        print(line)
    commentary = build_commentary(results_by_model)
    if commentary:
        print("\nCommentary:")
        for c in commentary:
            print(re.sub(r"\*\*(.+?)\*\*", r"\1", c).replace("`", ""))
    print()


def plot_radar(
    results_by_model: dict[str, dict[str, float]],
    output_path: str,
) -> None:
    """Saves a radar plot comparing all models across all metrics.

    Lower-is-better metrics are inverted so that outward always means better.
    All metrics are assumed to lie in [0, 1], so each axis uses that absolute
    range — this keeps all models visible even when one dominates.

    Args:
        results_by_model: ``{model_name: {score_name: value}}``.
        output_path: Destination path for the PNG image.
    """
    score_names: list[str] = list(
        dict.fromkeys(s for scores in results_by_model.values() for s in scores)
    )
    models = list(results_by_model.keys())
    n = len(score_names)

    # Invert lower-is-better metrics so outward = better on every axis.
    values_matrix: np.ndarray = np.array(
        [
            [
                (1.0 - results_by_model[m].get(s, 0.0))
                if s in LOWER_IS_BETTER
                else results_by_model[m].get(s, 0.0)
                for s in score_names
            ]
            for m in models
        ],
        dtype=float,
    )

    # Axis angles — evenly spaced, closing the polygon.
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, model in enumerate(models):
        vals = values_matrix[i].tolist() + values_matrix[i][:1].tolist()
        color = colors[i % len(colors)]
        ax.plot(angles, vals, linewidth=2, linestyle="solid", label=model, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    labels = [f"{s}\n(↓ inverted)" if s in LOWER_IS_BETTER else s for s in score_names]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"], size=7, color="grey")
    ax.set_ylim(0, 1)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title("Model comparison (absolute scale, higher = better)", pad=20, size=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Radar plot saved to {output_path}")


def write_report(
    results_by_model: dict[str, dict[str, float]],
    n_pairs_by_model: dict[str, int],
    output_path: str,
) -> None:
    """Writes a Markdown report and a JSON results file.

    The JSON file is written alongside the Markdown file with the same
    base name and a ``.json`` extension.

    Args:
        results_by_model: ``{model_name: {score_name: value}}``.
        n_pairs_by_model: ``{model_name: n_pairs}``.
        output_path: Destination path for the Markdown report.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    radar_path = output_path.replace(".md", "_radar.png")
    plot_radar(results_by_model, radar_path)

    lines: list[str] = [
        "# GAVELbench Evaluation Report",
        f"\n_Generated: {timestamp}_\n",
        "## Results\n",
    ]
    lines += format_table(results_by_model, n_pairs_by_model)

    # Embed radar plot using a relative path so the Markdown renders locally.
    lines += [
        "\n## Radar Plot\n",
        f"![Radar plot]({os.path.basename(radar_path)})\n",
    ]

    commentary = build_commentary(results_by_model)
    if commentary:
        lines += ["\n## Commentary\n"] + commentary

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {output_path}")

    json_path = output_path.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(
            {"models": results_by_model, "n_pairs": n_pairs_by_model}, f, indent=2
        )
    print(f"Raw results written to {json_path}")
