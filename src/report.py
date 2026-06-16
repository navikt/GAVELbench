"""Generates all report artifacts: JSON results, radar/bar PNG plots, and static QMD files."""

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
    """Formats a standard Markdown comparison table (metrics as rows, models as columns).

    Returns a list of strings, one per table row.
    """
    models = list(results_by_model.keys())
    all_score_names: list[str] = list(
        dict.fromkeys(s for scores in results_by_model.values() for s in scores)
    )

    def row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    header = row(["Metric"] + models)
    separator = row(["---"] + ["---:" for _ in models])
    pairs = row(
        ["pairs evaluated"] + [f"n={n_pairs_by_model.get(m, '?')}" for m in models]
    )

    lines = [header, separator, pairs]

    for score_name in all_score_names:
        lower_is_better = score_name in LOWER_IS_BETTER
        values = {m: results_by_model[m].get(score_name) for m in models}

        present = {m: v for m, v in values.items() if v is not None}
        best_val = (
            (min(present.values()) if lower_is_better else max(present.values()))
            if present
            else None
        )

        cells = []
        for m in models:
            v = values[m]
            cells.append(
                "N/A" if v is None else f"{v:.4f}" + (" ★" if v == best_val else "")
            )

        label = f"{score_name}{' ↓' if lower_is_better else ''}"
        lines.append(row([label] + cells))

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


def plot_radar_per_overkategori(
    scores_by_overkategori: dict[str, dict[str, dict[str, float]]],
    output_dir: str,
) -> dict[str, str]:
    """Saves one radar plot per overkategori, comparing all models on that category.

    Args:
        scores_by_overkategori: ``{model: {overkategori: {score_name: value}}}``.
        output_dir: Directory where PNG files are written.

    Returns:
        ``{overkategori: plot_path}`` for each generated plot.
    """
    overkategorier = sorted(
        {
            ovk
            for model_scores in scores_by_overkategori.values()
            for ovk in model_scores
        }
    )
    paths: dict[str, str] = {}
    for ovk in overkategorier:
        ovk_results = {
            model: scores_by_overkategori[model][ovk]
            for model in scores_by_overkategori
            if ovk in scores_by_overkategori[model]
        }
        if not ovk_results:
            continue
        safe_name = re.sub(r"[^\w]", "_", ovk).strip("_")
        radar_path = os.path.join(output_dir, f"radar_{safe_name}.png")
        plot_radar(ovk_results, radar_path)
        paths[ovk] = radar_path
    return paths


def plot_bar_charts(
    scores_by_overkategori: dict[str, dict[str, dict[str, float]]],
    output_dir: str,
) -> dict[str, str]:
    """Saves one bar chart PNG per metric comparing models across all overkategorier.

    Args:
        scores_by_overkategori: ``{model: {overkategori: {score_name: value}}}``.
        output_dir: Directory where PNG files are written.

    Returns:
        ``{metric: plot_path}`` for each generated chart.
    """
    models = list(scores_by_overkategori.keys())
    all_overkategorier = sorted(
        {
            ovk
            for model_scores in scores_by_overkategori.values()
            for ovk in model_scores
        }
    )
    all_metrics = list(
        dict.fromkeys(
            m
            for model_scores in scores_by_overkategori.values()
            for ovk_scores in model_scores.values()
            for m in ovk_scores
        )
    )

    paths: dict[str, str] = {}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for metric in all_metrics:
        lower = metric in LOWER_IS_BETTER
        fig, ax = plt.subplots(figsize=(max(8, len(all_overkategorier) * 1.2), 4))
        x = np.arange(len(all_overkategorier))
        width = 0.8 / max(len(models), 1)

        for i, model in enumerate(models):
            vals = [
                scores_by_overkategori.get(model, {})
                .get(ovk, {})
                .get(metric, float("nan"))
                for ovk in all_overkategorier
            ]
            ax.bar(
                x + i * width,
                vals,
                width,
                label=model,
                color=colors[i % len(colors)],
                alpha=0.8,
            )

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(all_overkategorier, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(metric)
        arrow = "↓ lavere er bedre" if lower else "↑ høyere er bedre"
        ax.set_title(f"{metric} per overkategori ({arrow})")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()

        safe_metric = re.sub(r"[^\w]", "_", metric).strip("_")
        bar_path = os.path.join(output_dir, f"bar_{safe_metric}.png")
        fig.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Bar chart saved to {bar_path}")
        paths[metric] = bar_path

    return paths


def write_quarto_index(
    results_by_model: dict[str, dict[str, float]],
    n_pairs_by_model: dict[str, int],
    radar_path: str,
    quarto_dir: str = "quarto",
) -> None:
    """Writes a static ``quarto/index.qmd`` with a results table and radar plot.

    The file contains no Python code blocks — all content is pre-generated.
    Image paths are relative to *quarto_dir*.

    Args:
        results_by_model: ``{model_name: {score_name: value}}``.
        n_pairs_by_model: ``{model_name: n_pairs}``.
        radar_path: Absolute or repo-relative path to the overall radar PNG.
        quarto_dir: Directory where ``index.qmd`` will be written.
    """
    models = list(results_by_model.keys())
    metrics = list(
        dict.fromkeys(s for scores in results_by_model.values() for s in scores)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Markdown table
    def _row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    header = _row(["Metrikk"] + models)
    sep = _row(["---"] + ["---:" for _ in models])
    n_row = _row(["Antall par"] + [f"n={n_pairs_by_model.get(m, '?')}" for m in models])
    table_lines = [header, sep, n_row]
    for metric in metrics:
        lower = metric in LOWER_IS_BETTER
        values = {m: results_by_model[m].get(metric) for m in models}
        present = {m: v for m, v in values.items() if v is not None}
        best = (
            (
                min(present, key=present.__getitem__)
                if lower
                else max(present, key=present.__getitem__)
            )
            if present
            else None
        )
        cells = [
            "N/A"
            if values[m] is None
            else f"{values[m]:.4f}" + (" ★" if m == best else "")
            for m in models
        ]
        label = metric + (" ↓" if lower else "")
        table_lines.append(_row([label] + cells))

    # Image path relative to quarto_dir
    rel_radar = os.path.relpath(radar_path, quarto_dir)

    lines = [
        "---",
        "# Denne filen er autogenerert av src/report.py — ikke rediger manuelt",
        "---",
        "",
        "# Evaluering av språkmodeller opp mot Bob-svar",
        "",
        f"_Generert: {timestamp}_",
        "",
        "Denne rapporten sammenligner språkmodeller mot svar fra Bob på tvers av ulike målemetoder.",
        "",
        "## Resultater",
        "",
    ]
    lines += table_lines
    lines += [
        "",
        "## Radarplot",
        "",
        f"![Modellsammenligning (absolutt skala, høyere = bedre)]({rel_radar})",
        "",
    ]

    out_path = os.path.join(quarto_dir, "index.qmd")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Quarto index written to {out_path}")


def write_quarto_per_kategori(
    scores_by_overkategori: dict[str, dict[str, dict[str, float]]],
    n_pairs_by_overkategori: dict[str, dict[str, int]],
    bar_chart_paths: dict[str, str],
    radar_paths: dict[str, str],
    quarto_dir: str = "quarto",
) -> None:
    """Writes a static ``quarto/per_kategori.qmd`` with per-category tables and plots.

    The file contains no Python code blocks — all content is pre-generated.
    Image paths are relative to *quarto_dir*.

    Args:
        scores_by_overkategori: ``{model: {overkategori: {score: value}}}``.
        n_pairs_by_overkategori: ``{model: {overkategori: n_pairs}}``.
        bar_chart_paths: ``{metric: path}`` from :func:`plot_bar_charts`.
        radar_paths: ``{overkategori: path}`` from :func:`plot_radar_per_overkategori`.
        quarto_dir: Directory where ``per_kategori.qmd`` will be written.
    """
    models = list(scores_by_overkategori.keys())
    all_overkategorier = sorted(
        {ovk for m_scores in scores_by_overkategori.values() for ovk in m_scores}
    )
    all_metrics = list(
        dict.fromkeys(
            metric
            for m_scores in scores_by_overkategori.values()
            for ovk_scores in m_scores.values()
            for metric in ovk_scores
        )
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    lines = [
        "---",
        "# Denne filen er autogenerert av src/report.py — ikke rediger manuelt",
        "---",
        "",
        "# Resultater per kategori",
        "",
        f"_Generert: {timestamp}_",
        "",
        "Denne siden viser evalueringsresultater gruppert etter **overkategori**.",
        "Spørsmål kan tilhøre flere overkategorier, og inngår i beregningene for hver av dem.",
        "",
        "## Tabeller per metrikk",
        "",
    ]

    for metric in all_metrics:
        lower = metric in LOWER_IS_BETTER
        label = metric + (" (↓ lavere er bedre)" if lower else "")
        lines.append(f"### {label}")
        lines.append("")
        header = _row(["Overkategori"] + models)
        sep = _row(["---"] + ["---:" for _ in models])
        lines += [header, sep]
        for ovk in all_overkategorier:
            values = {
                m: scores_by_overkategori.get(m, {}).get(ovk, {}).get(metric)
                for m in models
            }
            present = {m: v for m, v in values.items() if v is not None}
            best = (
                (
                    min(present, key=present.__getitem__)
                    if lower
                    else max(present, key=present.__getitem__)
                )
                if present
                else None
            )
            cells = []
            for m in models:
                v = values[m]
                n = n_pairs_by_overkategori.get(m, {}).get(ovk, "")
                if v is None:
                    cells.append("–")
                else:
                    star = " ★" if m == best else ""
                    cells.append(f"{v:.4f} (n={n}){star}")
            lines.append(_row([ovk] + cells))
        lines.append("")

    lines += ["## Søylediagram per overkategori", ""]
    for metric, bar_path in sorted(bar_chart_paths.items()):
        rel = os.path.relpath(bar_path, quarto_dir)
        lower = metric in LOWER_IS_BETTER
        arrow = "↓ lavere er bedre" if lower else "↑ høyere er bedre"
        lines.append(f"### {metric} ({arrow})")
        lines.append("")
        lines.append(f"![]({rel})")
        lines.append("")

    lines += ["## Radarplott per overkategori", ""]
    for ovk in all_overkategorier:
        lines.append(f"### {ovk}")
        lines.append("")
        if ovk in radar_paths:
            rel = os.path.relpath(radar_paths[ovk], quarto_dir)
            lines.append(f"![]({rel})")
        else:
            lines.append(f"_(Ingen radarplott funnet for {ovk})_")
        lines.append("")

    out_path = os.path.join(quarto_dir, "per_kategori.qmd")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Quarto per_kategori written to {out_path}")


def write_report(
    results_by_model: dict[str, dict[str, float]],
    n_pairs_by_model: dict[str, int],
    output_path: str,
    scores_by_overkategori: dict[str, dict[str, dict[str, float]]] | None = None,
    n_pairs_by_overkategori: dict[str, dict[str, int]] | None = None,
    quarto_dir: str = "quarto",
) -> None:
    """Writes all report artifacts: JSON results, radar/bar PNGs, and static QMD files.

    Args:
        results_by_model: ``{model_name: {score_name: value}}``.
        n_pairs_by_model: ``{model_name: n_pairs}``.
        output_path: Destination path for the JSON results file (and legacy MD report).
        scores_by_overkategori: Optional ``{model: {overkategori: {score: value}}}``.
        n_pairs_by_overkategori: Optional ``{model: {overkategori: n_pairs}}``.
        quarto_dir: Directory where the static QMD files are written.
    """
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Overall radar plot
    radar_path = output_path.replace(".md", "_radar.png").replace(".json", "_radar.png")
    if not radar_path.endswith("_radar.png"):
        radar_path = os.path.join(output_dir, "evaluation_report_radar.png")
    plot_radar(results_by_model, radar_path)

    # Per-overkategori radar + bar charts
    ovk_radar_paths: dict[str, str] = {}
    bar_chart_paths: dict[str, str] = {}
    if scores_by_overkategori:
        ovk_radar_paths = plot_radar_per_overkategori(
            scores_by_overkategori, output_dir
        )
        bar_chart_paths = plot_bar_charts(scores_by_overkategori, output_dir)

        # Write per-overkategori scores to JSON
        ovk_json_path = os.path.join(
            output_dir, "evaluation_report_scores_per_overkategori.json"
        )
        with open(ovk_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"scores": scores_by_overkategori, "n_pairs": n_pairs_by_overkategori},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Per-overkategori scores written to {ovk_json_path}")

    # Write aggregate results to JSON
    json_path = os.path.join(output_dir, "evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"models": results_by_model, "n_pairs": n_pairs_by_model},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Results written to {json_path}")

    # Write static QMD files
    write_quarto_index(results_by_model, n_pairs_by_model, radar_path, quarto_dir)
    if scores_by_overkategori and n_pairs_by_overkategori:
        write_quarto_per_kategori(
            scores_by_overkategori,
            n_pairs_by_overkategori,
            bar_chart_paths,
            ovk_radar_paths,
            quarto_dir,
        )


def regenerate_report(
    results_json_path: str,
    output_path: str,
    ovk_json_path: str | None = None,
    models_yaml_path: str | None = None,
    quarto_dir: str = "quarto",
) -> None:
    """Regenerates all report artifacts from saved JSON results.

    Loads the aggregate scores written by a previous evaluation run and calls
    :func:`write_report` so that all plots and static QMD files are recreated
    without re-running any model evaluation.  When *models_yaml_path* is
    provided, only models listed in that config are included.

    Args:
        results_json_path: Path to ``evaluation_report.json``.
        output_path: Destination path for the JSON output (used for radar PNG naming).
        ovk_json_path: Optional path to
            ``evaluation_report_scores_per_overkategori.json``.
        models_yaml_path: Optional path to ``models.yaml``.  When provided,
            results are filtered to only include models defined there.
        quarto_dir: Directory where the static QMD files are written.
    """
    from models import active_model_ids

    with open(results_json_path, encoding="utf-8") as f:
        data = json.load(f)

    results_by_model: dict[str, dict[str, float]] = data["models"]
    n_pairs_by_model: dict[str, int] = data["n_pairs"]

    if models_yaml_path and os.path.exists(models_yaml_path):
        active = active_model_ids(models_yaml_path)
        skipped = [m for m in results_by_model if m not in active]
        if skipped:
            print(f"Skipping models not in {models_yaml_path}: {skipped}")
        results_by_model = {m: v for m, v in results_by_model.items() if m in active}
        n_pairs_by_model = {m: v for m, v in n_pairs_by_model.items() if m in active}

    scores_by_overkategori = None
    n_pairs_by_overkategori = None
    if ovk_json_path and os.path.exists(ovk_json_path):
        with open(ovk_json_path, encoding="utf-8") as f:
            ovk_data = json.load(f)
        scores_by_overkategori = ovk_data.get("scores")
        n_pairs_by_overkategori = ovk_data.get("n_pairs")
        if (
            scores_by_overkategori
            and models_yaml_path
            and os.path.exists(models_yaml_path)
        ):
            scores_by_overkategori = {
                m: v for m, v in scores_by_overkategori.items() if m in active
            }
            n_pairs_by_overkategori = {
                m: v for m, v in (n_pairs_by_overkategori or {}).items() if m in active
            } or None

    write_report(
        results_by_model,
        n_pairs_by_model,
        output_path,
        scores_by_overkategori=scores_by_overkategori,
        n_pairs_by_overkategori=n_pairs_by_overkategori,
        quarto_dir=quarto_dir,
    )
