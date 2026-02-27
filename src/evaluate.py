"""Evaluates the generated answers against the ground truth from Bob based on various metrics."""

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Callable

import numpy as np
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# A metric function takes (references, hypotheses) and returns a dict of score_name -> float.
MetricFn = Callable[[list[str], list[str]], dict[str, float]]

_METRICS: dict[str, MetricFn] = {}

# Metrics where a lower value is better (used in commentary).
_LOWER_IS_BETTER: set[str] = {"jsd"}


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator to register a metric function by name."""

    def decorator(fn: MetricFn) -> MetricFn:
        _METRICS[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


@register_metric("rouge_l")
def compute_rouge_l(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Computes mean ROUGE-L F1 over all reference/hypothesis pairs."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(references, hypotheses)
    ]
    return {"rouge_l": float(np.mean(scores))}


@register_metric("bertscore")
def compute_bertscore(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Computes mean BERTScore F1 using a multilingual model."""
    _, _, f1 = bert_score_fn(
        hypotheses,
        references,
        model_type="bert-base-multilingual-cased",
        verbose=False,
    )
    return {"bertscore_f1": float(f1.mean())}


@register_metric("cosine_similarity")
def compute_cosine_similarity(
    references: list[str], hypotheses: list[str]
) -> dict[str, float]:
    """Computes mean pairwise TF-IDF cosine similarity."""
    vectorizer = TfidfVectorizer()
    all_texts = references + hypotheses
    tfidf = vectorizer.fit_transform(all_texts)
    n = len(references)
    ref_vecs = tfidf[:n]
    hyp_vecs = tfidf[n:]
    scores = [float(sklearn_cosine(ref_vecs[i], hyp_vecs[i])[0, 0]) for i in range(n)]
    return {"cosine_similarity": float(np.mean(scores))}


def _text_to_distribution(text: str, vocab: list[str]) -> np.ndarray:
    """Converts text to a smoothed word-frequency distribution over a shared vocab."""
    counts = Counter(text.lower().split())
    # Laplace smoothing to avoid zero-probability entries
    dist = np.array([counts.get(w, 0) + 1e-10 for w in vocab], dtype=float)
    return dist / dist.sum()


@register_metric("jsd")
def compute_jsd(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Computes mean Jensen-Shannon Divergence over shared vocabulary distributions."""
    vocab = list({w for text in references + hypotheses for w in text.lower().split()})
    scores = [
        float(
            jensenshannon(
                _text_to_distribution(ref, vocab), _text_to_distribution(hyp, vocab)
            )
        )
        for ref, hyp in zip(references, hypotheses)
    ]
    return {"jsd": float(np.mean(scores))}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict[str, str]]:
    """Loads a JSONL file and returns a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def model_name_from_path(path: str) -> str:
    """Derives a short model label from a generated-answers file path.

    Expects filenames like ``generated_answers_<model>.jsonl``.
    Falls back to the bare filename stem.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r"generated_answers_(.*)", stem)
    return match.group(1) if match else stem


def align_pairs(
    bob_records: list[dict[str, str]], generated_records: list[dict[str, str]]
) -> tuple[list[str], list[str]]:
    """Aligns bob answers with generated answers by question text.

    Returns (references, hypotheses) in matched order.
    """
    generated_by_question = {r["question"]: r["answer"] for r in generated_records}
    references, hypotheses = [], []
    for record in bob_records:
        question = record["contextualized_question"]
        if question in generated_by_question:
            references.append(record["answer_content"])
            hypotheses.append(generated_by_question[question])
    return references, hypotheses


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

# Metrics where lower is better get a downward arrow annotation in the header.
_DIRECTION_NOTE = "(↓ lower is better)"


def _build_commentary(results_by_model: dict[str, dict[str, float]]) -> list[str]:
    """Generates human-readable commentary comparing models across metrics.

    Args:
        results_by_model: ``{model_name: {score_name: value}}``.

    Returns:
        List of comment strings, one per metric plus a summary sentence.
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
        lower_is_better = score_name in _LOWER_IS_BETTER
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

    # Overall winner
    overall_winner = max(best_counts, key=best_counts.__getitem__)
    lines.append(
        f"\n**Overall**: `{overall_winner}` leads on {best_counts[overall_winner]}"
        f" of {len(all_score_names)} metric(s)."
    )
    return lines


def _format_table(
    results_by_model: dict[str, dict[str, float]],
    n_pairs_by_model: dict[str, int],
) -> list[str]:
    """Formats a Markdown comparison table (metrics as rows, models as columns)."""
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
        lower_is_better = score_name in _LOWER_IS_BETTER
        values = {m: results_by_model[m].get(score_name) for m in models}

        # Identify best value for highlighting
        present = {m: v for m, v in values.items() if v is not None}
        best_val = (
            (min(present.values()) if lower_is_better else max(present.values()))
            if present
            else None
        )

        cells = ""
        for m in models:
            v = values[m]
            if v is None:
                cell = "N/A"
            else:
                cell = f"{v:.4f}" + (" ★" if v == best_val else "")
            cells += f" {cell:<{col_model}} |"

        direction = " ↓" if lower_is_better else ""
        label = f"{score_name}{direction}"
        lines.append(f"| {label:<{col_metric}} |{cells}")

    lines.append(sep_row())
    return lines


def _write_report(
    results_by_model: dict[str, dict[str, float]],
    n_pairs_by_model: dict[str, int],
    output_path: str,
) -> None:
    """Writes a Markdown evaluation report to *output_path*."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# GAVELbench Evaluation Report",
        f"\n_Generated: {timestamp}_\n",
        "## Results\n",
    ]
    lines += _format_table(results_by_model, n_pairs_by_model)

    commentary = _build_commentary(results_by_model)
    if commentary:
        lines += ["\n## Commentary\n"] + commentary

    report_text = "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"\nReport written to {output_path}")


def _print_table(
    results_by_model: dict[str, dict[str, float]],
    n_pairs_by_model: dict[str, int],
) -> None:
    """Prints the comparison table to stdout."""
    print("\n=== GAVELbench Evaluation Report ===\n")
    for line in _format_table(results_by_model, n_pairs_by_model):
        print(line)
    commentary = _build_commentary(results_by_model)
    if commentary:
        print("\nCommentary:")
        for c in commentary:
            # Strip markdown bold markers for plain terminal output
            print(re.sub(r"\*\*(.+?)\*\*", r"\1", c).replace("`", ""))
    print()


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def _run_metrics_for_model(
    references: list[str],
    hypotheses: list[str],
    metrics: list[str],
    model_label: str,
) -> dict[str, float]:
    """Runs all requested metrics for a single model and returns score dict."""
    scores: dict[str, float] = {}
    for metric_name in metrics:
        if metric_name not in _METRICS:
            raise ValueError(
                f"Unknown metric '{metric_name}'. Available: {list(_METRICS)}"
            )
        print(f"  [{model_label}] Computing {metric_name}...")
        scores.update(_METRICS[metric_name](references, hypotheses))
    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    bob_path: str,
    generated_paths: list[str],
    metrics: list[str] | None = None,
    output_path: str = "data/evaluation_report.md",
) -> dict[str, dict[str, float]]:
    """Runs evaluation metrics for one or more generated-answer files and writes a report.

    Args:
        bob_path: Path to the bob_data.jsonl file (ground truth).
        generated_paths: One or more paths to generated-answer JSONL files.
            Each file is expected to have ``question`` and ``answer`` fields.
            The model name is inferred from the filename
            (``generated_answers_<model>.jsonl``).
        metrics: Metric names to run. Defaults to all registered metrics.
        output_path: Where to write the Markdown report.

    Returns:
        Nested dict ``{model_name: {score_name: value}}``.
    """
    if metrics is None:
        metrics = list(_METRICS.keys())

    bob_records = load_jsonl(bob_path)

    results_by_model: dict[str, dict[str, float]] = {}
    n_pairs_by_model: dict[str, int] = {}

    for path in generated_paths:
        model = model_name_from_path(path)
        generated_records = load_jsonl(path)
        references, hypotheses = align_pairs(bob_records, generated_records)
        if not references:
            print(f"  [{model}] WARNING: no matching pairs found, skipping.")
            continue
        n_pairs_by_model[model] = len(references)
        results_by_model[model] = _run_metrics_for_model(
            references, hypotheses, metrics, model
        )

    if not results_by_model:
        raise ValueError("No valid model results produced.")

    _print_table(results_by_model, n_pairs_by_model)
    _write_report(results_by_model, n_pairs_by_model, output_path)

    results_path = output_path.replace(".md", ".json")
    with open(results_path, "w") as f:
        json.dump(
            {"models": results_by_model, "n_pairs": n_pairs_by_model},
            f,
            indent=2,
        )
    print(f"Raw results written to {results_path}")

    return results_by_model


if __name__ == "__main__":
    import glob

    generated_files = sorted(glob.glob("data/generated_answers_*.jsonl"))
    evaluate(
        bob_path="data/bob_data.jsonl",
        generated_paths=generated_files,
    )
