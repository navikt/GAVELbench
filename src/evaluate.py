"""Evaluates the generated answers against the ground truth from Bob based on various metrics."""

import json
import os
import re
from collections import Counter
from typing import Callable

import numpy as np
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from transformers import pipeline

from report import print_report, write_report

# A metric function takes (references, hypotheses) and returns a dict of score_name -> float.
MetricFn = Callable[[list[str], list[str]], dict[str, float]]

_METRICS: dict[str, MetricFn] = {}


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


_NLI_MODEL = "alexandrainst/scandi-nli-small"


@register_metric("nli_entailment")
def compute_nli_entailment(
    references: list[str], hypotheses: list[str]
) -> dict[str, float]:
    """Computes mean NLI entailment probability using a Scandinavian NLI model.

    Framing:
        premise   = model output (hypothesis/generated answer)
        hypothesis = correct answer (reference from Bob)

    A high entailment score means the model output supports / contains the
    correct answer.
    """
    nli = pipeline(
        "text-classification",
        model=_NLI_MODEL,
        device=-1,
        top_k=None,
        truncation=True,
    )
    pairs = [
        {"text": hyp, "text_pair": ref} for hyp, ref in zip(hypotheses, references)
    ]
    results = nli(pairs)
    scores = [
        next(r["score"] for r in result if r["label"] == "entailment")
        for result in results
    ]
    return {"nli_entailment": float(np.mean(scores))}


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

    print_report(results_by_model, n_pairs_by_model)
    write_report(results_by_model, n_pairs_by_model, output_path)

    return results_by_model


if __name__ == "__main__":
    import glob

    generated_files = sorted(glob.glob("data/generated_answers_*.jsonl"))
    evaluate(
        bob_path="data/bob_data.jsonl",
        generated_paths=generated_files,
    )
