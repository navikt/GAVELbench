"""Fetches Q&A data from BigQuery and samples it by overarching category."""

import json
from typing import Any

from google.cloud import bigquery

_CITATION_PATTERN = r"\{[a-z0-9]+\}"

_QUERY = f"""
    with kategorier as (
        SELECT
            distinct
            bob.contextualized_question,
            bob.answer_content,
            base.data_categories
        FROM `nks-aiautomatisering-prod-194a.testgrunnlag.unnest_annotation_prod` as bob
        left join `nks-aiautomatisering-prod-194a.kunnskapsbase.kunnskapsartikler_oppsplittet` as base
        ON bob.context_title IN UNNEST(base.data_categories)
        WHERE NOT REGEXP_CONTAINS(bob.answer_content, r'{_CITATION_PATTERN}')
            AND base.data_categories IS NOT NULL
            AND ARRAY_LENGTH(base.data_categories) > 0
    )

    select * from kategorier
    """


def get_BQ_data() -> list[dict[str, str]]:
    """Queries BigQuery and returns rows as a list of dicts."""
    client = bigquery.Client()
    results = client.query(_QUERY).result()
    return [dict(row) for row in results]


def fetch_bob_data(output_path: str) -> None:
    """Fetches Bob Q&A data from BigQuery and writes it to *output_path* as JSON."""
    rows = get_BQ_data()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(rows)} rows to {output_path}")


def sample_by_overkategori(
    source_path: str,
    mapping_path: str,
    n_per_overkategori: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Samples up to *n_per_overkategori* Q&A entries for each overarching category.

    Each sampled row is tagged with a ``sampled_overkategori`` field. A Q&A that
    belongs to multiple overkategorier may appear more than once in the result —
    once per overkategori group it was drawn for.

    Args:
        source_path: Path to ``kunnskapsbase_kategorier.json``.
        mapping_path: Path to ``kategorier_mapping.json``.
        n_per_overkategori: Maximum number of Q&A pairs to sample per overkategori.
        seed: Random seed for reproducibility.

    Returns:
        List of Q&A dicts, each extended with a ``sampled_overkategori`` key.
    """
    import random

    with open(source_path, encoding="utf-8") as f:
        qa_data: list[dict[str, Any]] = json.load(f)

    with open(mapping_path, encoding="utf-8") as f:
        mapping_list: list[dict[str, Any]] = json.load(f)

    kategori_to_overkategorier: dict[str, list[str]] = {
        entry["kategori"]: entry["overkategori"] for entry in mapping_list
    }

    # Build pool: overkategori -> list of Q&A entries that belong to it
    pools: dict[str, list[dict[str, Any]]] = {}
    for entry in qa_data:
        for kategori in entry.get("data_categories") or []:
            for overkategori in kategori_to_overkategorier.get(
                str(kategori).strip(), []
            ):
                pools.setdefault(overkategori, []).append(entry)

    rng = random.Random(seed)
    result: list[dict[str, Any]] = []
    for overkategori, pool in sorted(pools.items()):
        sampled = rng.sample(pool, min(n_per_overkategori, len(pool)))
        for entry in sampled:
            result.append({**entry, "sampled_overkategori": overkategori})

    return result


if __name__ == "__main__":
    fetch_bob_data("data/kunnskapsbase_kategorier.json")
    with open("data/kunnskapsbase_kategorier.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    categories = set()
    for entry in data:
        if entry["data_categories"] is not None:
            categories.update(entry["data_categories"])
    print(f"Found {len(categories)} unique categories:")
    categories = {category.strip() for category in categories}
    with open("data/kunnskapsbase_kategorier.txt", "w", encoding="utf-8") as f:
        for category in sorted(categories):
            f.write(category + "\n")
    print("Wrote categories to data/kunnskapsbase_kategorier.txt")
