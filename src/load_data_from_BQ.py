"""Gets Bob-data from BigQuery, with questions and answers, and saves it as a jsonl file."""

import json

from google.cloud import bigquery

_URI = "nks-aiautomatisering-prod-194a.testgrunnlag.unnest_annotation_prod"
_QUERY = (
    f"SELECT distinct contextualized_question, answer_content FROM `{_URI}` limit 100"
)


def get_nks_bob_questions_answers() -> list[dict[str, str]]:
    """Queries BigQuery and returns rows as a list of dicts."""
    client = bigquery.Client()
    results = client.query(_QUERY).result()
    return [dict(row) for row in results]


def fetch_bob_data(output_path: str) -> None:
    """Fetches Bob Q&A data from BigQuery and writes it to *output_path* as JSON."""
    rows = get_nks_bob_questions_answers()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    fetch_bob_data("data/bob_data.json")
