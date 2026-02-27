"""Gets Bob-data from BigQuery, with questions and answers, and saves it as a jsonl file."""

import json
from typing import Dict, List

from google.cloud import bigquery


def get_nks_bob_questions_answers() -> List[Dict[str, str]]:
    """Gets Bob-data from BigQuery, with questions and answers."""
    URI = "nks-aiautomatisering-prod-194a.testgrunnlag.unnest_annotation_prod"
    client = bigquery.Client()
    query = f"SELECT distinct contextualized_question, answer_content FROM `{URI}` limit 100"
    query_job = client.query(query)
    results = query_job.result()

    rows = [dict(row) for row in results]
    return rows


if __name__ == "__main__":
    rows = get_nks_bob_questions_answers()
    with open("data/bob_data.jsonl", "w") as f:
        for row in rows:
            json.dump(row, f)
            f.write("\n")
