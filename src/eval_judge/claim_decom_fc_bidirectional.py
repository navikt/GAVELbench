#!/usr/bin/env python3
"""Batch Evaluation Module for RAG Factual Correctness.

Processes a HuggingFace Dataset containing 'response' and 'reference' columns,
decomposes claims, calculates metrics, and appends scores back to the dataset.
"""

import asyncio
import json
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset
from google import genai
from tqdm import tqdm

# ==================== Core Logic ====================


async def extract_claims(text: str, client: genai.Client, model: str) -> List[str]:
    """Extract atomic factual claims from the given text using the specified model.

    Args:
        text (str): The input text from which to extract claims.
        client (genai.Client): The GenAI client for making requests.
        model (str): The model to use for extraction.

    Returns:
        List[str]: A list of extracted claims.
    """
    if not text or not isinstance(text, str):
        return [""]

    system_prompt = """Hent ut uavhengige og enkle faktapåstander som en JSON-liste. Regler:
                        1) Hvert element må være ett etterrettelig faktum.
                        2) Svar på samme språk som kildeteksten.
                        3) Gi kun gyldig JSON som utdata."""

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=[system_prompt, f"Extract claims from:\n{text}"],
            config={"temperature": 0.0, "response_mime_type": "application/json"},
        )

        raw_text = response.text.strip()
        for marker in ["```json", "```"]:
            if raw_text.startswith(marker):
                raw_text = raw_text[len(marker) :]
            if raw_text.endswith(marker):
                raw_text = raw_text[: -len(marker)]

        data = json.loads(raw_text)
        return [str(c).strip() for c in data if str(c).strip()]
    except Exception as e:
        print(f"Error extracting claims: {e}")
        return []


async def check_claim_coverage(
    claim: str, reference_claims: List[str], client: genai.Client, model: str
) -> bool:
    """Check if the target claim is covered by the reference claims.

    Args:
        claim (str): The claim to check.
        reference_claims (List[str]): The list of reference claims.
        client (genai.Client): The GenAI client for making requests.
        model (str): The model to use for verification.

    Returns:
        bool: True if the claim is covered, False otherwise.
    """
    if not reference_claims:
        return False

    ref_context = "\n".join([f"- {c}" for c in reference_claims])
    prompt = f"""You are a strict factual verification judge. Compare the [Target Claim] against the [Context].

                [Context]
                {ref_context}

                [Target Claim]
                "{claim}"

                [Rules]
                - Read the Context and find the exact facts related to the Target Claim.
                - If the Context fully supports the claim, the answer is YES.
                - If the Context contradicts the claim, or does not mention it at all, the answer is NO.

                Provide your reasoning in one short sentence. Then, on a new line, write your final verdict as exactly 'FINAL_VERDICT: TRUE' or 'FINAL_VERDICT: FALSE'.
                Return only FINAL_VERDICT in your output, no explanations or additional text."""

    try:
        response = await client.aio.models.generate_content(
            model=model, contents=prompt, config={"temperature": 0.0}
        )
        if "TRUE" in response.text.lower():
            return True
        else:
            return False
    except Exception:
        return False


def calculate_metrics(
    resp_claims: List[str],
    ref_claims: List[str],
    coverage_map: Dict[str, bool],
) -> Dict[str, float]:
    """Calculate evaluation metrics based on response and reference claims.

    Args:
        resp_claims (List[str]): The list of response claims.
        ref_claims (List[str]): The list of reference claims.
        coverage_map (Dict[str, bool]): A mapping of claims to their coverage status.

    Returns:
        Dict: A dictionary containing precision, recall, F1 score, and counts of claims.
    """
    tp = sum(1 for covered in coverage_map.values() if covered)
    fp = len(resp_claims) - tp
    fn = max(0, len(ref_claims) - tp)

    precision = tp / len(resp_claims) if resp_claims else 0.0
    recall = tp / len(ref_claims) if ref_claims else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_response_claims": len(resp_claims),
        "total_reference_claims": len(ref_claims),
    }


# ==================== Batch Processing Functions ====================


async def process_single_row(
    idx: int, response: str, reference: str, client: genai.Client, model: str
) -> Dict[str, Any]:
    """Process a single row of the dataset and return metrics and details.

    Args:
        idx (int): The index of the row being processed.
        response (str): The response text.
        reference (str): The reference text.
        client (genai.Client): The GenAI client for making requests.
        model (str): The model to use for extraction and verification.

    Returns:
        Dict: A dictionary containing metrics and claims information.
    """
    # 1. Extract Claims
    resp_claims: List[str] = await extract_claims(response, client, model)
    ref_claims: List[str] = await extract_claims(reference, client, model)

    # 2. Verify Claims Coverage
    coverage_map = {}
    if resp_claims:
        tasks = [
            check_claim_coverage(c, ref_claims, client, model) for c in resp_claims
        ]
        coverage_results = await asyncio.gather(*tasks)
        coverage_map = dict(zip(resp_claims, coverage_results))

    # 3. Calculate Metrics
    metrics = calculate_metrics(resp_claims, ref_claims, coverage_map)
    fn_value = metrics["false_negatives"]
    fp_value = metrics["false_positives"]
    has_missing_facts: bool = isinstance(fn_value, (int, float)) and fn_value > 0
    has_hallucinations: bool = isinstance(fp_value, (int, float)) and fp_value > 0

    return {
        "index": idx,
        "metrics": metrics,
        "resp_claims_count": len(resp_claims),
        "ref_claims_count": len(ref_claims),
        "resp_claims_list": resp_claims,
        "ref_claims_list": ref_claims,
        "has_missing_facts": has_missing_facts,
        "has_hallucinations": has_hallucinations,
    }


async def evaluate_dataset_batch(
    dataset: Dataset, client: genai.Client, model: str, batch_size: int = 2
) -> List[Dict[Any, Any]]:
    """Evaluate the entire dataset in batches to manage concurrency and rate limits.

    Args:
        dataset (Dataset): The HuggingFace Dataset to evaluate.
        client (genai.Client): The GenAI client for making requests.
        model (str): The model to use for evaluation.
        batch_size (int): The number of rows to process in each batch.

    Returns:
        List[Dict]: A list of result dictionaries for each processed row.
    """
    total_rows = len(dataset)
    all_results = []

    print(f"...:Starting batch evaluation of {total_rows} rows...")
    for i in tqdm(
        range(0, total_rows, batch_size), desc="Processing batches", leave=False
    ):
        batch_end = min(i + batch_size, total_rows)
        batch_indices = list(range(i, batch_end))
        tqdm.write(
            f"⏳ Processing batch {i // batch_size + 1}: Rows {i}-{batch_end - 1}"
        )

        # Prepare tasks for this batch
        tasks = [
            process_single_row(
                idx=idx,
                response=dataset[idx]["response"],
                reference=dataset[idx]["reference"],
                client=client,
                model=model,
            )
            for idx in batch_indices
        ]

        # Execute concurrently
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)

        print("   ✅ Completed batch.")

    return all_results


def create_result_dataframe(
    results: List[Dict[str, Any]],
    original_dataset: Dataset = None,
    include_claims: bool = False,
) -> pd.DataFrame:
    """Converts list of result dicts into a Pandas DataFrame.

    Args:
        results: List of dictionaries returned by evaluate_dataset_batch
        original_dataset: The original HuggingFace Dataset to extract text from
        include_claims: If True, adds 'response_claims' and 'reference_claims' columns

    Returns:
        A Pandas DataFrame with metrics, text context, and optionally raw claims.
    """
    flat_data = []

    for r in results:
        idx = r["index"]
        metrics = r["metrics"]

        row_entry = {
            "row_index": idx,
            # --- Metrics ---
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
            "resp_claims_count": r["resp_claims_count"],
            "ref_claims_count": r["ref_claims_count"],
            "has_missing_facts": r["has_missing_facts"],
            "has_hallucinations": r["has_hallucinations"],
        }

        # --- Add Original Text Context ---
        if original_dataset is not None:
            try:
                row_entry["user_input"] = original_dataset[idx].get("user_input", "")
                row_entry["response"] = original_dataset[idx].get("response", "")
                row_entry["reference"] = original_dataset[idx].get("reference", "")
            except IndexError:
                row_entry["user_input"] = "N/A"
                row_entry["response"] = "N/A"
                row_entry["reference"] = "N/A"

        # --- Add Decomposed Claims (Optional) ---
        if include_claims:
            # Safe fallback check:
            resp_list = r.get("resp_claims_list", [])
            ref_list = r.get("ref_claims_list", [])

            row_entry["response_claims_json"] = json.dumps(
                resp_list, ensure_ascii=False
            )
            row_entry["reference_claims_json"] = json.dumps(
                ref_list, ensure_ascii=False
            )

        flat_data.append(row_entry)

    return pd.DataFrame(flat_data)
