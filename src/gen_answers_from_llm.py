"""Generates answers from a language model based on questions from Bob."""

import json
from typing import Any, Dict, List

from google import genai
from tqdm import tqdm


def read_questions_answers_from_file(file_path: str) -> List[Dict[str, str]]:
    """Reads generated answers from a jsonl file."""
    answers = []
    with open(file_path, "r") as f:
        for line in f:
            answers.append(json.loads(line))
    return answers


def generate_answers_from_llm(
    question: str,
    model: str = "gemini-2.0-flash-001",
    lenght_of_answer: int | None = None,
) -> str | Any:
    """Generates answers from a language model based on the provided questions."""
    system_instruction = """
            Du er en hjelpsom assistent som svarer på spørsmål basert på informasjonen du har fått.
            Hent fortrinnsvis relevant informasjon fra offisielle kilder som nav.no.
            Svar med maks 200 ord, og inkluder lenker til kildene du har brukt.
            Spørsmålet er som følger: \n
        """
    if lenght_of_answer:
        system_instruction += f"Svar med maks {lenght_of_answer} ord. \n"
    system_instruction += f"{question}"
    if model == "gemini-2.0-flash-001":
        client = genai.Client(
            vertexai=True, project="tada-prod-4dac", location="europe-west1"
        )
        response = client.models.generate_content(
            model=model, contents=system_instruction
        ).text
    return response


def answer_pipeline(
    model: str, questions: List[str], lengths_of_answers: List[int]
) -> None:
    """Pipeline for generating answers from a language model based on the provided questions."""
    answers = []
    for question, length_of_answer in zip(tqdm(questions), lengths_of_answers):
        answer = generate_answers_from_llm(question, lenght_of_answer=length_of_answer)
        answers.append({"question": question, "answer": answer})

    with open(f"data/generated_answers_{model}.jsonl", "w") as f:
        for qa in answers:
            json.dump(qa, f)
            f.write("\n")


if __name__ == "__main__":
    # Example usage
    questions_answers = read_questions_answers_from_file("data/bob_data.jsonl")
    questions = [qa["contextualized_question"] for qa in questions_answers]
    lengths_of_answers = [
        len(qa["answer_content"].strip().split()) for qa in questions_answers
    ]

    answer_pipeline("gemini-2.0-flash-001", questions, lengths_of_answers)
