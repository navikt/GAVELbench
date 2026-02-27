"""Generates answers from a language model based on questions from Bob."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from google import genai
from tqdm import tqdm


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    id: str
    provider: str
    description: str = ""
    project: str = ""
    location: str = ""
    extra: dict[str, str] = field(default_factory=dict)


def load_model_configs(yaml_path: str = "src/models.yaml") -> list[ModelConfig]:
    """Loads model configurations from a YAML file.

    Top-level ``defaults`` are merged into each model entry so shared settings
    (e.g. project, location) don't need to be repeated per model.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    defaults: dict[str, str] = raw.get("defaults", {})
    configs: list[ModelConfig] = []
    for entry in raw.get("models", []):
        merged = {**defaults, **entry}
        known = {"id", "provider", "description", "project", "location"}
        configs.append(
            ModelConfig(
                id=merged["id"],
                provider=merged["provider"],
                description=merged.get("description", ""),
                project=merged.get("project", ""),
                location=merged.get("location", ""),
                extra={k: v for k, v in merged.items() if k not in known},
            )
        )
    return configs


def load_questions_answers(file_path: str) -> list[dict[str, str]]:
    """Reads questions and reference answers from a JSONL file."""
    with open(file_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_prompt(question: str, max_words: int | None) -> str:
    """Builds the system prompt for a given question."""
    prompt = (
        "Du er en hjelpsom assistent som svarer på spørsmål basert på informasjonen du har fått.\n"
        "Hent fortrinnsvis relevant informasjon fra offisielle kilder som nav.no.\n"
        "Svar med maks 200 ord, og inkluder lenker til kildene du har brukt.\n"
    )
    if max_words:
        prompt += f"Svar med maks {max_words} ord.\n"
    prompt += f"Spørsmålet er som følger:\n{question}"
    return prompt


def generate_answer(
    question: str, config: ModelConfig, max_words: int | None = None
) -> str:
    """Generates a single answer for *question* using the model described by *config*."""
    prompt = _build_prompt(question, max_words)

    if config.provider == "vertex_ai":
        client = genai.Client(
            vertexai=True, project=config.project, location=config.location
        )
        text: str = client.models.generate_content(
            model=config.id, contents=prompt
        ).text
        return text

    raise ValueError(
        f"Unsupported provider '{config.provider}' for model '{config.id}'."
        " Add a new branch in generate_answer() to support it."
    )


def run_model(
    config: ModelConfig,
    questions: list[str],
    lengths: list[int],
    output_dir: str = "data",
) -> None:
    """Generates answers for all questions with one model and writes them to a JSONL file."""
    out_path = os.path.join(output_dir, f"generated_answers_{config.id}.jsonl")
    print(f"\n[{config.id}] {config.description}")
    print(f"  Writing to {out_path}")

    answers = []
    for question, max_words in zip(tqdm(questions, desc=config.id), lengths):
        answer = generate_answer(question, config, max_words=max_words)
        answers.append({"question": question, "answer": answer})

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for qa in answers:
            json.dump(qa, f, ensure_ascii=False)
            f.write("\n")


def run_pipeline(
    bob_path: str = "data/bob_data.jsonl",
    models_yaml: str = "src/models.yaml",
    output_dir: str = "data",
    model_ids: list[str] | None = None,
) -> None:
    """Runs the full answer-generation pipeline for all (or selected) models.

    Args:
        bob_path: Path to the ground-truth JSONL file.
        models_yaml: Path to the models config YAML.
        output_dir: Directory where generated-answer files are written.
        model_ids: If provided, only run these model IDs (subset of what's in the YAML).
    """
    records = load_questions_answers(bob_path)
    questions = [r["contextualized_question"] for r in records]
    lengths = [len(r["answer_content"].strip().split()) for r in records]

    configs = load_model_configs(models_yaml)
    if model_ids:
        configs = [c for c in configs if c.id in model_ids]
        if not configs:
            raise ValueError(f"None of the requested model IDs found in {models_yaml}.")

    for config in configs:
        run_model(config, questions, lengths, output_dir=output_dir)


if __name__ == "__main__":
    run_pipeline(output_dir="data/generated")
