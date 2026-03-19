"""Generates answers from a language model based on questions from Bob."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anthropic
import yaml
from google import genai
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import Pipeline

# Cache for loaded HuggingFace pipelines (model_id -> pipeline)
_hf_pipelines: dict[str, "Pipeline"] = {}


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
    """Reads questions and reference answers from a JSON file."""
    with open(file_path, encoding="utf-8") as f:
        data: list[dict[str, str]] = json.load(f)
    return data


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


def _get_hf_pipeline(config: ModelConfig) -> "Pipeline":
    """Returns a cached HuggingFace text-generation pipeline for *config*.

    The pipeline is loaded once per model and reused across calls.
    Requires ``transformers``, ``accelerate``, and a compatible ``torch``
    installation (``pip install transformers accelerate torch``).
    """
    if config.id not in _hf_pipelines:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for HuggingFace models. "
                "Install it with: uv sync --group huggingface"
            ) from exc

        hf_token: str | None = config.extra.get("hf_token") or os.environ.get(
            "HF_TOKEN"
        )
        _hf_pipelines[config.id] = pipeline(
            "text-generation",
            model=config.id,
            device_map="auto",
            token=hf_token,
        )
    return _hf_pipelines[config.id]


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

    if config.provider == "vertex_anthropic":
        client_anthropic = anthropic.AnthropicVertex(
            region=config.location, project_id=config.project
        )
        max_tokens: int = int(config.extra.get("max_tokens", 1024))
        message = client_anthropic.messages.create(
            model=config.id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(message.content[0].text)

    if config.provider == "huggingface":
        pipe = _get_hf_pipeline(config)
        max_new_tokens: int = int(config.extra.get("max_new_tokens", 512))
        messages = [{"role": "user", "content": prompt}]
        result: Any = pipe(messages, max_new_tokens=max_new_tokens)
        # transformers returns a list of dicts; the last message is the reply
        generated: Any = result[0]["generated_text"]
        if isinstance(generated, list):
            return str(generated[-1]["content"])
        return str(generated)

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
    """Generates answers for all questions with one model and writes them to a JSON file."""
    safe_id = config.id.replace("/", "__")
    out_path = os.path.join(output_dir, f"generated_answers_{safe_id}.json")
    print(f"\n[{config.id}] {config.description}")
    print(f"  Writing to {out_path}")

    answers = []
    for question, max_words in zip(tqdm(questions, desc=config.id), lengths):
        answer = generate_answer(question, config, max_words=max_words)
        answers.append({"question": question, "answer": answer})

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)
        f.write("\n")


def run_pipeline(
    bob_path: str = "data/bob_data.json",
    models_yaml: str = "src/models.yaml",
    output_dir: str = "data",
    model_ids: list[str] | None = None,
) -> None:
    """Runs the full answer-generation pipeline for all (or selected) models.

    Args:
        bob_path: Path to the ground-truth JSON file.
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
