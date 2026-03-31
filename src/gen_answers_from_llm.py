"""Generates answers from a language model based on questions from Bob."""

import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import yaml
from google import genai
from google.genai import errors as genai_errors
from tqdm import tqdm

_MAX_RETRIES = 6
_BASE_WAIT_SECONDS = 10

# Cache for loaded llama-cpp models (model_id -> Llama instance)
_llama_models: dict[str, Any] = {}

_LOCAL_MODEL_DIR = Path("models")


def _call_with_retry(fn: Callable[[], str]) -> str:
    """Calls fn(), retrying with exponential backoff on rate-limit (429) errors.

    Handles both google.genai.errors.ClientError (Vertex AI) and
    anthropic.RateLimitError (Anthropic on Vertex AI).
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except (genai_errors.ClientError, anthropic.RateLimitError) as exc:
            is_rate_limit = isinstance(exc, anthropic.RateLimitError) or (
                isinstance(exc, genai_errors.ClientError) and exc.status_code == 429
            )
            if not is_rate_limit or attempt == _MAX_RETRIES - 1:
                raise
            wait = _BASE_WAIT_SECONDS * (2**attempt) + random.uniform(0, 2)
            print(
                f"\n  [Rate limit 429] Waiting {wait:.0f}s before"
                f" retry {attempt + 1}/{_MAX_RETRIES - 1}…"
            )
            time.sleep(wait)
    raise RuntimeError("Unreachable")


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


def _import_huggingface_hub() -> Any:
    """Imports huggingface_hub, raising a helpful error if missing."""
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "The 'huggingface_hub' package is required for HuggingFace GGUF models. "
            "Install it with: uv sync --group huggingface"
        ) from exc
    return huggingface_hub


def _find_smallest_gguf_filename(repo_id: str) -> str:
    """Returns the filename of the smallest quantized GGUF file in *repo_id*.

    Quantization levels are ordered smallest-first: Q2 < Q3 < Q4 < Q5 < Q6 < Q8.
    Raises ``ValueError`` if no GGUF files are found.
    """
    hub = _import_huggingface_hub()
    gguf_files = [f for f in hub.list_repo_files(repo_id) if f.endswith(".gguf")]
    if not gguf_files:
        raise ValueError(f"No GGUF files found in HuggingFace repo '{repo_id}'.")

    quant_order = ["Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "F16", "F32"]

    def _quant_key(filename: str) -> int:
        upper = filename.upper()
        for i, q in enumerate(quant_order):
            if q in upper:
                return i
        return len(quant_order)

    return str(sorted(gguf_files, key=_quant_key)[0])


def _ensure_gguf_downloaded(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Returns the local path to *filename*, downloading it from *repo_id* if needed."""
    local_path = local_dir / filename
    if local_path.exists():
        return local_path

    hub = _import_huggingface_hub()
    print(f"  Downloading {filename} from {repo_id} → {local_dir}/")
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
    )
    return Path(downloaded)


def _get_llama_model(config: ModelConfig, local_model_dir: Path) -> Any:
    """Returns a cached llama-cpp ``Llama`` instance for *config*.

    On first call the GGUF file is downloaded from HuggingFace Hub (if absent)
    and the model is loaded into memory.  Subsequent calls return the cached
    instance immediately.
    """
    if config.id not in _llama_models:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "The 'llama-cpp-python' package is required for HuggingFace GGUF models. "
                "Install it with: uv sync --group huggingface"
            ) from exc

        gguf_filename: str = config.extra.get(
            "gguf_filename"
        ) or _find_smallest_gguf_filename(config.id)
        model_path = _ensure_gguf_downloaded(config.id, gguf_filename, local_model_dir)

        print(f"  Loading {gguf_filename}…")
        _llama_models[config.id] = Llama(
            model_path=str(model_path),
            n_ctx=int(config.extra.get("n_ctx", 2048)),
            verbose=False,
        )
    return _llama_models[config.id]


def generate_answer(
    question: str, config: ModelConfig, max_words: int | None = None
) -> str:
    """Generates a single answer for *question* using the model described by *config*."""
    prompt = _build_prompt(question, max_words)

    if config.provider == "vertex_ai":
        client = genai.Client(
            vertexai=True, project=config.project, location=config.location
        )
        return _call_with_retry(
            lambda: (
                client.models.generate_content(model=config.id, contents=prompt).text
            )
        )

    if config.provider == "vertex_anthropic":
        client_anthropic = anthropic.AnthropicVertex(
            region=config.location, project_id=config.project
        )
        max_tokens: int = int(config.extra.get("max_tokens", 1024))
        return _call_with_retry(
            lambda: str(
                client_anthropic.messages.create(
                    model=config.id,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                .content[0]
                .text
            )
        )

    if config.provider == "huggingface":
        local_model_dir = Path(
            config.extra.get("local_model_dir", str(_LOCAL_MODEL_DIR))
        )
        llm = _get_llama_model(config, local_model_dir)
        max_new_tokens: int = int(config.extra.get("max_new_tokens", 512))
        messages = [{"role": "user", "content": prompt}]
        result: Any = llm.create_chat_completion(
            messages=messages, max_tokens=max_new_tokens
        )
        return str(result["choices"][0]["message"]["content"])

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

    # Run HuggingFace (local GGUF) models first: a download failure is caught
    # early before any expensive cloud calls have been made.
    configs.sort(key=lambda c: 0 if c.provider == "huggingface" else 1)

    for config in configs:
        run_model(config, questions, lengths, output_dir=output_dir)


if __name__ == "__main__":
    run_pipeline(output_dir="data/generated")
