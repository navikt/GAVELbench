"""Smoke-tests access to all Vertex AI Gemini and Claude models in models.yaml.

For each model a short prompt is sent and the first 100 characters of
the response are printed.  Any authentication or API error is caught and
reported without stopping the remaining tests.

Usage::

    uv run python tests/test_api_access.py
"""

import sys

import anthropic
from google import genai
from google.genai.types import HttpOptions

sys.path.insert(0, "src")
from models import load_model_configs  # noqa: E402

_TEST_PROMPT = "Svar på norsk med én setning: Hva er NAV?"


def test_vertex_ai(model_id: str, project: str, location: str) -> None:
    """Sends a test prompt to a Vertex AI Gemini model and prints the response preview."""
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=HttpOptions(api_version="v1"),
    )
    response = client.models.generate_content(model=model_id, contents=_TEST_PROMPT)
    preview = (response.text or "").strip().replace("\n", " ")[:100]
    print(f"  ✅  {preview!r}")


def test_vertex_anthropic(model_id: str, project: str, location: str) -> None:
    """Sends a test prompt to an Anthropic model via Vertex AI and prints the response preview."""
    client = anthropic.AnthropicVertex(region=location, project_id=project)
    response = client.messages.create(
        model=model_id,
        max_tokens=256,
        messages=[{"role": "user", "content": _TEST_PROMPT}],
    )
    preview = str(response.content[0].text).strip().replace("\n", " ")[:100]
    print(f"  ✅  {preview!r}")


def main() -> None:
    """Runs smoke-tests for all cloud models in models.yaml."""
    configs = load_model_configs("src/models.yaml")
    cloud_configs = [
        c for c in configs if c.provider in ("vertex_ai", "vertex_anthropic")
    ]

    if not cloud_configs:
        print("No cloud models found in models.yaml.")
        sys.exit(1)

    print(f"Testing {len(cloud_configs)} cloud model(s) …\n")
    failures: list[str] = []

    for config in cloud_configs:
        provider_label = "Gemini" if config.provider == "vertex_ai" else "Claude"
        print(f"[{config.id}]  {config.description}  ({provider_label})")
        try:
            if config.provider == "vertex_ai":
                test_vertex_ai(config.id, config.project, config.location)
            else:
                test_vertex_anthropic(config.id, config.project, config.location)
        except Exception as exc:
            print(f"  ❌  {type(exc).__name__}: {exc}")
            failures.append(config.id)
        print()

    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All models OK.")


if __name__ == "__main__":
    main()
