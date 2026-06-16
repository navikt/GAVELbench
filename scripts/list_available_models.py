"""Lists which models are available in the Vertex AI project/location from models.yaml.

Reads the ``defaults`` (project + location) and configured models from
``src/models.yaml``, then queries the Vertex AI *publisher models* catalog for
each provider used in the config:

  * ``vertex_ai``        → ``publishers/google``   (Gemini models)
  * ``vertex_anthropic`` → ``publishers/anthropic`` (Claude models)

The catalog is region-specific, so it reflects what can actually be called in
the configured location.  ``huggingface`` (local GGUF) models are not Vertex
publisher models and are listed separately without an availability check.

The output is grouped so you can see at a glance:

  ✅  Configured AND available        — keep
  ❌  Configured but NOT available    — discontinued / wrong region → remove or move
  ➕  Available but NOT configured     — candidates you could add to models.yaml

Authentication uses Application Default Credentials (run
``gcloud auth login --update-adc`` first).

Usage::

    uv run python scripts/list_available_models.py
    uv run python scripts/list_available_models.py --all       # don't filter to chat models
    uv run python scripts/list_available_models.py --location us-central1
"""

import argparse
import sys
import urllib.error
import urllib.request
from typing import Any

import google.auth
import google.auth.transport.requests
import yaml

sys.path.insert(0, "src")
from models import load_model_configs  # noqa: E402

MODELS_YAML = "src/models.yaml"

# Maps a models.yaml provider to its Vertex AI publisher (None = not a Vertex
# publisher model, e.g. local HuggingFace GGUF models).
_PROVIDER_TO_PUBLISHER: dict[str, str | None] = {
    "vertex_ai": "google",
    "vertex_anthropic": "anthropic",
    "huggingface": None,
}


def _load_defaults(yaml_path: str) -> dict[str, str]:
    """Returns the ``defaults`` block (project, location, …) from models.yaml."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    defaults: dict[str, str] = raw.get("defaults", {})
    return defaults


def _access_token() -> str:
    """Returns an OAuth access token from Application Default Credentials."""
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    token: str = credentials.token
    return token


def list_publisher_models(
    publisher: str, project: str, location: str, token: str
) -> list[dict[str, Any]]:
    """Returns the raw publisher-model entries for *publisher* in *location*.

    Follows pagination and returns the combined ``publisherModels`` list.
    """
    base = (
        f"https://{location}-aiplatform.googleapis.com/v1beta1"
        f"/publishers/{publisher}/models"
    )
    models: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        url = f"{base}?pageSize=300"
        if page_token:
            url += f"&pageToken={page_token}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                # Required so ADC has a quota project for aiplatform.googleapis.com.
                "x-goog-user-project": project,
            },
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted Google host)
            data: dict[str, Any] = yaml.safe_load(resp.read().decode("utf-8"))
        models.extend(data.get("publisherModels", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return models


def _short_name(entry: dict[str, Any]) -> str:
    """Returns the trailing model ID from a publisher-model ``name`` field."""
    return str(entry["name"]).split("/")[-1]


def _is_chat_model(publisher: str, model_id: str) -> bool:
    """Heuristic: True if *model_id* is a text/chat generation model.

    Filters out embedding, image, TTS and live-audio models so the suggestions
    focus on models usable by this benchmark.
    """
    if publisher == "anthropic":
        return model_id.startswith("claude")
    if publisher == "google":
        if not model_id.startswith("gemini"):
            return False
        excluded = ("tts", "embedding", "live", "audio", "image")
        return not any(token in model_id for token in excluded)
    return True


def _print_section(title: str, items: list[str]) -> None:
    """Prints a titled, sorted bullet list (or a placeholder when empty)."""
    print(f"\n{title}")
    if not items:
        print("    (none)")
        return
    for item in sorted(items):
        print(f"    {item}")


def main(
    yaml_path: str = MODELS_YAML,
    location_override: str | None = None,
    show_all: bool = False,
) -> None:
    """Compares configured models against the Vertex AI catalog and prints a report."""
    defaults = _load_defaults(yaml_path)
    project = defaults.get("project", "")
    location = location_override or defaults.get("location", "")
    if not project or not location:
        print("models.yaml defaults must define both 'project' and 'location'.")
        sys.exit(1)

    configs = load_model_configs(yaml_path)

    print(f"Project:  {project}")
    print(f"Location: {location}")
    print(f"Source:   {yaml_path}")

    # Group configured model IDs by the publisher they map to.
    configured_by_publisher: dict[str, set[str]] = {}
    local_models: list[str] = []
    for cfg in configs:
        publisher = _PROVIDER_TO_PUBLISHER.get(cfg.provider, None)
        if publisher is None:
            local_models.append(f"{cfg.id}  (provider: {cfg.provider})")
            continue
        configured_by_publisher.setdefault(publisher, set()).add(cfg.id)

    # Always check the publishers we know about, even if nothing is configured
    # for them yet, so users can discover models for an unused provider.
    publishers = sorted(set(configured_by_publisher) | {"google", "anthropic"})

    token = _access_token()

    for publisher in publishers:
        configured = configured_by_publisher.get(publisher, set())
        try:
            entries = list_publisher_models(publisher, project, location, token)
        except urllib.error.HTTPError as exc:
            print(f"\n=== publisher: {publisher} ===")
            print(f"    Could not list models: HTTP {exc.code} {exc.reason}")
            continue

        all_available = {_short_name(e) for e in entries}
        chat_available = {
            mid for mid in all_available if _is_chat_model(publisher, mid)
        }
        # "Available" set used for suggestions: all models with --all, else chat only.
        suggestable = all_available if show_all else chat_available

        configured_available = sorted(configured & all_available)
        configured_missing = sorted(configured - all_available)
        not_configured = sorted(suggestable - configured)

        provider_name = next(
            (p for p, pub in _PROVIDER_TO_PUBLISHER.items() if pub == publisher),
            publisher,
        )
        print(
            f"\n=== publisher: {publisher} "
            f"(provider: {provider_name}) — {len(all_available)} model(s) in catalog ==="
        )
        _print_section("✅  Configured AND available (keep):", configured_available)
        _print_section(
            "❌  Configured but NOT available (discontinued / wrong region):",
            configured_missing,
        )
        label = "all catalog" if show_all else "chat-capable"
        _print_section(
            f"➕  Available but NOT configured ({label}, candidates to add):",
            not_configured,
        )

    if local_models:
        _print_section(
            "\nℹ️   Local models (no Vertex availability check):", local_models
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "List which models are available in the Vertex AI project/location"
            " configured in src/models.yaml, vs. what's currently configured."
        )
    )
    parser.add_argument(
        "--models-yaml",
        default=MODELS_YAML,
        help=f"Path to the models config YAML (default: {MODELS_YAML}).",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Override the location from models.yaml defaults (e.g. us-central1).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Suggest all catalog models, not just text/chat generation models.",
    )
    args = parser.parse_args()
    main(
        yaml_path=args.models_yaml,
        location_override=args.location,
        show_all=args.all,
    )
