"""Model configuration loading and active-model filtering utilities."""

from dataclasses import dataclass, field

import yaml


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


def active_model_ids(yaml_path: str = "src/models.yaml") -> set[str]:
    """Returns the set of safe model IDs (``/`` replaced with ``__``) from *yaml_path*.

    This is the canonical way to determine which models are currently active,
    used for filtering generated-answer files and result dicts.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return {m["id"].replace("/", "__") for m in raw.get("models", [])}


def filter_by_active_models(
    data: dict[str, object],
    yaml_path: str = "src/models.yaml",
    *,
    warn: bool = True,
) -> dict[str, object]:
    """Returns a copy of *data* with only the keys that match active models.

    Args:
        data: Any dict keyed by model ID (safe form with ``__``).
        yaml_path: Path to ``models.yaml``.
        warn: If True, prints a message listing skipped models.

    Returns:
        Filtered dict.
    """
    active = active_model_ids(yaml_path)
    skipped = [k for k in data if k not in active]
    if warn and skipped:
        print(f"Skipping models not in {yaml_path}: {skipped}")
    return {k: v for k, v in data.items() if k in active}
