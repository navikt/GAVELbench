"""Uploads all current data artifacts to the configured GCS bucket.

Uploads:
  - data/kunnskapsbase_kategorier.json
  - data/kategorier_mapping.json
  - data/bob_data.json
  - data/generated/generated_answers_*.json
  - data/results/*.json  (evaluation results)
  - data/results/*.png   (report charts)

Skips auto-generated web assets (data/results/evaluation_report_files/).

Usage::

    uv run python scripts/upload_data.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")
from storage import upload_file  # noqa: E402

_DATA_ROOT = Path("data")

_SINGLE_FILES = [
    _DATA_ROOT / "kunnskapsbase_kategorier.json",
    _DATA_ROOT / "kategorier_mapping.json",
    _DATA_ROOT / "bob_data.json",
]

_GLOB_PATTERNS = [
    (_DATA_ROOT / "generated", "generated_answers_*.json"),
    (_DATA_ROOT / "results", "*.json"),
    (_DATA_ROOT / "results", "*.png"),
]


def _collect() -> list[Path]:
    """Returns all local paths to upload, in a stable order."""
    paths: list[Path] = []
    for p in _SINGLE_FILES:
        if p.exists():
            paths.append(p)
    for directory, pattern in _GLOB_PATTERNS:
        paths.extend(sorted(directory.glob(pattern)))
    return paths


def main(dry_run: bool = False) -> None:
    """Uploads collected data files to the bucket."""
    paths = _collect()

    if not paths:
        print("No data files found — nothing to upload.")
        sys.exit(0)

    print(f"{'DRY RUN — ' if dry_run else ''}Uploading {len(paths)} file(s):\n")
    for p in paths:
        size_kb = p.stat().st_size / 1024
        print(f"  {p}  ({size_kb:.0f} KB)")

    if dry_run:
        print("\nDry run complete — no files were uploaded.")
        return

    print()
    failed: list[str] = []
    for p in paths:
        try:
            upload_file(str(p), str(p))
        except Exception as exc:
            print(f"  ❌  {p}: {type(exc).__name__}: {exc}")
            failed.append(str(p))

    print(f"\n{len(paths) - len(failed)}/{len(paths)} files uploaded successfully.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload current data to the GCS bucket."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be uploaded without actually uploading.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
