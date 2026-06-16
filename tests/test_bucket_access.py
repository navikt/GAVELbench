"""Smoke-test for GCS bucket connectivity defined in src/config.yml.

Verifies that:
1. The bucket is reachable and the configured project/bucket name are valid.
2. A small test object can be uploaded and then deleted again.

Usage::

    uv run python tests/test_bucket_access.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
from storage import _get_bucket, _load_cfg, test_connection  # noqa: E402

_TEST_BLOB_NAME = "gavelbench-connection-test.txt"


def test_connectivity() -> None:
    """Verifies bucket connectivity using test_connection()."""
    print("=== Test 1: connectivity ===")
    test_connection()


def test_upload_download_delete() -> None:
    """Uploads a small file, reads it back, then deletes it."""
    print("\n=== Test 2: upload → download → delete ===")
    cfg = _load_cfg()
    bucket = _get_bucket()

    content = b"gavelbench bucket test"
    blob = bucket.blob(_TEST_BLOB_NAME)

    # Upload
    blob.upload_from_string(content)
    print(f"  ↑ gs://{cfg['name']}/{_TEST_BLOB_NAME}")

    # Download and verify
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp_path = tmp.name
    blob.download_to_filename(tmp_path)
    downloaded = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    assert downloaded == content, f"Content mismatch: {downloaded!r} != {content!r}"
    print(f"  ↓ gs://{cfg['name']}/{_TEST_BLOB_NAME}  (content verified)")

    # Delete
    blob.delete()
    print(f"  ✗ gs://{cfg['name']}/{_TEST_BLOB_NAME}  (deleted)")

    print("  ✅  Upload / download / delete OK")


def main() -> None:
    """Runs all bucket smoke-tests."""
    failures: list[str] = []

    for test_fn in [test_connectivity, test_upload_download_delete]:
        try:
            test_fn()
        except Exception as exc:
            print(f"  ❌  {test_fn.__name__}: {type(exc).__name__}: {exc}")
            failures.append(test_fn.__name__)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All bucket tests passed.")


if __name__ == "__main__":
    main()
