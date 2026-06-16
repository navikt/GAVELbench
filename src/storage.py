"""GCS bucket utilities for caching pipeline data artifacts.

All blob names mirror the local ``data/`` directory structure, e.g.
``data/bob_data.json`` locally maps to the same path in the bucket.
"""

from pathlib import Path

import yaml
from google.cloud import storage

CONFIG_PATH = "src/config.yml"


def _load_cfg(config_path: str = CONFIG_PATH) -> dict[str, str]:
    with open(config_path) as f:
        cfg: dict[str, str] = yaml.safe_load(f)["bucket"]
        return cfg


def _get_bucket(config_path: str = CONFIG_PATH) -> storage.Bucket:
    cfg = _load_cfg(config_path)
    client = storage.Client(project=cfg["project"])
    return client.bucket(cfg["name"])


def test_connection(config_path: str = CONFIG_PATH) -> None:
    """Verifies bucket connectivity. Raises on failure."""
    cfg = _load_cfg(config_path)
    client = storage.Client(project=cfg["project"])
    bucket = client.bucket(cfg["name"])
    bucket.reload()
    print(f"✅  Connected to gs://{cfg['name']} (project: {cfg['project']})")


def upload_file(
    local_path: str, blob_name: str, config_path: str = CONFIG_PATH
) -> None:
    """Uploads *local_path* to the bucket under *blob_name*."""
    bucket = _get_bucket(config_path)
    bucket.blob(blob_name).upload_from_filename(local_path)
    print(f"  ↑ gs://{bucket.name}/{blob_name}")


def download_file(
    blob_name: str, local_path: str, config_path: str = CONFIG_PATH
) -> bool:
    """Downloads *blob_name* from the bucket to *local_path*.

    Returns True if the blob existed and was downloaded, False if it was not found.
    """
    bucket = _get_bucket(config_path)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return False
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"  ↓ gs://{bucket.name}/{blob_name}")
    return True


def upload_dir(
    local_dir: str, blob_prefix: str, config_path: str = CONFIG_PATH
) -> None:
    """Uploads every file in *local_dir* to the bucket under *blob_prefix*/<filename>."""
    for path in sorted(Path(local_dir).iterdir()):
        if path.is_file():
            upload_file(str(path), f"{blob_prefix}/{path.name}", config_path)


def download_dir(
    blob_prefix: str, local_dir: str, config_path: str = CONFIG_PATH
) -> list[str]:
    """Downloads all blobs whose name starts with *blob_prefix* into *local_dir*.

    Returns the list of local paths that were written.
    """
    cfg = _load_cfg(config_path)
    client = storage.Client(project=cfg["project"])
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for blob in client.list_blobs(cfg["name"], prefix=blob_prefix + "/"):
        if blob.name.endswith("/"):
            continue
        local_path = str(Path(local_dir) / Path(blob.name).name)
        blob.download_to_filename(local_path)
        print(f"  ↓ gs://{cfg['name']}/{blob.name}")
        downloaded.append(local_path)
    return downloaded
