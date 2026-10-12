import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

REPO = "convaiinnovations/laya-multilingual"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download_checkpoint(destination: Path) -> dict:
    """Download a pinned public checkpoint directly, avoiding a second HF cache copy."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    info = get_json(f"https://huggingface.co/api/models/{REPO}")
    revision = info["sha"]
    entries = get_json(f"https://huggingface.co/api/models/{REPO}/tree/{revision}?recursive=true")
    entries = [entry for entry in entries if entry["type"] == "file"
               and entry["path"].endswith((".json", ".safetensors", ".md"))]
    required = sum(entry["size"] for entry in entries if not (destination / entry["path"]).exists())
    if shutil.disk_usage(destination).free < required + 256 * 1024 * 1024:
        raise RuntimeError(f"Checkpoint needs {required:,} bytes plus 256 MiB free headroom at {destination}.")
    hashes = {}
    for entry in entries:
        target = (destination / entry["path"]).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Unsafe upstream artifact path.")
        expected = (entry.get("lfs") or {}).get("oid")
        if target.exists():
            with target.open("rb") as existing_file:
                existing = hashlib.file_digest(existing_file, "sha256").hexdigest()
            if expected and existing == expected:
                hashes[entry["path"]] = existing
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        url = f"https://huggingface.co/{REPO}/resolve/{revision}/{entry['path']}"
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        actual = digest.hexdigest()
        if partial.stat().st_size != entry["size"] or (expected and actual != expected):
            raise ValueError(f"Artifact integrity check failed: {entry['path']}")
        partial.replace(target)
        hashes[entry["path"]] = actual
    manifest = {"role": "base", "repo": REPO, "revision": revision, "artifact_sha256": hashes,
                "calibration_status": "unvalidated", "specialized": False,
                "notice": "Unmodified base checkpoint; not fine-tuned for architecture reviews."}
    (destination / "architecture_review_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
