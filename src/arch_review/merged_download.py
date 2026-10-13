"""Pinned, streamed installation of the opt-in experimental merged checkpoint."""
import hashlib
import json
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

MODEL_ID = "archrivet-322m-experimental-v0.1.0"
RELEASE = "https://github.com/Ujjansh05/archrivet/releases/download/model-v0.1.0"
# Filled from the verified, immutable release archive during packaging.
ARCHIVE_SHA256 = "fade8bd93f6f9b8bfa80c5eaf717a6eb0c8f6cff2faf09d8516c0612092648e6"
ARCHIVE_BYTES = 964394458
EXPANDED_BYTES = 1322096788


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or ":" in name
            or any(part in ("", ".", "..") or part.endswith((" ", ".")) for part in name.split("/"))
            or path.as_posix() != name):
        raise ValueError("Unsafe model archive path")
    return path


def verify_directory(directory):
    manifest = json.loads((directory / "architecture_review_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("standalone") is not True:
        raise ValueError("Unexpected model manifest")
    artifacts = manifest.get("artifact_sha256", {})
    required = {"model.safetensors", "rl_agent_config.json", "encoder/config.json",
                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "LICENSE", "NOTICE"}
    if not required <= artifacts.keys() or len(artifacts) > 128:
        raise ValueError("Incomplete model manifest")
    for name, expected in artifacts.items():
        safe_relative(name)
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()) or digest(target) != expected:
            raise ValueError(f"Model checksum mismatch: {name}")
    return manifest


def extract_archive(archive, destination, expected_bytes):
    """Extract files only into a fresh private staging directory, with exact size bounds."""
    destination.mkdir()
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        names = [item.filename for item in members]
        if not 1 <= len(members) <= 128 or len(set(name.casefold() for name in names)) != len(names):
            raise ValueError("Unexpected or duplicate archive members")
        if sum(item.file_size for item in members) != expected_bytes:
            raise ValueError("Archive expanded size differs from the pinned release")
        for item in members:
            safe_relative(item.filename)
            kind = stat.S_IFMT(item.external_attr >> 16)
            if item.is_dir() or kind not in (0, stat.S_IFREG):
                raise ValueError("Model archive may contain regular files only")
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Archive path escapes the destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
    return verify_directory(destination)


def download_merged_checkpoint(destination: Path):
    if len(ARCHIVE_SHA256) != 64 or ARCHIVE_BYTES <= 0 or EXPANDED_BYTES <= 0:
        raise ValueError("Merged release has not been pinned")
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Choose a new destination; existing model folders are never overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = ARCHIVE_BYTES + EXPANDED_BYTES + 256 * 1024 ** 2
    if shutil.disk_usage(destination.parent).free < required:
        raise ValueError(f"Model installation needs {required:,} free bytes including temporary download space")
    url = f"{RELEASE}/{MODEL_ID}.zip"
    with tempfile.TemporaryDirectory(prefix=".archrivet-download-", dir=destination.parent) as temporary:
        root = Path(temporary)
        archive = root / "model.zip"
        checksum, count = hashlib.sha256(), 0
        request = urllib.request.Request(url, headers={"User-Agent": "ArchRivet-model-installer"})
        with urllib.request.urlopen(request, timeout=120) as incoming, archive.open("xb") as outgoing:
            while chunk := incoming.read(1024 * 1024):
                count += len(chunk)
                if count > ARCHIVE_BYTES:
                    raise ValueError("Model download exceeds the pinned release size")
                checksum.update(chunk)
                outgoing.write(chunk)
        if count != ARCHIVE_BYTES or checksum.hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Model archive failed its pinned SHA256/size verification; nothing installed")
        staged = root / "checkpoint"
        manifest = extract_archive(archive, staged, EXPANDED_BYTES)
        # A complete checkpoint appears only after all checks pass. Refuse races.
        if destination.exists():
            raise ValueError("Destination appeared during installation; refusing overwrite")
        staged.rename(destination)
    return {"checkpoint": str(destination), "archive_sha256": ARCHIVE_SHA256,
            "configured": False, "model": manifest}
