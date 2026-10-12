import json
import os
import secrets
from pathlib import Path

from filelock import FileLock
from pydantic import Field

from .schema import StrictModel


class Settings(StrictModel):
    port: int = Field(default=8765, ge=1024, le=65535)
    checkpoint: str | None = None
    model_role: str = "unconfigured"
    idle_seconds: int = Field(default=300, ge=1)
    inference_timeout_seconds: int = Field(default=180, ge=1)
    threads: int = Field(default=4, ge=1, le=32)


def runtime_dir() -> Path:
    root = Path(os.environ.get("ARCH_REVIEW_HOME", str(Path.home() / ".arch-review"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_settings() -> Settings:
    target = runtime_dir() / "settings.json"
    return Settings.model_validate_json(target.read_text(encoding="utf-8")) if target.exists() else Settings()


def save_settings(settings: Settings):
    root = runtime_dir()
    with FileLock(str(root / "settings.lock")):
        temporary = root / "settings.json.tmp"
        temporary.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(root / "settings.json")


def get_token() -> str:
    root = runtime_dir()
    with FileLock(str(root / "token.lock")):
        target = root / "token"
        if not target.exists():
            with target.open("x", encoding="utf-8") as stream:
                stream.write(secrets.token_urlsafe(32))
            if os.name != "nt":
                target.chmod(0o600)
        token = target.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Local authentication token is empty; remove the empty token file and retry.")
        return token


def model_metadata(checkpoint: str | None) -> dict:
    if not checkpoint:
        return {"role": "unconfigured", "calibration_status": "unvalidated"}
    manifest = Path(checkpoint) / "architecture_review_manifest.json"
    if not manifest.exists():
        return {"role": "unregistered", "calibration_status": "unvalidated", "checkpoint": checkpoint}
    return json.loads(manifest.read_text(encoding="utf-8"))
