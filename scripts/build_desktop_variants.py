"""Build separate Rules and Laya CPU desktop packages from one engine wheel.

Both packages intentionally use the same MCP identity. Installing the other variant
replaces the previous installation; two copies must not be registered simultaneously.
Only the fixed public file list below is packaged. No model weights are included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "extensions" / "desktop"
VERSION = "0.3.0-alpha.1"
MODEL_PACKAGES = {"torch", "laya", "transformers", "safetensors", "numpy", "huggingface-hub", "tokenizers"}
STATIC_FILES = {"server/main.py", "server/setup_model.py", "README.md", "LICENSE",
                "THIRD_PARTY_NOTICES.md", "icon.png", "icons/icon-128.png", ".python-version"}


def wheel_version(wheel):
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Engine wheel must contain exactly one package metadata record")
        parsed = BytesParser().parsebytes(archive.read(metadata[0]))
    if parsed["Name"].replace("_", "-") != "architecture-review-plugin":
        raise ValueError("Expected the ArchRivet engine wheel")
    return parsed["Version"]


def manifest_for(variant):
    if variant not in {"rules", "laya-cpu"}:
        raise ValueError("Unknown bundle variant")
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    manifest["version"] = VERSION
    manifest["name"] = "archrivet"
    manifest["display_name"] = "ArchRivet - " + ("Rules" if variant == "rules" else "Laya CPU") + " (Preview)"
    manifest["long_description"] = (
        "Windows x64 experimental preview. Eight scoped backend checks, named missing facts and before/after reviews. "
        "Requires an MCPB 0.4 UV host. This variant replaces other ArchRivet installations. "
        + ("Rules only: no Torch, Laya or model download."
           if variant == "rules" else
           "Optional experimental Laya 322M CPU inference. Enable Laya explicitly to download the model; strict accuracy gates remain unmet."))
    env = manifest["server"]["mcp_config"]["env"]
    env.pop("ARCHRIVET_ENABLE_LAYA", None)
    manifest["user_config"] = {}
    if variant == "laya-cpu":
        env["ARCHRIVET_ENABLE_LAYA"] = "${user_config.enable_laya}"
        manifest["user_config"] = {"enable_laya": {
            "type": "boolean", "title": "Enable experimental local Laya model", "default": False, "required": False,
            "description": "Download the optional approximately 1 GB Laya checkpoint for CPU inference. Reuses verified existing model files."}}
    manifest["tools"] = [tool for tool in manifest["tools"] if tool["name"] != "cancel_model_setup"]
    manifest["tools"].insert(2, {"name": "cancel_model_setup", "description": "Pause unfinished model setup and retain resumable files."})
    return manifest


def project_for(variant, version, filename):
    dependency = "architecture-review-plugin" + ("[model]" if variant == "laya-cpu" else "") + "==" + version
    dependencies = [dependency] + (["torch==2.6.0+cpu"] if variant == "laya-cpu" else [])
    project = f'''[project]
name = "archrivet-desktop-extension"
version = "0.3.0a1"
description = "ArchRivet {variant} standalone MCP extension"
requires-python = ">=3.12,<3.13"
dependencies = {json.dumps(dependencies)}

[tool.uv]
package = false
environments = ["sys_platform == 'win32' and platform_machine == 'AMD64'"]

[tool.uv.sources]
architecture-review-plugin = {{ path = "vendor/{filename}" }}
'''
    if variant == "laya-cpu":
        project += '''torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
'''
    return project


def prepare_variant(directory, wheel, variant):
    directory.mkdir(parents=True, exist_ok=False)
    version = wheel_version(wheel)
    for name in STATIC_FILES:
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, destination)
    (directory / "vendor").mkdir()
    shutil.copyfile(wheel, directory / "vendor" / wheel.name)
    (directory / "manifest.json").write_text(json.dumps(manifest_for(variant), indent=2) + "\n", encoding="utf-8")
    (directory / "pyproject.toml").write_text(project_for(variant, version, wheel.name), encoding="utf-8")
    (directory / "bundle-config.json").write_text(json.dumps({"variant": variant, "engine_version": version,
        "extension_version": VERSION}, indent=2) + "\n", encoding="utf-8")
    return version


def audit_lock(directory, variant):
    lock = tomllib.loads((directory / "uv.lock").read_text(encoding="utf-8"))
    packages = sorted(package["name"] for package in lock["package"])
    forbidden = MODEL_PACKAGES.intersection(packages)
    if variant == "rules" and forbidden:
        raise ValueError("Rules dependency lock unexpectedly installs model libraries: " + ", ".join(sorted(forbidden)))
    if variant == "laya-cpu" and not {"torch", "laya"}.issubset(packages):
        raise ValueError("Laya dependency lock is missing the model runtime")
    return packages


def build(wheel, output, uv="uv", offline=False):
    wheel = wheel.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for variant in ("rules", "laya-cpu"):
        directory = output / variant
        version = prepare_variant(directory, wheel, variant)
        command = [uv, "lock", "--directory", str(directory)] + (["--offline"] if offline else [])
        subprocess.run(command, check=True)
        packages = audit_lock(directory, variant)
        files = STATIC_FILES | {"manifest.json", "pyproject.toml", "bundle-config.json", "uv.lock", "vendor/" + wheel.name}
        name = f"archrivet-windows-x64-{variant}-{VERSION}"
        artifact = output / (name + ".mcpb")
        with zipfile.ZipFile(artifact, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for member in sorted(files):
                info = zipfile.ZipInfo(member, date_time=(2026, 10, 16, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (directory / member).read_bytes(), compresslevel=9)
        portable = output / (name + ".zip")
        shutil.copyfile(artifact, portable)
        with artifact.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        results.append({"variant": variant, "engine_version": version, "extension_version": VERSION,
                        "plugin_identity": "archrivet", "dependency_packages": packages,
                        "files": sorted(files), "bytes": artifact.stat().st_size,
                        "artifact": artifact.name, "sha256": digest})
    report = {"variants": results, "publication_performed": False, "model_downloaded": False,
              "gui_installation_verified": False, "switching_variants": "replace_existing_archrivet_registration"}
    (output / "bundle-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    build(args.wheel, args.output, args.uv, args.offline)
