"""Build the icon-enabled desktop bundle from the verified 0.3 preview bundle.

The engine wheel, dependency lock and optional-model installer are preserved byte for
byte. Nothing outside the explicit package file list is included.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA256 = "5b8131ab090a9c9165f38ea3a46247933bc01744d620268143418c0b28347077"
WHEEL = "vendor/architecture_review_plugin-0.3.0a1-py3-none-any.whl"
BASE_FILES = {"manifest.json", "pyproject.toml", ".python-version", "README.md", "server/main.py",
              "server/setup_model.py", "LICENSE", "THIRD_PARTY_NOTICES.md", "uv.lock", WHEEL}
UPDATED = {"manifest.json", "README.md", "server/main.py"}
ADDED = {"icon.png", "icons/icon-128.png"}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def build(base, output):
    if digest(base.read_bytes()) != BASE_SHA256:
        raise ValueError("Expected the original, checksum-verified v0.3.0a1 desktop bundle")
    with zipfile.ZipFile(base) as archive:
        if len(archive.namelist()) != len(BASE_FILES) or set(archive.namelist()) != BASE_FILES:
            raise ValueError("Unexpected original bundle files")
        files = {name: archive.read(name) for name in BASE_FILES}
    original = {name: digest(value) for name, value in files.items()}
    for name in UPDATED | ADDED:
        files[name] = (ROOT / "extensions/desktop" / name).read_bytes()
    manifest = json.loads(files["manifest.json"])
    if manifest["version"] != "0.2.0-alpha.2":
        raise ValueError("This historical builder needs the 0.2.0-alpha.2 source checkout. "
                         "For current Rules/Laya packages use scripts/build_desktop_variants.py.")
    assert manifest["icon"] == "icon.png"
    assert manifest["icons"][0]["src"] in files
    assert all(digest(files[name]) == original[name] for name in BASE_FILES - UPDATED)
    output.mkdir(parents=True, exist_ok=False)
    name = "archrivet-windows-x64-0.2.0-alpha.2"
    artifact = output / (name + ".mcpb")
    with zipfile.ZipFile(artifact, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for member in sorted(files):
            # Stable timestamps make the archive reproducible across machines.
            info = zipfile.ZipInfo(member, date_time=(2026, 10, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, files[member], compresslevel=9)
    with zipfile.ZipFile(artifact) as archive:
        assert set(archive.namelist()) == BASE_FILES | ADDED
        assert archive.testzip() is None
        for member in files:
            assert archive.read(member) == files[member]
    portable = output / (name + ".zip")
    portable.write_bytes(artifact.read_bytes())
    # Every path comes from the fixed allowlist, and output is a newly created directory.
    extracted = output / "extracted"
    for member, data in files.items():
        target = extracted / member
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    audit = {"extension_version": manifest["version"], "engine_version": "0.3.0a1",
             "base_sha256": BASE_SHA256, "artifact": artifact.name,
             "sha256": digest(artifact.read_bytes()), "bytes": artifact.stat().st_size,
             "files": {name: digest(value) for name, value in sorted(files.items())},
             "engine_lockfile_and_model_installer_unchanged": True,
             "private_training_files_included": False}
    audit_path = output / "branding-bundle-audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    assets = (artifact, portable, audit_path)
    (output / "BRANDING-SHA256SUMS.txt").write_text(
        "".join(digest(path.read_bytes()) + "  " + path.name + "\n" for path in assets), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.base_bundle, args.output)
