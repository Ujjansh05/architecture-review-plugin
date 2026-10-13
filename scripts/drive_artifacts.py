"""Copy explicit experiment artifacts to one user-selected Drive folder via API.

No mounting, broad Drive crawl, deletion, overwriting, or credential files. Use
interactive Colab authentication or existing application-default credentials.
"""
import argparse
import hashlib
import json
import mimetypes
import re
from pathlib import Path

ROOT_NAMES = {"run.json", "report.json"}
ADAPTER_NAMES = {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json", "chat_template.jinja",
                 "generation_config.json"}
MAX_FILE_BYTES = 1024 ** 3
MAX_TOTAL_BYTES = 2 * 1024 ** 3


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_plan(directory):
    root = directory.resolve()
    if not root.is_dir() or not (root / "run.json").is_file():
        raise ValueError("Select one experiment output directory containing run.json")
    plan = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        allowed = (relative.as_posix() in ROOT_NAMES
                   or (len(relative.parts) == 2 and relative.parts[0] == "adapter" and path.name in ADAPTER_NAMES))
        if not allowed:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Artifact symlinks may not leave the selected experiment directory")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError("An artifact exceeds the 1 GiB upload bound")
        plan.append({"path": relative.as_posix(), "name": root.name + "__" + "__".join(relative.parts),
                     "sha256": sha_file(path), "bytes": size})
    if sum(item["bytes"] for item in plan) > MAX_TOTAL_BYTES:
        raise ValueError("Experiment artifacts exceed the 2 GiB upload bound")
    return plan


def validate_folder_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{10,200}", value or ""):
        raise ValueError("Enter the intended Google Drive folder ID")
    return value


def verified_content(remote, item):
    """Use Drive's output-only checksum, never caller-controlled appProperties."""
    # https://developers.google.com/workspace/drive/api/reference/rest/v3/files
    return (str(remote.get("size", "")) == str(item["bytes"])
            and remote.get("sha256Checksum", "").lower() == item["sha256"])


def upload_artifacts(service, directory, folder_id, plan=None, media_factory=None):
    """Idempotent by source path/name plus SHA-256; changed artifacts create new files."""
    folder_id = validate_folder_id(folder_id)
    current = artifact_plan(directory)
    if plan is not None and current != plan:
        raise ValueError("Artifacts changed after the upload plan was prepared")
    folder = service.files().get(fileId=folder_id, fields="id,mimeType,capabilities(canAddChildren)",
                                 supportsAllDrives=True).execute()
    if (folder.get("mimeType") != "application/vnd.google-apps.folder"
            or not folder.get("capabilities", {}).get("canAddChildren")):
        raise ValueError("The selected Drive folder is not writable")
    if media_factory is None:
        from googleapiclient.http import MediaFileUpload
        media_factory = MediaFileUpload
    uploaded = []
    for item in current:
        key = hashlib.sha256(item["name"].encode()).hexdigest()
        query = (f"'{folder_id}' in parents and trashed = false and "
                 f"appProperties has {{ key='archrivet_path' and value='{key}' }} and "
                 f"appProperties has {{ key='sha256' and value='{item['sha256']}' }}")
        found = service.files().list(q=query, spaces="drive", pageSize=100,
            fields="files(id,name,size,sha256Checksum,appProperties)",
            supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        matches = [row for row in found.get("files", []) if verified_content(row, item)]
        if matches:
            uploaded.append({"path": item["path"], "sha256": item["sha256"], "status": "already_present"})
            continue
        path = directory / item["path"]
        if sha_file(path) != item["sha256"]:
            raise ValueError("Artifact changed during upload")
        media = media_factory(str(path), mimetype=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                              resumable=True, chunksize=8 * 1024 ** 2)
        request = service.files().create(body={"name": item["name"], "parents": [folder_id],
            "appProperties": {"archrivet_path": key, "sha256": item["sha256"], "artifact_kind": "pilot_adapter"}},
            media_body=media, fields="id,size,sha256Checksum,appProperties", supportsAllDrives=True)
        result = None
        while result is None:
            _, result = request.next_chunk(num_retries=3)
        if not result.get("sha256Checksum"):
            result = service.files().get(fileId=result["id"], fields="id,size,sha256Checksum",
                                         supportsAllDrives=True).execute()
        if not verified_content(result, item):
            raise ValueError("Drive server SHA-256 verification failed; artifact is not marked complete")
        uploaded.append({"path": item["path"], "sha256": item["sha256"], "status": "uploaded"})
    return uploaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--folder-id")
    parser.add_argument("--upload", action="store_true", help="Upload the displayed allowlisted artifacts")
    args = parser.parse_args()
    plan = artifact_plan(args.directory)
    if not args.upload:
        print(json.dumps({"status": "dry_run", "artifacts": plan}, indent=2))
        return
    validate_folder_id(args.folder_id)
    import google.auth
    from googleapiclient.discovery import build

    credentials, _ = google.auth.default()
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    print(json.dumps(upload_artifacts(service, args.directory, args.folder_id, plan=plan), indent=2))


if __name__ == "__main__":
    main()
