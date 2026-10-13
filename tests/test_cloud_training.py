"""Offline helper tests: no Torch, model downloads, OAuth, or cloud execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qlora = script("finetune_qlora")
drive = script("drive_artifacts")


def test_frozen_dataset_loads_only_non_test_splits(monkeypatch):
    original = Path.read_bytes
    opened = []

    def checked(path):
        opened.append(path.name)
        assert path.name != "test.jsonl", "Cloud training must not open final-test examples"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", checked)
    provenance, rows = qlora.load_dataset(ROOT / "datasets" / "backend-diagnostics-v1")
    assert {split: len(values) for split, values in rows.items()} == {
        "train": 120, "development": 36, "calibration": 36}
    assert provenance["final_test_opened"] is False
    assert set(opened) == {"manifest.json", "train.jsonl", "development.jsonl", "calibration.jsonl"}


def test_changed_dataset_bytes_fail_even_if_manifest_hash_is_rewritten(tmp_path):
    source = ROOT / "datasets" / "backend-diagnostics-v1"
    manifest = json.loads((source / "manifest.json").read_text())
    changed = (source / "train.jsonl").read_bytes() + b"\n"
    (tmp_path / "train.jsonl").write_bytes(changed)
    manifest["files"]["train.jsonl"]["sha256"] = qlora.digest(changed)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Frozen dataset hash mismatch"):
        qlora.load_dataset(tmp_path)


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and enable_thinking is False
        text = "".join(f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n" for item in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return [ord(character) for character in text]


def small_row(label="supported"):
    return {"id": "sample", "state": '{"evidence": [{"id": "original", "lesson": "Keep this unchanged."}]}',
            "questions": {"check": {"criteria": {"supported": "yes", "contradicted": "no"}}},
            "expected": {"check": label}}


def test_prompt_preserves_inputs_and_masks_every_non_assistant_token():
    row = small_row()
    prompt = qlora.messages(row)
    assert json.loads(prompt[1]["content"]) == {"state": row["state"], "questions": row["questions"]}
    assert qlora.messages(small_row("contradicted")) == prompt
    encoded = qlora.tokenize_example(CharacterTokenizer(), row)
    first_target = next(index for index, token in enumerate(encoded["labels"]) if token != -100)
    assert all(token == -100 for token in encoded["labels"][:first_target])
    target = "".join(chr(token) for token in encoded["labels"][first_target:])
    assert target == '{"verdict": "supported"}<|im_end|>\n'
    assert encoded["input_ids"][first_target:] == encoded["labels"][first_target:]
    with pytest.raises(ValueError, match="fit unchanged"):
        qlora.tokenize_example(CharacterTokenizer(), row, max_length=50)


def test_resource_checks_fail_before_any_model_initialization(tmp_path):
    with pytest.raises(ValueError, match="30 GiB"):
        qlora.check_scratch(tmp_path, free_bytes=30 * 1024 ** 3 - 1)
    assert qlora.check_scratch(tmp_path, free_bytes=30 * 1024 ** 3)["required_free_bytes"] == 30 * 1024 ** 3
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ValueError, match="CUDA GPU.*no model files"):
        qlora.check_cuda(torch)


@pytest.mark.parametrize("text", ['{"verdict": []}', '{"verdict": "yes"}', 'not json',
                                '{"verdict": "supported", "reason": "invented"}'])
def test_invalid_or_unsupported_output_is_not_scored_as_a_verdict(text):
    assert qlora.parse_prediction(text) is None
    assert qlora.parse_prediction('{"verdict": "contradicted"}') == "contradicted"


def artifact_directory(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text('{"status": "experimental"}')
    (run / "adapter").mkdir()
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"fake-adapter")
    (run / "token.json").write_text('"private-do-not-copy"')
    (run / "model.safetensors").write_bytes(b"base-weights-do-not-copy")
    (run / "training.log").write_text("Logs are not in the upload allowlist")
    return run


def test_drive_plan_excludes_credentials_logs_and_base_weights(tmp_path):
    plan = drive.artifact_plan(artifact_directory(tmp_path))
    assert {item["path"] for item in plan} == {"run.json", "adapter/adapter_model.safetensors"}
    assert all(len(item["sha256"]) == 64 for item in plan)


def test_drive_upload_is_idempotent_and_never_replaces_files(tmp_path):
    directory = artifact_directory(tmp_path)
    plan = drive.artifact_plan(directory)
    service, files = Mock(), Mock()
    service.files.return_value = files
    files.get.return_value.execute.return_value = {
        "mimeType": "application/vnd.google-apps.folder", "capabilities": {"canAddChildren": True}}
    files.list.return_value.execute.side_effect = [
        {"files": [{"id": "existing", "size": str(item["bytes"]), "sha256Checksum": item["sha256"]}]}
        for item in plan]
    result = drive.upload_artifacts(service, directory, "exampleFolderID123", plan=plan, media_factory=Mock())
    assert all(item["status"] == "already_present" for item in result)
    files.create.assert_not_called()
    files.update.assert_not_called()
    files.delete.assert_not_called()
    for call in files.list.call_args_list:
        assert "'exampleFolderID123' in parents" in call.kwargs["q"]
        assert "key='sha256'" in call.kwargs["q"]


def test_drive_refuses_changed_artifact_before_contacting_api(tmp_path):
    directory = artifact_directory(tmp_path)
    plan = drive.artifact_plan(directory)
    (directory / "run.json").write_text('{"status": "changed"}')
    service = Mock()
    with pytest.raises(ValueError, match="changed after"):
        drive.upload_artifacts(service, directory, "exampleFolderID123", plan=plan)
    service.files.assert_not_called()


def test_new_drive_artifacts_go_only_to_selected_folder_with_hash_metadata(tmp_path):
    directory = artifact_directory(tmp_path)
    plan = drive.artifact_plan(directory)
    service, files, media = Mock(), Mock(), Mock()
    service.files.return_value = files
    files.get.return_value.execute.return_value = {
        "mimeType": "application/vnd.google-apps.folder", "capabilities": {"canAddChildren": True}}
    files.list.return_value.execute.return_value = {"files": []}
    uploads = []
    for item in plan:
        request = Mock()
        request.next_chunk.return_value = (None, {"id": "new", "size": str(item["bytes"]),
                                                  "sha256Checksum": item["sha256"],
                                                  "appProperties": {"sha256": item["sha256"]}})
        uploads.append(request)
    files.create.side_effect = uploads
    result = drive.upload_artifacts(service, directory, "exampleFolderID123", plan=plan, media_factory=media)
    assert all(item["status"] == "uploaded" for item in result)
    for call, item in zip(files.create.call_args_list, plan):
        assert call.kwargs["body"]["parents"] == ["exampleFolderID123"]
        assert call.kwargs["body"]["appProperties"]["sha256"] == item["sha256"]
    files.update.assert_not_called()
    files.delete.assert_not_called()


def test_drive_never_trusts_shared_folder_sha_app_properties_without_server_checksum():
    item = {"sha256": "a" * 64, "bytes": 10}
    untrusted = {"size": "10", "appProperties": {"sha256": "a" * 64}}
    assert not drive.verified_content(untrusted, item)
    untrusted["sha256Checksum"] = "b" * 64
    assert not drive.verified_content(untrusted, item)
    untrusted["sha256Checksum"] = "a" * 64
    assert drive.verified_content(untrusted, item)


def test_notebook_is_unexecuted_with_opt_in_training_and_blank_drive_folder():
    notebook = json.loads((ROOT / "notebooks" / "archrivet_cloud_training.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in cells)
    for cell in cells:
        compile("".join(cell["source"]), cell["id"], "exec")
    source = "\n".join("".join(cell["source"]) for cell in cells)
    assert "DRIVE_FOLDER_ID = ''" in source
    assert "RUN_TRAINING = False" in source and "UPLOAD_TO_DRIVE = False" in source
    assert source.index("torch.cuda.is_available()") < source.index("'git', 'clone'")
    assert "drive.mount" not in source
