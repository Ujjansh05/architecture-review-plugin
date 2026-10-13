import importlib.util
import io
import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from arch_review import merged_download as download


def make_archive(tmp_path, extra=None):
    files = {name: b"fixture" for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json",
             "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "LICENSE", "NOTICE")}
    import hashlib

    files["architecture_review_manifest.json"] = json.dumps({"model_id": download.MODEL_ID, "standalone": True,
        "artifact_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}}).encode()
    if extra:
        files.update(extra)
    target = tmp_path / "fixture.zip"
    with zipfile.ZipFile(target, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return target, sum(map(len, files.values()))


def test_installer_checks_pinned_archive_and_every_file(tmp_path, monkeypatch):
    archive, expanded = make_archive(tmp_path)
    monkeypatch.setattr(download, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(download, "EXPANDED_BYTES", expanded)
    monkeypatch.setattr(download, "ARCHIVE_SHA256", download.digest(archive))
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(archive.read_bytes()))
    target = tmp_path / "installed"
    result = download.download_merged_checkpoint(target)
    assert result["configured"] is False
    assert download.verify_directory(target)["standalone"]
    assert not list(tmp_path.glob(".archrivet-download-*"))
    with pytest.raises(ValueError, match="never overwritten"):
        download.download_merged_checkpoint(target)
    (target / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        download.verify_directory(target)


def test_bad_download_never_installs_or_leaves_staging(tmp_path, monkeypatch):
    archive, expanded = make_archive(tmp_path)
    monkeypatch.setattr(download, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(download, "EXPANDED_BYTES", expanded)
    monkeypatch.setattr(download, "ARCHIVE_SHA256", "0" * 64)
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(archive.read_bytes()))
    with pytest.raises(ValueError, match="SHA256"):
        download.download_merged_checkpoint(tmp_path / "installed")
    assert not (tmp_path / "installed").exists()
    assert not list(tmp_path.glob(".archrivet-download-*"))


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/escape", "a\\escape", "x/../escape",
                                  "x//y", "./x", "file:stream", "x. /y", "x./y"])
def test_unsafe_paths_rejected(path):
    with pytest.raises(ValueError, match="Unsafe"):
        download.safe_relative(path)


def test_archive_rejects_symlink_and_size_mismatch(tmp_path):
    archive, expanded = make_archive(tmp_path)
    with pytest.raises(ValueError, match="expanded size"):
        download.extract_archive(archive, tmp_path / "wrong-size", expanded + 1)
    with zipfile.ZipFile(archive, "a") as output:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(info, "../escape")
    with pytest.raises(ValueError, match="regular files"):
        download.extract_archive(archive, tmp_path / "link-test", expanded + len("../escape"))


def merger():
    path = Path(__file__).resolve().parents[1] / "scripts/merge_laya_adapter.py"
    spec = importlib.util.spec_from_file_location("merge_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_preserves_bias_and_function_and_removes_adapter():
    torch = pytest.importorskip("torch")
    merge = merger()
    trainer = merge.helper("train_partial_encoder")
    torch.manual_seed(17)
    base = torch.nn.Linear(12, 7, bias=True)
    bias = base.bias.detach().clone()
    wrapped = trainer.wrap_linear_adapter(base, 8, 16, torch.Generator().manual_seed(29))
    with torch.no_grad():
        wrapped.lora_B.normal_(std=.1)
    model = torch.nn.Module()
    model.layer = wrapped
    value = torch.randn(3, 12)
    expected = model.layer(value).detach()
    count = merge.merge_adapters(model, {"adapter_targets": {"layer": {}}, "adapter_alpha": 16, "adapter_rank": 8})
    assert count == 1 and type(model.layer) is torch.nn.Linear
    assert torch.equal(bias, model.layer.bias)
    torch.testing.assert_close(model.layer(value), expected, atol=1e-6, rtol=1e-5)
    assert set(model.state_dict()) == {"layer.weight", "layer.bias"}


def test_fidelity_gate_refuses_changed_decisions_and_probability_drift():
    merge = merger()
    reference = [{"id": "a", "choice": "supported", "probabilities": {"supported": .8, "contradicted": .2}}]
    assert merge.comparison(reference, reference)["identical_labels"] == 1
    with pytest.raises(ValueError, match="changed a decision"):
        merge.comparison(reference, [{**reference[0], "choice": "contradicted"}])
    with pytest.raises(ValueError, match="Probability drift"):
        merge.comparison(reference, [{**reference[0], "probabilities": {"supported": .9, "contradicted": .1}}])


def test_insufficient_disk_rejects_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "ARCHIVE_SHA256", "0" * 64)
    monkeypatch.setattr(download, "ARCHIVE_BYTES", 1)
    monkeypatch.setattr(download, "EXPANDED_BYTES", 2)
    monkeypatch.setattr(download.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="free bytes"):
        download.download_merged_checkpoint(tmp_path / "new")
