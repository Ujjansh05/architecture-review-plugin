import pytest


def test_windows_loader_preserves_values_and_validates_file(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    from arch_review.checkpoint import load_checkpoint_numpy

    original = {
        "half": torch.arange(24, dtype=torch.float16).reshape(4, 6),
        "single": torch.tensor([0.125, -3.5], dtype=torch.float32),
        "scalar": torch.tensor(7, dtype=torch.int64),
        "empty": torch.empty((0, 2)),
    }
    path = tmp_path / "weights.safetensors"
    save_file(original, path)
    loaded = load_checkpoint_numpy(path)
    assert set(loaded) == set(original)
    for key, value in original.items():
        assert loaded[key].dtype == value.dtype
        assert loaded[key].shape == value.shape
        assert torch.equal(loaded[key], value)
    # The actual consumer copies tensors; model writes must not change the checkpoint.
    layer = torch.nn.Linear(6, 4, bias=False)
    layer.load_state_dict({"weight": loaded["half"]})
    with torch.no_grad():
        layer.weight.fill_(0)
    assert torch.equal(loaded["half"], original["half"])
    malformed = tmp_path / "malformed.safetensors"
    malformed.write_bytes(b"not a valid checkpoint")
    with pytest.raises(Exception):
        load_checkpoint_numpy(malformed)
