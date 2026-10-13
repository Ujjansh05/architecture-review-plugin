import copy
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location("partial_encoder_test", ROOT / "scripts/train_partial_encoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_checkpoint(base, adapter_targets=False):
    (base / "encoder").mkdir(parents=True)
    (base / "tokenizer").mkdir()
    for name in ("config.json", "rl_agent_config.json", "architecture_review_manifest.json", "encoder/config.json",
                 "tokenizer/tokenizer.json"):
        (base / name).write_text('{"version":1}', encoding="utf-8")
    header = {f"encoder.layers.{index}.weight": {"shape": [2, 3]} for index in range(4)}
    if adapter_targets:
        shapes = {"attn.Wqkv": [12, 4], "attn.Wo": [4, 4], "mlp.Wi": [8, 4], "mlp.Wo": [4, 4]}
        header = {f"encoder.layers.{index}.{name}.weight": {"shape": shape}
                  for index in range(4) for name, shape in shapes.items()}
    header.update({"encoder.embeddings.weight": {"shape": [100, 2]},
                   "encoder.final_norm.weight": {"shape": [2]}, "head.weight": {"shape": [2, 2]},
                   "temperature": {"shape": [3]}, "__metadata__": {"format": "pt"}})
    raw = json.dumps(header).encode()
    (base / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    return base


def test_header_only_parameter_scope_and_estimate(trainer, tmp_path):
    base = fake_checkpoint(tmp_path / "base")
    one = trainer.checkpoint_plan(base, 1)
    two = trainer.checkpoint_plan(base, 2)
    assert one["selected_encoder_layers"] == [3]
    assert one["trainable_tensor_names"] == ["encoder.layers.3.weight", "head.weight"]
    assert one["estimated_trainable_parameters"] == 10
    assert two["estimated_trainable_parameters"] == 16
    assert two["estimated_delta_fp32_bytes"] == 64
    assert "encoder/config.json" in two["base_config_files_sha256"]
    assert "rl_agent_config.json" in two["base_config_files_sha256"]
    assert "architecture_review_manifest.json" in two["base_config_files_sha256"]


@pytest.mark.parametrize("name", ["rl_agent_config.json", "encoder/config.json", "architecture_review_manifest.json"])
def test_each_inference_configuration_is_fingerprinted(trainer, tmp_path, name):
    base = fake_checkpoint(tmp_path / "base")
    before = trainer.configuration_hashes(base)
    (base / name).write_text('{"version":2}', encoding="utf-8")
    after = trainer.configuration_hashes(base)
    assert before["base_config_sha256"] != after["base_config_sha256"]
    assert before["base_tokenizer_sha256"] == after["base_tokenizer_sha256"]


def test_tokenizer_changes_have_separate_fingerprint(trainer, tmp_path):
    base = fake_checkpoint(tmp_path / "base")
    before = trainer.configuration_hashes(base)
    (base / "tokenizer/tokenizer.json").write_text('{"version":2}', encoding="utf-8")
    after = trainer.configuration_hashes(base)
    assert before["base_config_sha256"] == after["base_config_sha256"]
    assert before["base_tokenizer_sha256"] != after["base_tokenizer_sha256"]


def test_layer_selection_uses_numeric_order_and_rejects_unknown_structure(trainer):
    names = [f"encoder.layers.{index}.weight" for index in range(12)] + ["head.weight"]
    layers, selected = trainer.selected_names(names, 2)
    assert layers == [10, 11]
    assert selected == {"encoder.layers.10.weight", "encoder.layers.11.weight", "head.weight"}
    with pytest.raises(ValueError, match="contiguous"):
        trainer.selected_names(["encoder.layers.0.weight", "encoder.layers.2.weight", "head.weight"], 1)
    with pytest.raises(ValueError, match="one or two"):
        trainer.selected_names(names, 3)


def test_gradient_accumulation_is_bounded_and_counts_partial_windows(trainer):
    argv = ["--data", ".", "--base", ".", "--output", "."]
    assert trainer.parse_args(argv).grad_accum == 8
    assert trainer.parse_args([*argv, "--grad-accum", "1"]).grad_accum == 1
    for invalid in ("0", "3", "16"):
        with pytest.raises(SystemExit):
            trainer.parse_args([*argv, "--grad-accum", invalid])
    assert trainer.optimizer_update_counts(120, 3, 8) == {
        "optimizer_updates_per_epoch": 15, "optimizer_updates_total": 45}
    assert trainer.optimizer_update_counts(360, 3, 1) == {
        "optimizer_updates_per_epoch": 360, "optimizer_updates_total": 1080}
    assert trainer.parse_args([*argv, "--epochs", "6"]).epochs == 6
    assert trainer.parse_args(argv).epochs == 3
    assert trainer.optimizer_update_counts(360, 6, 1) == {
        "optimizer_updates_per_epoch": 360, "optimizer_updates_total": 2160}
    with pytest.raises(ValueError, match="one to six epochs"):
        trainer.optimizer_update_counts(360, 7, 1)
    assert trainer.optimizer_update_counts(121, 2, 8) == {
        "optimizer_updates_per_epoch": 16, "optimizer_updates_total": 32}


def test_adapter_scope_is_header_only_complete_and_excludes_original_encoder_weights(trainer, tmp_path):
    base = fake_checkpoint(tmp_path / "base", adapter_targets=True)
    plan = trainer.checkpoint_plan(base, encoder_adapter_rank=8)
    assert plan["training_mode"] == "encoder_low_rank_adapters"
    assert plan["selected_encoder_layers"] == [0, 1, 2, 3]
    assert plan["original_encoder_weights_frozen"] is True
    assert len(plan["adapter_targets"]) == 16
    assert plan["estimated_trainable_parameters"] == 1412
    assert plan["estimated_delta_fp32_bytes"] == 1412 * 4
    assert all(not name.startswith("encoder.") or name.endswith((".lora_A", ".lora_B"))
               for name in plan["trainable_tensor_names"])
    with pytest.raises(ValueError, match="two-dimensional linear weight"):
        trainer.checkpoint_plan(fake_checkpoint(tmp_path / "wrong-targets"), encoder_adapter_rank=8)


def test_adapter_option_is_bounded_and_exclusive_with_partial_layers(trainer):
    argv = ["--data", ".", "--base", ".", "--output", "."]
    assert trainer.parse_args(argv).encoder_adapter_rank is None
    assert trainer.parse_args([*argv, "--encoder-adapter-rank", "8"]).encoder_adapter_rank == 8
    for value in ("0", "4", "16"):
        with pytest.raises(SystemExit):
            trainer.parse_args([*argv, "--encoder-adapter-rank", value])
    with pytest.raises(SystemExit):
        trainer.parse_args([*argv, "--last-layers", "2", "--encoder-adapter-rank", "8"])


def test_tiny_adapters_start_equivalent_train_with_checkpointing_and_round_trip_only_delta(trainer, tmp_path):
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    from torch.utils.checkpoint import checkpoint

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.Wqkv = torch.nn.Linear(4, 12)
            self.attn.Wo = torch.nn.Linear(4, 4)
            self.mlp = torch.nn.Module()
            self.mlp.Wi = torch.nn.Linear(4, 8)
            self.mlp.Wo = torch.nn.Linear(4, 4)

        def forward(self, value):
            query, key, val = self.attn.Wqkv(value).chunk(3, dim=-1)
            value = value + self.attn.Wo(query + key + val)
            left, right = self.mlp.Wi(value).chunk(2, dim=-1)
            return value + self.mlp.Wo(torch.tanh(left) * right)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.input = torch.nn.Linear(4, 4)
            self.encoder.layers = torch.nn.ModuleList([Layer(), Layer()])
            self.encoder.final_norm = torch.nn.LayerNorm(4)
            self.head = torch.nn.Linear(4, 3)

        def forward(self, value, checkpointed=False):
            value = self.encoder.input(value)
            for layer in self.encoder.layers:
                value = checkpoint(layer, value, use_reentrant=False) if checkpointed else layer(value)
            return self.head(self.encoder.final_norm(value))

    torch.manual_seed(17)
    model = Tiny()
    original_parameters = dict(model.named_parameters())
    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    tensors = {name: {"shape": list(value.shape)} for name, value in model.named_parameters()}
    plan = trainer.adapter_scope(tensors, 8)
    values = torch.randn(3, 4)  # No input gradients: the frozen prefix must still allow adapter gradients.
    expected = model(values).detach()
    layers, trainable, _ = trainer.configure_adapters(model, plan, seed=17)
    assert layers == [0, 1]
    assert torch.equal(model(values), expected)
    assert {name for name, value in model.named_parameters() if value.requires_grad} == trainable
    assert all(not parameter.requires_grad for name, parameter in original_parameters.items() if name.startswith("encoder."))
    assert all(torch.count_nonzero(value).item() == 0 for name, value in model.named_parameters() if name.endswith(".lora_B"))
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=.01)
    model(values, checkpointed=True).square().mean().backward()
    assert any(value.grad is not None and torch.count_nonzero(value.grad).item() > 0
               for name, value in model.named_parameters() if name.endswith(".lora_B"))
    assert all(parameter.grad is None for name, parameter in original_parameters.items() if name.startswith("encoder."))
    optimizer.step()
    assert all(torch.equal(parameter, original_state[name])
               for name, parameter in original_parameters.items() if name.startswith("encoder."))
    delta = {name: value.detach().clone() for name, value in model.named_parameters() if name in trainable}
    assert sum(value.numel() for value in delta.values()) == plan["estimated_trainable_parameters"]
    assert all(not name.startswith("encoder.") or name.endswith((".lora_A", ".lora_B")) for name in delta)
    path = tmp_path / "only-delta.safetensors"
    safe.save_file(delta, str(path))
    loaded = safe.load_file(str(path))
    assert set(loaded) == trainable
    assert all(torch.equal(loaded[name], value) for name, value in delta.items())
    restored = Tiny()
    restored.load_state_dict(original_state)
    trainer.configure_adapters(restored, plan, seed=29)
    missing, unexpected = restored.load_state_dict(loaded, strict=False)
    assert not unexpected and not trainable.intersection(missing)
    assert torch.equal(restored(values), model(values))


def test_implementation_fingerprints_cover_trainer_evaluation_and_library_versions(trainer, monkeypatch):
    monkeypatch.setattr(trainer.importlib.metadata, "version", lambda name: f"test-{name}")
    result = trainer.implementation_fingerprints()
    assert result["installed_library_versions"] == {name: f"test-{name}" for name in ("laya", "torch", "transformers")}
    assert set(result["implementation_sha256"]) == {
        "scripts/train_partial_encoder.py", "scripts/evaluate_backend.py",
        "src/arch_review/evaluation_v2.py", "src/arch_review/evaluation.py"}
    assert all(len(value) == 64 for value in result["implementation_sha256"].values())
    assert result["implementation_sha256"]["scripts/train_partial_encoder.py"] == trainer.file_digest(
        ROOT / "scripts/train_partial_encoder.py")


@pytest.mark.parametrize("last_layers", [1, 2])
def test_only_last_layers_and_head_train_and_all_dropout_is_zero(trainer, last_layers):
    class Parameter:
        requires_grad = True

        def requires_grad_(self, value):
            self.requires_grad = value

    class Dropout:
        p = .3

    params = {f"encoder.layers.{index}.weight": Parameter() for index in range(4)}
    params.update({name: Parameter() for name in ["encoder.embeddings.weight", "encoder.final_norm.weight", "head.weight"]})
    config = SimpleNamespace(attention_dropout=.1, embedding_dropout=.2, mlp_dropout=.3, classifier_dropout=.4)
    modules = [Dropout(), SimpleNamespace(attention_dropout=.2, dropout=.3)]
    model = SimpleNamespace(named_parameters=lambda: params.items(), modules=lambda: modules,
                            encoder=SimpleNamespace(config=config))
    layers, chosen, _ = trainer.configure_trainable(model, last_layers)
    assert layers == list(range(4 - last_layers, 4))
    assert {name for name, parameter in params.items() if parameter.requires_grad} == chosen
    assert not params["encoder.embeddings.weight"].requires_grad
    assert not params["encoder.final_norm.weight"].requires_grad
    assert params["head.weight"].requires_grad
    assert modules[0].p == modules[1].attention_dropout == modules[1].dropout == 0
    assert all(value == 0 for value in vars(config).values())


@pytest.mark.parametrize("dataset,train_count", [("backend-diagnostics-v1", 120), ("backend-augmented-v1", 360)])
def test_facts_rendering_preserves_frozen_labels_evidence_splits_and_never_formats_test(trainer, dataset, train_count):
    path = ROOT / "datasets" / dataset
    original, cases, metadata = trainer.prepare_data(path, "narrative")
    snapshot = copy.deepcopy(original)
    facts, _, provenance = trainer.prepare_data(path, "facts")
    assert {key: len(value) for key, value in facts.items()} == {"train": train_count, "development": 36, "calibration": 36}
    assert "test" not in facts
    assert metadata["test_formatting_performed"] is False
    assert provenance["facts_are_reference_declarations"] is True
    assert provenance["document_extraction_evaluated"] is False
    for split in facts:
        for before, after in zip(original[split], facts[split], strict=True):
            state = json.loads(after["state"])
            assert state["facts"] == cases[before["id"]]["check"]["facts"]
            assert state["evidence"] == json.loads(before["state"])["evidence"]
            assert {key: value for key, value in before.items() if key != "state"} == {
                key: value for key, value in after.items() if key != "state"}
            assert "expected" not in state and "id" not in state
    assert original == snapshot
    # Removing the context can merge formerly distinct frozen scenario families.
    assert metadata["formatted_inputs_split_isolated"] is True
    trainer.require_formatted_split_isolation(metadata)
    assert provenance["formatted_inputs_split_isolated"] is False
    assert provenance["formatted_input_split_overlaps"]["train:development"] == 12
    with pytest.raises(ValueError, match="overlap frozen"):
        trainer.require_formatted_split_isolation(provenance)


def test_changed_reference_cases_are_rejected_before_training(trainer, tmp_path):
    copied = tmp_path / "data"
    shutil.copytree(ROOT / "datasets/backend-diagnostics-v1", copied)
    path = copied / "cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["check"]["facts"]["unreviewed_edit"] = True
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen dataset manifest"):
        trainer.prepare_data(copied, "facts")


def test_resource_preflight_guards_disk_ram_and_cap(trainer, tmp_path):
    plan = {"estimated_delta_fp32_bytes": 100 * trainer.MIB, "checkpoint_bytes": trainer.GIB}
    args = [plan, tmp_path / "new", int(3.5 * trainer.GIB), trainer.GIB]
    good = trainer.resource_preflight(*args, available_bytes=4 * trainer.GIB, free_bytes=400 * trainer.MIB)
    assert good["can_run"]
    assert good["required_disk_bytes"] == 328 * trainer.MIB
    assert not trainer.resource_preflight(*args, available_bytes=4 * trainer.GIB, free_bytes=327 * trainer.MIB)["can_run"]
    assert not trainer.resource_preflight(*args, available_bytes=2 * trainer.GIB, free_bytes=400 * trainer.MIB)["can_run"]
    args[2] = int(1.5 * trainer.GIB)
    assert not trainer.resource_preflight(*args, available_bytes=4 * trainer.GIB, free_bytes=400 * trainer.MIB)["can_run"]


@pytest.mark.parametrize("free_mib,can_run", [(6000, True), (1000, False)])
def test_cuda_preflight_is_header_only_and_reserves_headroom(trainer, monkeypatch, free_mib, can_run):
    def query(command, **kwargs):
        assert command[0] == "nvidia-smi" and "--id=0" in command
        assert kwargs["timeout"] == 10
        return SimpleNamespace(stdout=f"6144, {free_mib}\n")

    monkeypatch.setattr(trainer.subprocess, "run", query)
    result = trainer.cuda_preflight({"estimated_delta_fp32_bytes": 100 * trainer.MIB, "checkpoint_bytes": trainer.GIB})
    assert result["can_run"] is can_run
    assert result["allocator_memory_fraction"] == .8
    assert result["required_free_vram_bytes"] == 2560 * trainer.MIB


@pytest.mark.parametrize("flag,value", [("--encoder-lr", "nan"), ("--head-lr", "inf"), ("--encoder-lr", ".1"),
                                       ("--epochs", "7"), ("--seed", "30"), ("--context-tokens", "8192"),
                                       ("--max-rss-gib", "4"), ("--min-available-gib", "0"),
                                       ("--timeout-seconds", "1801")])
def test_unbounded_training_configuration_is_rejected(trainer, flag, value):
    with pytest.raises(SystemExit):
        trainer.parse_args(["--data", ".", "--base", ".", "--output", ".", flag, value])


def test_supervisor_saves_stdout_stderr_and_measures_rss(trainer, tmp_path):
    command = [sys.executable, "-u", "-c", "import sys,time; print('output'); print('error',file=sys.stderr); time.sleep(.3)"]
    result = trainer.supervise(command, dict(os.environ), tmp_path, 5, trainer.GIB, 0, interval=.025)
    assert result["status"] == "completed"
    assert result["worker_exit_code"] == 0
    assert result["sampled_peak_process_tree_rss_bytes"] > 0
    assert result["sample_interval_seconds"] == .025
    assert "output" in (tmp_path / "worker.log").read_text()
    assert "error" in (tmp_path / "worker.log").read_text()


@pytest.mark.parametrize("timeout,rss_limit,expected", [(.15, 1024 ** 3, "timeout"), (5, 1, "worker_rss_limit")])
def test_supervisor_stops_only_its_bounded_worker(trainer, tmp_path, timeout, rss_limit, expected):
    import psutil

    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"],
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        command = [sys.executable, "-u", "-c", "import os,time; print(os.getpid()); time.sleep(10)"]
        result = trainer.supervise(command, dict(os.environ), tmp_path, timeout, rss_limit, 0, interval=.025)
        assert result["status"] == expected
        assert result["worker_exit_code"] != 0
        assert result["elapsed_seconds"] < 5
        assert unrelated.poll() is None
        content = (tmp_path / "worker.log").read_text().strip()
        if content:
            assert not psutil.pid_exists(int(content))
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_dry_run_imports_no_torch_and_creates_no_output(trainer, tmp_path):
    base = fake_checkpoint(tmp_path / "base")
    output = tmp_path / "never-created"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/train_partial_encoder.py"),
                             "--data", str(ROOT / "datasets/backend-diagnostics-v1"), "--base", str(base),
                             "--output", str(output), "--dry-run"], capture_output=True, text=True, timeout=15, check=True)
    report = json.loads(result.stdout)
    assert report["torch_imported"] is False
    assert report["model_downloaded"] is False
    assert report["test_evaluated"] is False
    assert not output.exists()
