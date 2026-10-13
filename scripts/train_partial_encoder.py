"""Bounded CPU/CUDA diagnostics: partial encoder layers or low-rank adapters.

Only a delta is saved. This is an experimental template-data trial, not a
deployable checkpoint or independent accuracy evaluation. Test inference is forbidden.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

from arch_review.evaluation_v2 import (
    CONTEXT_LENGTHS,
    HEAD_TOKENS,
    LABELS,
    check_laya_usage,
    checked_prediction,
    file_digest,
    ordered_questions,
    selection_score,
    summarize,
    training_row,
    validate_backend_export,
)

GIB = 1024 ** 3
MIB = 1024 ** 2
ACTIVE_SPLITS = ("train", "development", "calibration")
LAYER = re.compile(r"^encoder\.layers\.(\d+)\.")
ADAPTER_TARGETS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def implementation_fingerprints():
    root = Path(__file__).resolve().parents[1]
    paths = ("scripts/train_partial_encoder.py", "scripts/evaluate_backend.py",
             "src/arch_review/evaluation_v2.py", "src/arch_review/evaluation.py")
    return {"implementation_sha256": {name: file_digest(root / name) for name in paths},
            "installed_library_versions": {name: importlib.metadata.version(name)
                                           for name in ("laya", "torch", "transformers")}}


def optimizer_update_counts(train_rows, epochs, grad_accum):
    if train_rows < 1 or epochs not in range(1, 7) or grad_accum not in (1, 2, 4, 8):
        raise ValueError("Update counts require positive rows, one to six epochs, and bounded accumulation")
    per_epoch = math.ceil(train_rows / grad_accum)  # The micro-batch is fixed at one.
    return {"optimizer_updates_per_epoch": per_epoch, "optimizer_updates_total": per_epoch * epochs}


def selected_names(names, last_layers):
    if last_layers not in (1, 2):
        raise ValueError("Only the final one or two encoder layers may be trained")
    layers = sorted({int(match.group(1)) for name in names if (match := LAYER.match(name))})
    if len(layers) < last_layers or layers != list(range(len(layers))):
        raise ValueError("Expected contiguous ModernBERT encoder.layers numbered from zero")
    chosen = layers[-last_layers:]
    selected = {name for name in names if
                (not name.startswith("encoder.") and name != "temperature")
                or ((match := LAYER.match(name)) and int(match.group(1)) in chosen)}
    if not any(not name.startswith("encoder.") for name in selected):
        raise ValueError("The checkpoint contains no decision-head parameters")
    return chosen, selected


def configuration_hashes(base):
    paths = list(base.glob("*.json"))
    if (base / "encoder" / "config.json").is_file():
        paths.append(base / "encoder" / "config.json")
    config = {path.relative_to(base).as_posix(): file_digest(path) for path in sorted(paths)}
    tokenizer = {path.relative_to(base).as_posix(): file_digest(path)
                 for path in sorted((base / "tokenizer").rglob("*")) if path.is_file()}
    if "rl_agent_config.json" not in config or "encoder/config.json" not in config or not tokenizer:
        raise ValueError("A complete local Laya configuration, encoder config and tokenizer are required")
    return {"base_config_files_sha256": config, "base_config_sha256": digest_object(config),
            "base_tokenizer_files_sha256": tokenizer, "base_tokenizer_sha256": digest_object(tokenizer)}


def adapter_scope(tensors, rank):
    if rank != 8:
        raise ValueError("This bounded adapter pilot supports rank eight only")
    selected_names(tensors, 1)  # Validate the encoder layer numbering and presence of a head.
    layers = sorted({int(match.group(1)) for name in tensors if (match := LAYER.match(name))})
    targets = {}
    names = {name for name in tensors if not name.startswith("encoder.") and name != "temperature"}
    count = sum(math.prod(tensors[name]["shape"]) for name in names)
    for index in layers:
        for suffix in ADAPTER_TARGETS:
            module_name = f"encoder.layers.{index}.{suffix}"
            shape = tensors.get(f"{module_name}.weight", {}).get("shape")
            if not isinstance(shape, list) or len(shape) != 2 or min(shape) < 1:
                raise ValueError(f"Adapter target {module_name} must have a two-dimensional linear weight")
            targets[module_name] = {"out_features": shape[0], "in_features": shape[1]}
            names.update((f"{module_name}.lora_A", f"{module_name}.lora_B"))
            count += rank * sum(shape)
    return {"selected_encoder_layers": layers, "trainable_tensor_names": sorted(names),
            "estimated_trainable_parameters": count, "estimated_delta_fp32_bytes": count * 4,
            "training_mode": "encoder_low_rank_adapters", "adapter_rank": rank, "adapter_alpha": 16,
            "adapter_targets": targets, "original_encoder_weights_frozen": True}


def checkpoint_plan(base, last_layers=1, encoder_adapter_rank=None):
    path = base / "model.safetensors"
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Incomplete safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= 16 * MIB:
            raise ValueError("Invalid safetensors header size")
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError("Incomplete safetensors metadata")
        header = json.loads(raw)
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    for value in tensors.values():
        if (not isinstance(value, dict) or not isinstance(value.get("shape"), list)
                or any(type(size) is not int or size < 0 for size in value["shape"])):
            raise ValueError("Invalid tensor shape metadata")
    if encoder_adapter_rank is not None:
        scope = adapter_scope(tensors, encoder_adapter_rank)
    else:
        layers, names = selected_names(tensors, last_layers)
        count = sum(math.prod(tensors[name]["shape"]) for name in names)
        scope = {"selected_encoder_layers": layers, "trainable_tensor_names": sorted(names),
                 "estimated_trainable_parameters": count, "estimated_delta_fp32_bytes": count * 4,
                 "training_mode": "partial_encoder_layers", "adapter_rank": None,
                 "original_encoder_weights_frozen": False}
    return {**scope, "checkpoint_header_sha256": hashlib.sha256(raw).hexdigest(),
            "checkpoint_bytes": path.stat().st_size, **configuration_hashes(base)}


def prepare_data(directory, state_format):
    if state_format not in ("narrative", "facts"):
        raise ValueError("State format must be narrative or facts")
    manifest, exported, cases, audit = validate_backend_export(directory)
    prepared = {}
    for split in ACTIVE_SPLITS:
        prepared[split] = []
        for original in exported[split]:
            row = copy.deepcopy(original)
            if state_format == "facts":
                state = json.loads(original["state"])
                row["state"] = json.dumps({"facts": cases[row["id"]]["check"]["facts"],
                                           "evidence": state["evidence"]}, ensure_ascii=False)
            prepared[split].append(row)
    model_inputs = {
        key: {digest_object({"state": json.loads(row["state"]), "questions": row["questions"]}) for row in value}
        for key, value in prepared.items()
    }
    overlaps = {f"{left}:{right}": len(model_inputs[left] & model_inputs[right])
                for left, right in itertools.combinations(ACTIVE_SPLITS, 2)}
    metadata = {"dataset_manifest_sha256": file_digest(directory / "manifest.json"),
                "dataset_manifest": manifest, "dataset_audit": audit, "state_format": state_format,
                "formatted_split_sha256": {key: digest_object(value) for key, value in prepared.items()},
                "unique_formatted_model_inputs": {key: len(value) for key, value in model_inputs.items()},
                "formatted_input_split_overlaps": overlaps,
                "formatted_inputs_split_isolated": not any(overlaps.values()),
                "facts_are_reference_declarations": state_format == "facts", "document_extraction_evaluated": False,
                "evidence_changed": False, "questions_or_labels_changed": False,
                "test_inference_performed": False, "test_formatting_performed": False}
    return prepared, cases, metadata


def require_formatted_split_isolation(metadata):
    if not metadata["formatted_inputs_split_isolated"]:
        raise ValueError("Formatted model inputs overlap frozen train/development/calibration splits; "
                         "do not train on this rendering or reshuffle held-out rows")


def existing_parent(path):
    while not path.exists():
        path = path.parent
    return path


def resource_preflight(plan, output, max_rss_bytes, min_available_bytes, available_bytes=None, free_bytes=None):
    import psutil

    available = psutil.virtual_memory().available if available_bytes is None else available_bytes
    free = shutil.disk_usage(existing_parent(output.resolve())).free if free_bytes is None else free_bytes
    disk_required = 2 * plan["estimated_delta_fp32_bytes"] + 128 * MIB
    # The supervisor enforces the actual RSS and available-RAM limits throughout.
    working_lower_bound = max(2 * GIB, plan["checkpoint_bytes"] + 3 * plan["estimated_delta_fp32_bytes"])
    memory_required = working_lower_bound + min_available_bytes
    failures = []
    if free < disk_required:
        failures.append("Insufficient disk for two delta artifacts plus 128 MiB headroom")
    if available < memory_required:
        failures.append("Insufficient available RAM for the estimated worker and reserved OS margin")
    if working_lower_bound > max_rss_bytes:
        failures.append("The estimated working set exceeds the configured worker RSS cap")
    return {"can_run": not failures, "blocking_reasons": failures, "free_disk_bytes": free,
            "required_disk_bytes": disk_required, "available_ram_bytes": available,
            "required_available_ram_bytes": memory_required, "estimated_working_set_lower_bound_bytes": working_lower_bound,
            "max_worker_tree_rss_bytes": max_rss_bytes, "min_system_available_bytes": min_available_bytes,
            "memory_estimate_is_not_a_guarantee": True}


def cuda_preflight(plan):
    """Inspect physical GPU zero without importing Torch or initializing CUDA."""
    result = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.total,memory.free",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=10)
    total_mib, free_mib = [int(value.strip()) for value in result.stdout.strip().split(",")]
    required = max(2 * GIB, plan["checkpoint_bytes"] + 3 * plan["estimated_delta_fp32_bytes"])
    cap = int(total_mib * MIB * .8)
    return {"physical_gpu_index": 0, "total_vram_bytes": total_mib * MIB, "free_vram_bytes": free_mib * MIB,
            "required_free_vram_bytes": required + 512 * MIB, "allocator_memory_fraction": .8,
            "allocator_cap_bytes": cap, "can_run": free_mib * MIB >= required + 512 * MIB and cap >= required,
            "limitation": "Allocator cap covers PyTorch allocations, not other processes or all CUDA runtime memory."}


def configure_trainable(model, last_layers):
    parameters = dict(model.named_parameters())
    layers, names = selected_names(parameters, last_layers)
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in names)
    return layers, names, disable_dropout(model)


def disable_dropout(model):
    disabled = 0
    for module in model.modules():
        if type(module).__name__.startswith("Dropout") and hasattr(module, "p"):
            module.p = 0.0
            disabled += 1
        for attr in ("attention_dropout", "dropout"):
            if isinstance(getattr(module, attr, None), (float, int)):
                setattr(module, attr, 0.0)
                disabled += 1
    config = getattr(getattr(model, "encoder", None), "config", None)
    if config is not None:
        for attr in ("attention_dropout", "embedding_dropout", "mlp_dropout", "classifier_dropout"):
            if hasattr(config, attr):
                setattr(config, attr, 0.0)
    return disabled


def wrap_linear_adapter(base, rank, alpha, generator):
    """An unmerged LoRA update; zero B leaves the original linear function unchanged."""
    import torch
    from torch import nn
    from torch.nn import functional as functional

    if not isinstance(base, nn.Linear) or rank != 8 or alpha != 16:
        raise ValueError("Adapters require an ordinary Linear and the fixed rank-eight, alpha-sixteen configuration")

    class AdapterLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.in_features, self.out_features = base.in_features, base.out_features
            self.scaling = alpha / rank
            self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                                   dtype=base.weight.dtype))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                                   dtype=base.weight.dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)

        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        def forward(self, value):
            update = functional.linear(functional.linear(value, self.lora_A), self.lora_B)
            return self.base(value) + update * self.scaling

    return AdapterLinear()


def configure_adapters(model, plan, seed):
    import torch
    from torch import nn

    if plan["training_mode"] != "encoder_low_rank_adapters":
        raise ValueError("Expected an adapter-specific scope plan")
    pending = []
    for name, dimensions in sorted(plan["adapter_targets"].items()):
        module = model.get_submodule(name)
        if (not isinstance(module, nn.Linear) or module.in_features != dimensions["in_features"]
                or module.out_features != dimensions["out_features"]):
            raise ValueError(f"Runtime adapter target differs from checkpoint metadata: {name}")
        parent_name, attribute = name.rsplit(".", 1)
        pending.append((model.get_submodule(parent_name), attribute, module))
    if not pending:
        raise ValueError("The adapter scope contains no linear targets")
    generator = torch.Generator(device=pending[0][2].weight.device).manual_seed(seed)
    for parent, attribute, module in pending:
        setattr(parent, attribute, wrap_linear_adapter(module, plan["adapter_rank"], plan["adapter_alpha"], generator))
    names = set(plan["trainable_tensor_names"])
    parameters = dict(model.named_parameters())
    if not names <= parameters.keys():
        raise ValueError("Runtime adapter parameters differ from the scope plan")
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in names)
    if any(torch.count_nonzero(parameters[name]).item() for name in names if name.endswith(".lora_B")):
        raise ValueError("Adapter B must be zero at initialization")
    return plan["selected_encoder_layers"], names, disable_dropout(model)


def core(args, plan):
    implementation = implementation_fingerprints()
    from arch_review.checkpoint import configure_checkpoint_loading

    configure_checkpoint_loading()
    import laya
    import torch
    from laya.calibrate import fit_temperature_map
    from laya.train import TrainConfig, calibration_records, items_from_rows, train_model
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but this Python environment has no usable CUDA device")
        torch.cuda.set_per_process_memory_fraction(.8, device=0)
        torch.cuda.reset_peak_memory_stats(device=0)
    started = time.monotonic()
    rows, cases, provenance = prepare_data(args.data, args.state_format)
    require_formatted_split_isolation(provenance)
    base_hash = file_digest(args.base / "model.safetensors")
    print(json.dumps({"stage": "loading_local_model", "download_performed": False}), flush=True)
    agent = laya.load(str(args.base), device=args.device)
    # Training and calibration below use FP32; keep decision evaluation on that same path.
    agent.amp_enabled = False
    agent.dtype = torch.float32
    model, tokenizer = agent.model, agent.tok
    base_model_parameters = sum(parameter.numel() for parameter in model.parameters())
    adapter_mode = plan["training_mode"] == "encoder_low_rank_adapters"
    layers, trainable, dropout_count = (configure_adapters(model, plan, args.seed) if adapter_mode
                                       else configure_trainable(model, args.last_layers))
    if trainable != set(plan["trainable_tensor_names"]):
        raise ValueError("Runtime parameters differ from the header-only scope plan")
    parameters = dict(model.named_parameters())
    if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        raise ValueError("This bounded pilot requires FP32 model parameters")
    if any(parameter.device.type != args.device for parameter in parameters.values()):
        raise ValueError("The model did not load entirely on the requested device")

    def parameter_digest(names):
        digest = hashlib.sha256()
        for name in sorted(names):
            digest.update(name.encode())
            raw = parameters[name].detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw).cast("B"))
        return digest.hexdigest()

    frozen = set(parameters) - trainable
    frozen_before = parameter_digest(frozen)
    encoder_names = {name for name in trainable if name.startswith("encoder.")}
    encoder_before = parameter_digest(encoder_names)
    items, lengths = {}, {}
    state_budget = args.context_tokens - HEAD_TOKENS - 8
    for split in ACTIVE_SPLITS:
        lengths[split] = max(len(tokenizer.encode(row["state"], add_special_tokens=False)) for row in rows[split])
        if lengths[split] > state_budget:
            raise ValueError(f"{split} exceeds the {state_budget}-token state budget; no truncation permitted")
        rendered = [training_row(row, args.neutral_labels) for row in rows[split]]
        items[split], skipped = items_from_rows(tokenizer, rendered, args.context_tokens, HEAD_TOKENS)
        if any(skipped.values()) or len(items[split]) != len(rows[split]):
            raise ValueError("Training preprocessing skipped or changed a decision")

    def evaluate(stage, perturb=False):
        model.eval()
        results = []
        for row in rows["development"]:
            check, expected = next(iter(row["expected"].items()))

            def predict(order=None):
                questions, mapping = ordered_questions(row, order, args.neutral_labels)
                answer = agent.predict(row["state"], questions, max_len=args.context_tokens, head_max_len=HEAD_TOKENS)
                check_laya_usage(answer, check)
                chosen = answer["answers"][check]
                return checked_prediction(chosen["choice"], chosen["probabilities"], mapping)

            predicted, probabilities = predict()
            result = {"id": row["id"], "scenario_family": row["scenario_family"], "check_id": check,
                      "expected": expected, "predicted": predicted, "probabilities": probabilities,
                      "critical": cases[row["id"]].get("critical", False), "error": None,
                      "perturbation_complete": False}
            if perturb:
                primary = tuple(row["questions"][check]["criteria"])
                orders = [{"order": list(order), "predicted": predicted if order == primary else predict(order)[0]}
                          for order in itertools.permutations(LABELS)]
                result.update(changed=any(item["predicted"] != predicted for item in orders),
                              perturbation_complete=True, perturbation_orders_scored=6, perturbation_orders=orders)
            results.append(result)
            if len(results) % 3 == 0:
                print(json.dumps({"stage": stage, "completed": len(results), "total": len(rows["development"])}), flush=True)
        return summarize(results, {key: provenance["dataset_audit"][key] for key in
                                  ("independently_reviewed", "final_test_minimum_counts_met", "release_eligible")})

    before = evaluate("before", perturb=True)
    best, history = {"score": (-1.0, 0), "epoch": None, "state": None}, []

    def on_epoch(epoch, loss):
        if not math.isfinite(float(loss)):
            raise ValueError("Training produced a non-finite loss")
        evaluation = evaluate(f"epoch_{epoch + 1}")
        score = selection_score(evaluation["overall"])
        history.append({"epoch": epoch + 1, "loss": float(loss), "development_macro_f1": score[0],
                        "development_false_supported": -score[1], "development_accuracy": evaluation["overall"]["accuracy"]})
        if score > best["score"]:
            best.update(score=score, epoch=epoch + 1,
                        state={name: parameters[name].detach().cpu().contiguous().clone() for name in trainable})
        print(json.dumps({"stage": "epoch_completed", **history[-1]}), flush=True)
        model.train()  # Dropout remains zero; the frozen parameters retain requires_grad=False.

    config = TrainConfig(epochs=args.epochs, micro_batch=1, grad_accum=args.grad_accum, encoder_lr=args.encoder_lr,
                         head_lr=args.head_lr, freeze_encoder=False, loss="soft-ce", shuffle_options=("choice",),
                         max_len=args.context_tokens, head_max_len=HEAD_TOKENS, calib_frac=0, calib_max=0,
                         seed=args.seed, log_every=5, amp=False, gradient_checkpointing=adapter_mode)
    config.validate()
    print(json.dumps({"stage": "training", "selected_layers": layers, "epochs": args.epochs,
                      "training_mode": plan["training_mode"], "gradient_accumulation": args.grad_accum,
                      "trainable_parameters": sum(parameters[name].numel() for name in trainable)}), flush=True)
    train_model(model, tokenizer, items["train"], config, device, args.context_tokens, HEAD_TOKENS,
                on_epoch_end=on_epoch)
    if best["state"] is None:
        raise RuntimeError("Training produced no selectable checkpoint")
    missing, unexpected = model.load_state_dict(best["state"], strict=False)
    if unexpected or trainable.intersection(missing):
        raise ValueError("Selected delta does not match the model")
    if any(not torch.isfinite(tensor).all() for tensor in best["state"].values()):
        raise ValueError("Selected weights contain non-finite values")
    after_unfitted = evaluate("after_unfitted")
    records = calibration_records(model, tokenizer, items["calibration"], device,
                                  args.context_tokens, HEAD_TOKENS, batch_size=1)
    fitted = fit_temperature_map(records)
    agent.temperature = fitted["temperature"]
    agent.temperature_by_options = fitted.get("temperature_by_options", {})
    after = evaluate("after", perturb=True)
    delta = args.output / "experimental-delta.safetensors"
    save_file(best["state"], str(delta))
    reloaded = load_file(str(delta), device="cpu")
    if set(reloaded) != trainable or any(not torch.equal(reloaded[name], best["state"][name]) for name in trainable):
        raise ValueError("Delta save/load round trip failed")
    if parameter_digest(frozen) != frozen_before:
        raise RuntimeError("A parameter outside the authorized training scope changed")
    config_after = configuration_hashes(args.base)
    if file_digest(args.base / "model.safetensors") != base_hash or config_after != {
        key: plan[key] for key in config_after
    }:
        raise RuntimeError("The original checkpoint or inference configuration changed")
    report = {"status": ("experimental_encoder_adapter_training_completed" if adapter_mode
                          else "experimental_partial_encoder_training_completed"), "release_approved": False,
              "training_mode": plan["training_mode"],
              "artifact_format": "trainable_parameter_delta_requires_exact_base_and_calibration_not_standalone",
              "device": args.device, "actual_model_device": str(next(model.parameters()).device),
              "seed": args.seed, "epochs": args.epochs, "selected_epoch": best["epoch"],
              "selected_encoder_layers": layers, "trainable_tensor_names": sorted(trainable),
              "base_model_parameters": base_model_parameters,
              "model_parameters_after_adapter_installation": sum(parameter.numel() for parameter in parameters.values()),
              "trainable_parameters": sum(parameters[name].numel() for name in trainable),
              "frozen_parameters_unchanged_verified": True, "base_checkpoint_unchanged_verified": True,
              "original_encoder_weights_frozen": adapter_mode,
              "selected_encoder_parameters_changed": parameter_digest(encoder_names) != encoder_before,
              "artifact_round_trip_verified": True, "base_weights_sha256": base_hash,
              **configuration_hashes(args.base), "delta_sha256": file_digest(delta), "delta_bytes": delta.stat().st_size,
              "dropout": 0, "dropout_attributes_disabled": dropout_count,
              "encoder_learning_rate": args.encoder_lr, "head_learning_rate": args.head_lr,
              "micro_batch": 1, "gradient_accumulation": args.grad_accum,
              "gradient_checkpointing": adapter_mode,
              "gradient_checkpointing_use_reentrant": False if adapter_mode else None,
              **optimizer_update_counts(len(rows["train"]), args.epochs, args.grad_accum),
              "selected_epoch_optimizer_updates": optimizer_update_counts(
                  len(rows["train"]), best["epoch"], args.grad_accum)["optimizer_updates_total"],
              **implementation,
              "context_tokens": args.context_tokens, "head_tokens": HEAD_TOKENS, "state_token_budget": state_budget,
              "max_state_tokens": lengths, "neutral_labels": args.neutral_labels,
              "selection_rule": "development_macro_f1_then_fewer_false_supported; earliest_epoch_on_exact_tie",
              "before": before, "after_unfitted": after_unfitted, "after": after, "history": history,
              "calibration": fitted, "calibration_status": "fitted_unvalidated",
              "test_evaluated": False, "expert_reviewed_labels": False,
              "same_input_and_evidence_before_after": True, "split_counts": {key: len(value) for key, value in rows.items()},
              **provenance, "elapsed_seconds": time.monotonic() - started, "paid_gpu_spend_usd": 0,
              "notice": "Template-label experiment. Delta is not installed in the plugin and is not an independently validated specialist."}
    if adapter_mode:
        report["adapter_configuration"] = {
            "rank": plan["adapter_rank"], "alpha": plan["adapter_alpha"],
            "targets": plan["adapter_targets"], "initialization": "seeded_kaiming_uniform_A_zero_B",
            "seed": args.seed, "merged_into_base": False,
            "initial_adapter_sha256": encoder_before, "selected_adapter_sha256": parameter_digest(encoder_names),
            "adapter_parameters_changed": parameter_digest(encoder_names) != encoder_before,
            "all_original_encoder_parameters_unchanged_verified": True,
            "loader_requirement": "Recreate these adapters on the exact base, then load the delta and reported calibration.",
        }
    if args.device == "cuda":
        report["gpu_resource_observations"] = {
            "device_name": torch.cuda.get_device_name(0), "allocator_memory_fraction": .8,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "precision": "float32", "autocast": False,
        }
    write_json(args.output / "report.json", report)
    print(json.dumps({"stage": "completed", "selected_epoch": best["epoch"], "delta_bytes": delta.stat().st_size,
                      "before_accuracy": before["overall"]["accuracy"], "after_accuracy": after["overall"]["accuracy"]}), flush=True)


def stop_owned_tree(child, process):
    import psutil

    try:
        members = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        members = []
    for member in reversed(members):
        try:
            member.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if child.poll() is None:
        child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)
    _, alive = psutil.wait_procs(members, timeout=2)
    for member in alive:
        try:
            member.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def supervise(command, env, output, timeout_seconds, max_rss_bytes, min_available_bytes, interval=.25):
    import psutil

    started, peak, samples, lowest_available = time.monotonic(), 0, 0, None
    status, position, last_update = "running", 0, 0.0
    log_path = output / "worker.log"
    with log_path.open("wb") as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        process = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                samples += 1
                try:
                    members = [process, *process.children(recursive=True)]
                    rss = 0
                    for member in members:
                        try:
                            rss += member.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    available = psutil.virtual_memory().available
                except psutil.AccessDenied:
                    status = "memory_monitor_unavailable"
                    break
                except psutil.NoSuchProcess:
                    continue
                peak = max(peak, rss)
                lowest_available = available if lowest_available is None else min(lowest_available, available)
                elapsed = time.monotonic() - started
                if rss > max_rss_bytes:
                    status = "worker_rss_limit"
                    break
                if available < min_available_bytes:
                    status = "system_available_ram_limit"
                    break
                if elapsed >= timeout_seconds:
                    status = "timeout"
                    break
                with log_path.open("rb") as reader:
                    reader.seek(position)
                    for line in reader:
                        if not line.endswith(b"\n"):
                            break
                        position += len(line)
                        try:
                            item = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            continue
                        if isinstance(item, dict) and "stage" in item:
                            print(json.dumps({"worker_progress": item}), flush=True)
                if elapsed - last_update >= 15:
                    print(json.dumps({"stage": "supervising", "elapsed_seconds": round(elapsed, 1),
                                      "sampled_peak_rss_bytes": peak}), flush=True)
                    last_update = elapsed
                time.sleep(interval)
            if status == "running":
                status = "completed" if child.returncode == 0 else "worker_failed"
        finally:
            stop_owned_tree(child, process)
    return {"status": status, "worker_exit_code": child.returncode, "sampled_peak_process_tree_rss_bytes": peak or None,
            "lowest_sampled_system_available_bytes": lowest_available, "samples": samples,
            "sample_interval_seconds": interval, "elapsed_seconds": time.monotonic() - started,
            "max_worker_tree_rss_bytes": max_rss_bytes, "min_system_available_bytes": min_available_bytes,
            "log_file": "worker.log", "limitations": "Sampled RSS is not an exact peak; shared pages may be counted twice."}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--last-layers", type=int, choices=(1, 2), default=1)
    scope.add_argument("--encoder-adapter-rank", type=int, choices=(8,), default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, choices=range(1, 7), default=3,
                        help="Local diagnostic epochs: 1–6, default 3; the 1800-second and resource caps still apply.")
    parser.add_argument("--grad-accum", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--seed", type=int, choices=(17, 29), default=17)
    parser.add_argument("--state-format", choices=("narrative", "facts"), default="narrative")
    parser.add_argument("--neutral-labels", action="store_true")
    parser.add_argument("--context-tokens", type=int, choices=CONTEXT_LENGTHS, default=1024)
    parser.add_argument("--encoder-lr", type=float, default=2.5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-rss-gib", type=float, default=3.5)
    parser.add_argument("--min-available-gib", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name, low, high in (("encoder_lr", 1e-6, 5e-5), ("head_lr", 1e-6, 5e-4),
                            ("max_rss_gib", 1.5, 3.5), ("min_available_gib", .5, 4.0)):
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            parser.error(f"{name} must be finite and between {low} and {high}")
    if not 1 <= args.timeout_seconds <= 1800:
        parser.error("timeout-seconds must be between 1 and 1800")
    args.data, args.base, args.output = args.data.resolve(), args.base.resolve(), args.output.resolve()
    return args


def main():
    args = parse_args()
    plan = checkpoint_plan(args.base, args.last_layers, args.encoder_adapter_rank)
    _, _, provenance = prepare_data(args.data, args.state_format)
    limits = resource_preflight(plan, args.output, int(args.max_rss_gib * GIB), int(args.min_available_gib * GIB))
    gpu_limits = cuda_preflight(plan) if args.device == "cuda" else None
    if args.dry_run:
        print(json.dumps({"status": "dry_run", **plan, "resources": limits,
                          "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
                          "state_format": args.state_format, "device": args.device, "gpu_resources": gpu_limits,
                          "unique_formatted_model_inputs": provenance["unique_formatted_model_inputs"],
                          "formatted_input_split_overlaps": provenance["formatted_input_split_overlaps"],
                          "data_can_train": provenance["formatted_inputs_split_isolated"],
                          "model_downloaded": False,
                          "torch_imported": "torch" in sys.modules, "test_evaluated": False}, indent=2))
        return
    require_formatted_split_isolation(provenance)
    if not limits["can_run"]:
        raise ValueError("; ".join(limits["blocking_reasons"]))
    if gpu_limits and not gpu_limits["can_run"]:
        raise ValueError("Insufficient free GPU memory within the bounded allocator budget")
    if args.worker:
        if os.environ.get("ARCH_REVIEW_PARTIAL_WORKER") != "1":
            raise ValueError("Use the supervised entry point")
        core(args, plan)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "preflight.json", {"plan": plan, "resources": limits, "gpu_resources": gpu_limits})
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0",
               PYTHONUNBUFFERED="1", ARCH_REVIEW_PARTIAL_WORKER="1", OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads))
    if args.device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = "0"
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--data", str(args.data),
               "--base", str(args.base), "--output", str(args.output),
               "--epochs", str(args.epochs), "--grad-accum", str(args.grad_accum),
               "--seed", str(args.seed), "--state-format", args.state_format,
               "--device", args.device,
               "--context-tokens", str(args.context_tokens), "--encoder-lr", str(args.encoder_lr),
               "--head-lr", str(args.head_lr), "--threads", str(args.threads),
               "--max-rss-gib", str(args.max_rss_gib), "--min-available-gib", str(args.min_available_gib)]
    if args.encoder_adapter_rank is not None:
        command.extend(["--encoder-adapter-rank", str(args.encoder_adapter_rank)])
    else:
        command.extend(["--last-layers", str(args.last_layers)])
    if args.neutral_labels:
        command.append("--neutral-labels")
    observations = supervise(command, env, args.output, args.timeout_seconds,
                             int(args.max_rss_gib * GIB), int(args.min_available_gib * GIB))
    write_json(args.output / "supervision.json", observations)
    report_path = args.output / "report.json"
    if observations["status"] != "completed" or not report_path.is_file():
        failure = {"status": "incomplete_experiment", "release_approved": False,
                   "resource_observations": observations}
        write_json(args.output / "failure.json", failure)
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.update(failure)
            write_json(report_path, report)
        raise SystemExit("Partial-encoder diagnostic did not complete; inspect worker.log and supervision.json.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["resource_observations"] = observations
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "delta_bytes": report["delta_bytes"],
                      "before_accuracy": report["before"]["overall"]["accuracy"],
                      "after_accuracy": report["after"]["overall"]["accuracy"], "release_approved": False}), flush=True)


if __name__ == "__main__":
    main()
