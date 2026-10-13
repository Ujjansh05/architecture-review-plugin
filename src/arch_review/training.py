"""Training on explicit scenario splits; never use an automatic row-level calibration split."""
import contextlib
import hashlib
import json
import math
import multiprocessing as mp
import shutil
import sys
import time
import uuid
from pathlib import Path

from filelock import FileLock

from .settings import runtime_dir


def validate_export(directory: Path, allow_unreviewed=False):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("expert_reviewed") and not allow_unreviewed:
        raise ValueError("Training requires reviewed data; --allow-unreviewed explicitly marks a development experiment.")
    loaded, groups, signatures = {}, {}, {}
    for split in ["train", "development", "calibration", "test"]:
        name = split + ".jsonl"
        content = (directory / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"Dataset manifest hash mismatch: {name}")
        rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Empty split: {split}")
        for row in rows:
            group = row["scenario_family"]
            if groups.setdefault(group, split) != split:
                raise ValueError("Scenario family leakage in exported data")
            signature = json.dumps([row["state"], row["questions"]], sort_keys=True)
            if signatures.setdefault(signature, split) != split:
                raise ValueError("Identical model input leaked across exported splits")
        loaded[split] = rows
    if groups != manifest["family_splits"]:
        raise ValueError("Family split manifest differs from exported data")
    return manifest, loaded


def reserve_budget(rate, seconds, root=None):
    if not math.isfinite(rate) or rate < 0 or seconds < 1:
        raise ValueError("Training rate and timeout must be finite and nonnegative/positive.")
    # Round up to billed minutes and cents; reserve USD 5 for overhead outside this runner.
    maximum = math.ceil(math.ceil(seconds / 60) * rate / 60 * 100) / 100
    root = root or runtime_dir()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "training-budget.json"
    with FileLock(str(root / "training-budget.lock")):
        ledger = json.loads(target.read_text()) if target.exists() else {"cap_usd": 25, "reserve_usd": 5, "runs": []}
        allocated = sum(row["maximum_compute_usd"] for row in ledger["runs"])
        if allocated + maximum > 20:
            raise ValueError("This run would exceed the USD 20 compute allocation inside the USD 25 experiment cap.")
        entry = {"id": str(uuid.uuid4()), "maximum_compute_usd": maximum, "hourly_rate_usd": rate,
                 "timeout_seconds": seconds, "status": "reserved", "actual_provider_charge_usd": None}
        ledger["runs"].append(entry)
        target.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return entry


def _train_core(options, connection):
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from .checkpoint import configure_checkpoint_loading

            configure_checkpoint_loading()
            import torch
            from laya.calibrate import fit_temperature_map
            from laya.train import (
                TrainConfig,
                calibration_records,
                calibration_report,
                items_from_rows,
                load_checkpoint,
                save_checkpoint,
                train_model,
            )
            from safetensors.torch import load_file
            options = dict(options)
            directory, output = Path(options["data"]), Path(options["output"])
            manifest, rows = validate_export(directory, options["allow_unreviewed"])
            device = torch.device(options["device"])
            if device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable; select an actual GPU environment or explicitly use --device cpu.")
            torch.set_num_threads(4)
            torch.manual_seed(options["seed"])
            model, tokenizer, config = load_checkpoint(options["base"])
            model.to(device)
            max_len, head_len = 1024, 256
            items = {}
            for split in ["train", "development", "calibration"]:
                # Conservative context guard preserves the exact state used at inference.
                if any(len(tokenizer.encode(row["state"], add_special_tokens=False)) > max_len - head_len - 8 for row in rows[split]):
                    raise ValueError(f"{split} contains an over-budget state; shorten evidence/context before training.")
                items[split], skipped = items_from_rows(tokenizer, rows[split], max_len, head_len)
                if any(skipped.values()) or len(items[split]) != len(rows[split]):
                    raise ValueError(f"Training preprocessing skipped or changed the decision count in {split}: {skipped}")
            train_config = TrainConfig(epochs=options["epochs"], micro_batch=2, grad_accum=8,
                loss="soft-ce", shuffle_options=("choice",), max_len=max_len, head_max_len=head_len,
                seed=options["seed"], freeze_encoder=options["freeze_encoder"], calib_frac=0, calib_max=0)
            train_config.validate()
            output.mkdir(parents=True, exist_ok=True)
            best = {"accuracy": -1.0, "epoch": None}
            history = []
            def on_epoch(epoch, loss):
                records = calibration_records(model, tokenizer, items["development"], device, max_len, head_len, batch_size=2)
                accuracy = sum(int(record[1].argmax() == record[2].argmax()) for record in records) / len(records)
                history.append({"epoch": epoch, "loss": float(loss), "development_accuracy": accuracy})
                if accuracy > best["accuracy"]:
                    best.update(accuracy=accuracy, epoch=epoch)
                    save_checkpoint(model, tokenizer, dict(config, max_len=max_len, head_max_len=head_len), str(output))
                model.train()
                if options["freeze_encoder"]:
                    model.encoder.eval()
            train_model(model, tokenizer, items["train"], train_config, device, max_len, head_len, on_epoch_end=on_epoch)
            # The held-out test split never chooses the checkpoint or calibration parameters.
            model.load_state_dict(load_file(str(output / "model.safetensors"), device="cpu"))
            records = calibration_records(model, tokenizer, items["calibration"], device, max_len, head_len, batch_size=2)
            fitted = fit_temperature_map(records)
            calibration = calibration_report(records, fitted["temperature"])
            final_config = dict(config, max_len=max_len, head_max_len=head_len, fine_tuned=True,
                                temperature=fitted["temperature"])
            final_config.pop("temperature_by_options", None)
            if fitted.get("temperature_by_options"):
                final_config["temperature_by_options"] = fitted["temperature_by_options"]
            save_checkpoint(model, tokenizer, final_config, str(output))
            metadata = {"role": "specialized" if manifest["expert_reviewed"] else "experimental_specialized",
                        "calibration_status": "fitted_unvalidated", "expert_reviewed": manifest["expert_reviewed"],
                        "dataset_manifest": manifest, "selected_epoch": best["epoch"],
                        "history": history, "calibration": calibration, "base": options["base"],
                        "seed": options["seed"], "freeze_encoder": options["freeze_encoder"],
                        "notice": "Calibration fit requires independent validation; model quality is not release-approved."}
            (output / "architecture_review_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            connection.send({"status": "trained_candidate", "output": str(output), "metadata": metadata})
    except Exception as exc:
        connection.send({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def run_training(args):
    manifest, rows = validate_export(args.data, args.allow_unreviewed)
    if args.epochs < 1 or args.epochs > 4:
        raise ValueError("This pilot runner allows one to four epochs per experiment.")
    if args.dry_run:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.base / "tokenizer"), local_files_only=True)
        lengths = {split: [len(tokenizer.encode(row["state"], add_special_tokens=False)) for row in values]
                   for split, values in rows.items()}
        return {"status": "dry_run", "counts": {k: len(v) for k, v in rows.items()},
                "max_state_tokens": {k: max(v) for k, v in lengths.items()}, "state_token_budget": 760,
                "all_states_fit": all(max(v) <= 760 for v in lengths.values()),
                "expert_reviewed": manifest["expert_reviewed"], "gpu_spend_usd": 0}
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory; existing checkpoints will not be overwritten.")
    storage_parent = args.output.resolve()
    while not storage_parent.exists():
        storage_parent = storage_parent.parent
    checkpoint_bytes = sum(path.stat().st_size for path in args.base.rglob("*") if path.is_file())
    required_bytes = checkpoint_bytes + 256 * 1024 ** 2
    if shutil.disk_usage(storage_parent).free < required_bytes:
        raise ValueError(f"Training output needs at least {required_bytes:,} free bytes for a checkpoint and headroom; choose another output volume.")
    options = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    reservation = reserve_budget(args.hourly_rate_usd, args.timeout_seconds)
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_train_core, args=(options, child))
    started = time.monotonic()
    process.start()
    child.close()
    try:
        if not parent.poll(args.timeout_seconds):
            process.terminate()
            return {"status": "timeout", "reservation": reservation,
                    "notice": "The local training process was stopped; separately enforce the provider's job timeout."}
        result = parent.recv()
        result.update(reservation=reservation, elapsed_seconds=time.monotonic() - started)
        return result
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        parent.close()
        process.close()
