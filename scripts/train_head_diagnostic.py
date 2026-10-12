"""Bounded CPU training diagnostic that saves only Laya's non-encoder weights.

This is an experimental head update on draft labels, NOT a deployable specialist.
The original checkpoint stays unchanged. Test data is never evaluated.
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from arch_review.training import validate_export


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def core(args):
    from arch_review.checkpoint import configure_checkpoint_loading

    configure_checkpoint_loading()
    import laya
    import torch
    from laya.calibrate import fit_temperature_map
    from laya.train import TrainConfig, calibration_records, items_from_rows, train_model
    from safetensors.torch import load_file, save_file

    from arch_review.evaluation import metrics

    torch.set_num_threads(4)
    torch.manual_seed(0)
    manifest, rows = validate_export(args.data, allow_unreviewed=True)
    agent = laya.load(str(args.base), device="cpu")
    model, tokenizer = agent.model, agent.tok
    head_keys = {name for name in model.state_dict() if not name.startswith("encoder.")}
    head_bytes = sum(value.numel() * value.element_size() for name, value in model.state_dict().items() if name in head_keys)
    if shutil.disk_usage(args.output).free < 2 * head_bytes + 64 * 1024 ** 2:
        raise ValueError("Insufficient storage for the small head artifact and headroom")
    original_hash = digest_file(args.base / "model.safetensors")
    started = time.monotonic()

    def encoder_digest():
        digest = hashlib.sha256()
        for name, tensor in model.encoder.state_dict().items():
            digest.update(name.encode())
            digest.update(memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B"))
        return digest.hexdigest()

    encoder_before = encoder_digest()
    items = {}
    for split in ["train", "development", "calibration"]:
        if any(len(tokenizer.encode(row["state"], add_special_tokens=False)) > 760 for row in rows[split]):
            raise ValueError("An input exceeds the inference context budget")
        items[split], skipped = items_from_rows(tokenizer, rows[split], 1024, 256)
        if any(skipped.values()) or len(items[split]) != len(rows[split]):
            raise ValueError("Training preprocessing skipped examples")

    def evaluate(perturb=False):
        predictions, changes = [], []
        model.eval()
        for row in rows["development"]:
            check, expected = next(iter(row["expected"].items()))
            result = agent.predict(row["state"], row["questions"], max_len=1024, head_max_len=256)
            if result.get("usage", {}).get("truncated"):
                raise ValueError("Inference truncated an input")
            answer = result["answers"][check]
            predictions.append({"id": row["id"], "check_id": check, "expected": expected,
                                "predicted": answer["choice"], "probabilities": answer["probabilities"]})
            if perturb:
                questions = copy.deepcopy(row["questions"])
                questions[check]["criteria"] = dict(reversed(list(questions[check]["criteria"].items())))
                flipped = agent.predict(row["state"], questions, max_len=1024, head_max_len=256)
                changes.append({"id": row["id"], "changed": answer["choice"] != flipped["answers"][check]["choice"]})
        return {"overall": metrics(predictions), "results": predictions, "option_order_changes": changes}

    before = evaluate()
    best = {"accuracy": -1.0, "epoch": None, "state": None}
    history = []

    def epoch_end(epoch, loss):
        result = evaluate()
        score = result["overall"]["accuracy"]
        history.append({"epoch": epoch + 1, "loss": float(loss), "development_accuracy": score})
        if score > best["accuracy"]:
            best.update(accuracy=score, epoch=epoch + 1,
                        state={name: tensor.detach().cpu().contiguous().clone() for name, tensor in model.state_dict().items() if name in head_keys})
        print(json.dumps({"epoch_completed": history[-1]}), flush=True)
        model.train()
        model.encoder.eval()

    config = TrainConfig(epochs=args.epochs, micro_batch=2, grad_accum=4, loss="soft-ce", shuffle_options=("choice",),
                         max_len=1024, head_max_len=256, freeze_encoder=True, calib_frac=0, calib_max=0, seed=0, log_every=5)
    train_model(model, tokenizer, items["train"], config, torch.device("cpu"), 1024, 256, on_epoch_end=epoch_end)
    missing, unexpected = model.load_state_dict(best["state"], strict=False)
    if unexpected or any(not key.startswith("encoder.") for key in missing):
        raise ValueError("Head artifact does not match the base model")
    after_unfitted = evaluate()
    records = calibration_records(model, tokenizer, items["calibration"], torch.device("cpu"), 1024, 256, batch_size=2)
    fitted = fit_temperature_map(records)
    agent.temperature = fitted["temperature"]
    agent.temperature_by_options = fitted.get("temperature_by_options", {})
    after = evaluate(perturb=True)
    path = args.output / "experimental-head.safetensors"
    save_file(best["state"], str(path))
    reloaded = load_file(str(path), device="cpu")
    if set(reloaded) != head_keys or any(not torch.equal(reloaded[key], best["state"][key]) for key in head_keys):
        raise ValueError("Head save/load round-trip mismatch")
    if encoder_digest() != encoder_before or digest_file(args.base / "model.safetensors") != original_hash:
        raise RuntimeError("The frozen encoder or original checkpoint changed")
    report = {"status": "experimental_head_training_completed", "device": "cpu", "epochs": args.epochs,
              "selected_epoch": best["epoch"], "train_cases": len(rows["train"]), "development_cases": len(rows["development"]),
              "calibration_cases": len(rows["calibration"]), "test_evaluated": False, "expert_reviewed_labels": False,
              "encoder_unchanged_verified": True, "base_checkpoint_unchanged_verified": True,
              "artifact_round_trip_verified": True, "base_weights_sha256": original_hash, "head_sha256": digest_file(path),
              "head_bytes": path.stat().st_size, "history": history, "before": before, "after_unfitted": after_unfitted,
              "after": after, "calibration": fitted, "calibration_status": "fitted_unvalidated_small_sample",
              "dataset_manifest": manifest, "elapsed_seconds": time.monotonic() - started, "paid_gpu_spend_usd": 0,
              "release_approved": False, "artifact_format": "non_encoder_state_dict_requires_exact_base_not_standalone",
              "notice": "Draft-label experiment; not full-model fine-tuning, human validation, or a deployable specialist."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["status", "selected_epoch", "elapsed_seconds", "head_bytes", "release_approved"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, choices=[1, 2], default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data, args.base, args.output = args.data.resolve(), args.base.resolve(), args.output.resolve()
    if args.worker:
        core(args)
        return
    validate_export(args.data, allow_unreviewed=True)
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--data", str(args.data), "--base", str(args.base),
               "--output", str(args.output), "--epochs", str(args.epochs)]
    try:
        subprocess.run(command, env=env, timeout=args.timeout_seconds, check=True)
    except subprocess.TimeoutExpired:
        (args.output / "timeout.json").write_text(json.dumps({"status": "timeout", "seconds": args.timeout_seconds}), encoding="utf-8")
        raise SystemExit("Diagnostic stopped at its wall-clock limit; no successful training result is claimed.")


if __name__ == "__main__":
    main()
