"""Bounded CPU training diagnostic that saves only Laya's non-encoder weights.

This is an experimental head update on draft labels, NOT a deployable specialist.
The original checkpoint stays unchanged. Test data is never evaluated.
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import shutil
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
    ordered_questions,
    selection_score,
    training_row,
)
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
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    manifest, rows = validate_export(args.data, allow_unreviewed=True)
    if manifest.get("version") == "backend-diagnostics-1":
        from arch_review.backend_dataset import audit_cases, digest

        source_cases = [json.loads(line) for line in (args.data / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
        if digest(json.dumps(source_cases, sort_keys=True).encode()) != manifest["source_cases_sha256"]:
            raise ValueError("Backend case provenance differs from the exported dataset")
        dataset_audit = audit_cases(source_cases)
    else:
        dataset_audit = {"independently_reviewed": bool(manifest.get("expert_reviewed")),
                         "release_eligible": False}
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
    items, context_lengths = {}, {}
    state_budget = args.context_tokens - HEAD_TOKENS - 8
    for split in ["train", "development", "calibration"]:
        context_lengths[split] = max(len(tokenizer.encode(row["state"], add_special_tokens=False)) for row in rows[split])
        if context_lengths[split] > state_budget:
            raise ValueError(f"{split} exceeds the explicit {state_budget}-token state budget; no truncation is allowed")
        rendered = [training_row(row, args.neutral_labels) for row in rows[split]]
        items[split], skipped = items_from_rows(tokenizer, rendered, args.context_tokens, HEAD_TOKENS)
        if any(skipped.values()) or len(items[split]) != len(rows[split]):
            raise ValueError("Training preprocessing skipped examples")

    def evaluate(perturb=False):
        predictions, changes = [], []
        model.eval()
        for row in rows["development"]:
            check, expected = next(iter(row["expected"].items()))
            def predict(order=None):
                questions, mapping = ordered_questions(row, order, args.neutral_labels)
                result = agent.predict(row["state"], questions, max_len=args.context_tokens, head_max_len=HEAD_TOKENS)
                check_laya_usage(result, check)
                answer = result["answers"][check]
                return checked_prediction(answer["choice"], answer["probabilities"], mapping)

            predicted, probabilities = predict()
            item = {"id": row["id"], "scenario_family": row["scenario_family"], "check_id": check,
                    "expected": expected, "predicted": predicted, "probabilities": probabilities,
                    "error": None, "perturbation_complete": False}
            if perturb:
                primary_order = tuple(row["questions"][check]["criteria"])
                answers = [{"order": list(order), "predicted": predicted if order == primary_order else predict(order)[0]}
                           for order in itertools.permutations(LABELS)]
                changed = any(answer["predicted"] != predicted for answer in answers)
                item.update(changed=changed, perturbation_complete=True, perturbation_orders_scored=len(answers))
                changes.append({"id": row["id"], "changed": changed, "orders": answers})
            predictions.append(item)
        overall = metrics(predictions)
        overall["inference_failures"] = 0  # Any failed call aborts this diagnostic instead of producing a success report.
        overall["perturbation_change_fraction"] = (sum(item["changed"] for item in changes) / len(changes)
                                                    if changes else None)
        return {"overall": overall,
                "per_check": {check: metrics([row for row in predictions if row["check_id"] == check])
                              for check in sorted({row["check_id"] for row in predictions})},
                "results": predictions, "option_order_changes": changes}

    before = evaluate(perturb=True)
    best = {"score": (-1.0, 0), "epoch": None, "state": None}
    history = []

    def epoch_end(epoch, loss):
        result = evaluate()
        score = selection_score(result["overall"])
        history.append({"epoch": epoch + 1, "loss": float(loss),
                        "development_accuracy": result["overall"]["accuracy"],
                        "development_macro_f1": score[0], "development_false_supported": -score[1]})
        if score > best["score"]:
            best.update(score=score, epoch=epoch + 1,
                        state={name: tensor.detach().cpu().contiguous().clone() for name, tensor in model.state_dict().items() if name in head_keys})
        print(json.dumps({"epoch_completed": history[-1]}), flush=True)
        model.train()
        model.encoder.eval()

    config = TrainConfig(epochs=args.epochs, micro_batch=2, grad_accum=4, loss="soft-ce", shuffle_options=("choice",),
                         max_len=args.context_tokens, head_max_len=HEAD_TOKENS, freeze_encoder=True,
                         calib_frac=0, calib_max=0, seed=args.seed, log_every=5)
    config.validate()
    train_model(model, tokenizer, items["train"], config, torch.device("cpu"), args.context_tokens, HEAD_TOKENS,
                on_epoch_end=epoch_end)
    if best["state"] is None:
        raise RuntimeError("Training produced no selectable development checkpoint")
    missing, unexpected = model.load_state_dict(best["state"], strict=False)
    if unexpected or any(not key.startswith("encoder.") for key in missing):
        raise ValueError("Head artifact does not match the base model")
    after_unfitted = evaluate()
    records = calibration_records(model, tokenizer, items["calibration"], torch.device("cpu"),
                                  args.context_tokens, HEAD_TOKENS, batch_size=2)
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
              "seed": args.seed, "context_tokens": args.context_tokens, "head_tokens": HEAD_TOKENS,
              "state_token_budget": state_budget, "max_state_tokens": context_lengths,
              "neutral_labels": args.neutral_labels, "same_input_and_evidence_before_after": True,
              "selection_rule": "development_macro_f1_then_fewer_false_supported; earliest_epoch_on_exact_tie",
              "selected_epoch": best["epoch"], "train_cases": len(rows["train"]), "development_cases": len(rows["development"]),
              "calibration_cases": len(rows["calibration"]), "test_evaluated": False, "expert_reviewed_labels": False,
              "encoder_unchanged_verified": True, "base_checkpoint_unchanged_verified": True,
              "artifact_round_trip_verified": True, "base_weights_sha256": original_hash, "head_sha256": digest_file(path),
              "head_bytes": path.stat().st_size, "history": history, "before": before, "after_unfitted": after_unfitted,
              "after": after, "calibration": fitted, "calibration_status": "fitted_unvalidated_small_sample",
              "dataset_manifest": manifest, "dataset_manifest_sha256": digest_file(args.data / "manifest.json"),
              "dataset_audit": dataset_audit, "elapsed_seconds": time.monotonic() - started, "paid_gpu_spend_usd": 0,
              "release_approved": False, "artifact_format": "non_encoder_state_dict_requires_exact_base_not_standalone",
              "notice": "Draft-label experiment; not full-model fine-tuning, human validation, or a deployable specialist."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in ["status", "selected_epoch", "elapsed_seconds", "head_bytes", "release_approved"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--context-tokens", type=int, choices=CONTEXT_LENGTHS, default=1024)
    parser.add_argument("--neutral-labels", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout_seconds < 1 or args.seed < 0:
        raise ValueError("Timeout must be positive and seed nonnegative")
    args.data, args.base, args.output = args.data.resolve(), args.base.resolve(), args.output.resolve()
    if args.worker:
        core(args)
        return
    validate_export(args.data, allow_unreviewed=True)
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--data", str(args.data), "--base", str(args.base),
               "--output", str(args.output), "--epochs", str(args.epochs), "--seed", str(args.seed),
               "--context-tokens", str(args.context_tokens)]
    if args.neutral_labels:
        command.append("--neutral-labels")
    try:
        usage = supervise(command, env, args.timeout_seconds)
        report_path = args.output / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["resource_observations"] = usage
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    except subprocess.TimeoutExpired:
        (args.output / "timeout.json").write_text(json.dumps({"status": "timeout", "seconds": args.timeout_seconds}), encoding="utf-8")
        raise SystemExit("Diagnostic stopped at its wall-clock limit; no successful training result is claimed.")


def supervise(command, env, timeout_seconds, sample_interval_seconds=.25):
    """Sample worker-tree RSS; shared pages may be counted twice and peaks can be missed."""
    import psutil

    started, samples, peak = time.monotonic(), 0, 0
    child = subprocess.Popen(command, env=env,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    process = psutil.Process(child.pid)
    try:
        while True:
            samples += 1
            try:
                members = [process, *process.children(recursive=True)]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                members = []
            total = 0
            for member in members:
                try:
                    total += member.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak = max(peak, total)
            status = child.poll()
            if status is not None:
                if status:
                    raise subprocess.CalledProcessError(status, command)
                break
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            time.sleep(min(sample_interval_seconds, remaining))
    finally:
        if child.poll() is None:
            try:
                descendants = process.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                descendants = []
            for member in descendants:
                try:
                    member.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
    return {"sampled_peak_process_tree_rss_bytes": peak, "sample_interval_seconds": sample_interval_seconds,
            "samples": samples, "supervised_elapsed_seconds": time.monotonic() - started,
            "scope": "training_worker_and_descendants; supervisor_excluded",
            "limitations": "Sampled RSS is not an exact peak; shared pages may be counted more than once."}


if __name__ == "__main__":
    main()
