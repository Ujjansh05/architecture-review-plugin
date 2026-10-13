"""Export an experimental adapter as a complete FP32 Laya checkpoint; no training.

Run export on the same base/delta recorded by the trainer, then verify in a fresh
process with --verify-only. Never reads final-test inputs or modifies the base.
"""
import argparse
import copy
import gc
import importlib.util
import json
import shutil
from pathlib import Path

from arch_review.evaluation_v2 import check_laya_usage, file_digest, ordered_questions

ROOT = Path(__file__).resolve().parents[1]


def helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def merge_adapters(model, plan):
    """W_merged = W_base + (alpha / rank) * B @ A; preserve original biases."""
    import torch
    from torch import nn

    count = 0
    with torch.no_grad():
        for name in sorted(plan["adapter_targets"]):
            module = model.get_submodule(name)
            if (not isinstance(module.base, nn.Linear) or module.lora_A.dtype != torch.float32
                    or module.lora_B.dtype != torch.float32 or module.base.weight.dtype != torch.float32
                    or module.scaling != plan["adapter_alpha"] / plan["adapter_rank"]):
                raise ValueError("Unexpected adapter layout or precision")
            update = (module.lora_B @ module.lora_A) * module.scaling
            if update.shape != module.base.weight.shape or not torch.isfinite(update).all():
                raise ValueError("Invalid adapter update")
            module.base.weight.add_(update)
            if not torch.isfinite(module.base.weight).all():
                raise ValueError("Merged weights are non-finite")
            parent, _, attribute = name.rpartition(".")
            setattr(model.get_submodule(parent), attribute, module.base)
            count += 1
    if any(".lora_" in name or ".base." in name for name in model.state_dict()):
        raise ValueError("Unmerged adapter tensors remain")
    return count


def comparison(reference, candidate, tolerance=5e-4):
    if not reference or len(reference) != len(candidate):
        raise ValueError("Comparison inputs differ")
    import math

    max_difference = 0.0
    for old, new in zip(reference, candidate, strict=True):
        if old["id"] != new["id"] or old["choice"] != new["choice"]:
            raise ValueError("Merged/reloaded model changed a decision")
        if set(old["probabilities"]) != set(new["probabilities"]):
            raise ValueError("Probability labels differ")
        for label, value in old["probabilities"].items():
            other = new["probabilities"][label]
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in (value, other)):
                raise ValueError("Invalid probabilities")
            max_difference = max(max_difference, abs(value - other))
    if max_difference > tolerance:
        raise ValueError(f"Probability drift {max_difference} exceeds {tolerance}")
    return {"compared": len(reference), "identical_labels": len(reference),
            "maximum_absolute_probability_difference": max_difference, "tolerance": tolerance}


def predict(agent, inputs, context):
    answers = []
    for row in inputs:
        check = next(iter(row["questions"]))
        result = agent.predict(row["state"], copy.deepcopy(row["questions"]), max_len=context, head_max_len=256)
        check_laya_usage(result, check)
        answer = result["answers"][check]
        answers.append({"id": row["id"], "choice": answer["choice"], "probabilities": answer["probabilities"]})
    return answers


def verify_files(directory):
    manifest = json.loads((directory / "architecture_review_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifact_sha256"].items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()) or file_digest(target) != digest:
            raise ValueError("Merged checkpoint checksum mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.verification.mkdir(parents=True, exist_ok=True)
    from arch_review.checkpoint import configure_checkpoint_loading

    configure_checkpoint_loading()
    import laya
    import torch
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(4)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.8, 0)
    if args.verify_only:
        manifest = verify_files(args.output)
        inputs = json.loads((args.verification / "inputs.json").read_text(encoding="utf-8"))
        reference = json.loads((args.verification / "adapter-answers.json").read_text(encoding="utf-8"))
        agent = laya.load(str(args.output), device=args.device)
        agent.amp_enabled, agent.dtype = False, torch.float32
        answers = predict(agent, inputs, manifest["context_tokens"])
        result = {"stage": "fresh_process_reload", "device": str(agent.device), "precision": "float32",
                  "autocast": False, **comparison(reference, answers), "test_evaluated": False,
                  "model_weights_sha256": manifest["artifact_sha256"]["model.safetensors"]}
        write_json(args.verification / f"reload-{args.device}.json", result)
        print(json.dumps(result), flush=True)
        return
    if not args.base or not args.experiment or args.output.exists():
        parser.error("Export requires --base, --experiment and a new --output directory")
    trainer, regression = helper("train_partial_encoder"), helper("evaluate_adapter_regressions")
    report, plan = regression.verified_experiment(args.base, args.experiment, trainer)
    if report["neutral_labels"] or report["context_tokens"] != 1024:
        raise ValueError("This export supports the selected narrative, non-neutral 1024-token candidate")
    # Existing development and regression examples only: this is a fidelity check,
    # not new checkpoint selection or evidence of improved accuracy.
    rows, _, _ = trainer.prepare_data(ROOT / "datasets/backend-augmented-v1", "narrative")
    inputs = []
    for row in rows["development"]:
        questions, _ = ordered_questions(row, neutral_labels=False)
        inputs.append({"id": row["id"], "state": row["state"], "questions": questions})
    cases, _ = regression.load_cases(ROOT / "datasets/executable-backend-v1")
    payloads, _ = regression.prepare_inputs(cases, False)
    for case, payload in zip(cases, payloads, strict=True):
        inputs.append({"id": case["id"], "state": payload["state"], "questions": payload["questions"]})
    write_json(args.verification / "inputs.json", inputs)
    if shutil.disk_usage(args.output.parent).free < 3 * 1024 ** 3:
        raise ValueError("Reserve at least 3 GiB for the FP32 checkpoint and compressed release")
    agent = laya.load(str(args.base), device=args.device)
    agent.amp_enabled, agent.dtype = False, torch.float32
    trainer.configure_adapters(agent.model, plan, report["seed"])
    delta = load_file(str(args.experiment / "experimental-delta.safetensors"), device="cpu")
    parameters = dict(agent.model.named_parameters())
    if (set(delta) != set(plan["trainable_tensor_names"])
            or any(t.dtype != torch.float32 or t.shape != parameters[n].shape or not torch.isfinite(t).all()
                   for n, t in delta.items())):
        raise ValueError("Invalid delta tensors")
    missing, unexpected = agent.model.load_state_dict(delta, strict=False)
    if unexpected or set(delta).intersection(missing):
        raise ValueError("Incomplete delta load")
    del delta, parameters
    gc.collect()
    agent.temperature = report["calibration"]["temperature"]
    agent.temperature_by_options = report["calibration"]["temperature_by_options"]
    agent.model.eval()
    before = predict(agent, inputs, report["context_tokens"])
    write_json(args.verification / "adapter-answers.json", before)
    print(json.dumps({"stage": "adapter_checked", "cases": len(before)}), flush=True)
    count = merge_adapters(agent.model, plan)
    after = predict(agent, inputs, report["context_tokens"])
    checked = comparison(before, after)
    args.output.mkdir()
    for name in ("config.json", "encoder/config.json", *report["base_tokenizer_files_sha256"]):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.base / name, target)
    cfg = copy.deepcopy(agent.cfg)
    cfg.update(temperature=agent.temperature, temperature_by_options=agent.temperature_by_options,
               model_name="archrivet-322m-experimental-v0.1.0", amp_dtype="fp32",
               training={"base_training": agent.cfg.get("training"), "fine_tuned_from_checkpoint": True,
                         "epochs_completed": report["epochs"], "selected_epoch": report["selected_epoch"],
                         "seed": report["seed"], "method": "merged_rank8_lora_and_decision_head"})
    write_json(args.output / "rl_agent_config.json", cfg)
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in agent.model.state_dict().items()},
              str(args.output / "model.safetensors"), metadata={"format": "pt"})
    artifacts = {p.relative_to(args.output).as_posix(): file_digest(p)
                 for p in sorted(args.output.rglob("*")) if p.is_file()}
    manifest = {"role": "experimental_merged", "model_id": "archrivet-322m-experimental-v0.1.0",
                "specialized": True, "standalone": True, "license": "Apache-2.0",
                "base_repo": "convaiinnovations/laya-multilingual",
                "base_revision": "1720e3e3357cfe1e281542e223f8273b0890ca34",
                "base_weights_sha256": report["base_weights_sha256"], "delta_sha256": report["delta_sha256"],
                "training_report_sha256": file_digest(args.experiment / "report.json"),
                "artifact_sha256": artifacts, "parameter_count": sum(p.numel() for p in agent.model.parameters()),
                "inference_precision": "float32", "autocast": False, "context_tokens": 1024,
                "calibration_status": "fitted_unvalidated", "recommended_for_production": False,
                "notice": "Experimental merged checkpoint. Development 25/36; separate regressions 4/12. "
                          "Not a general architecture expert. Confidence is not production failure risk."}
    write_json(args.output / "architecture_review_manifest.json", manifest)
    regression.verified_experiment(args.base, args.experiment, trainer)
    result = {"stage": "merged", "device": str(agent.device), "merged_linear_layers": count,
              "parameter_count": manifest["parameter_count"], **checked, "test_evaluated": False,
              "new_training_performed": False, "base_unchanged_verified": True,
              "weights_bytes": (args.output / "model.safetensors").stat().st_size,
              "weights_sha256": artifacts["model.safetensors"], "precision": "float32", "autocast": False}
    write_json(args.verification / "merge.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
