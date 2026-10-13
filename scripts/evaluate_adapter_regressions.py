"""Offline, bounded adapter/base comparison on twelve executable local regressions.

This packet is development regression coverage, never independent production accuracy,
checkpoint selection, final-test evaluation or permission to promote a checkpoint.
"""
import argparse
import copy
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from arch_review.evaluation import metrics
from arch_review.evaluation_v2 import (
    HEAD_TOKENS,
    LABELS,
    check_laya_usage,
    checked_prediction,
    file_digest,
    ordered_questions,
)
from arch_review.evidence_v2 import PersistentEvidence
from arch_review.registry import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
PACKET_CASES_SHA256 = "54bd7d683fff2a5fd15b8767123d1c9f5c03e80d2316862cbc4d4d5684d7a511"
CHECKS = {"retry_safety", "ack_durability", "queue_overload", "transaction_atomicity"}


def trainer_helpers():
    spec = importlib.util.spec_from_file_location("regression_adapter_trainer", ROOT / "scripts/train_partial_encoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(directory):
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("version") != "executable-local-regressions-1"
            or manifest.get("split") != "development_regression"
            or manifest.get("training_eligible") is not False or manifest.get("final_test_eligible") is not False):
        raise ValueError("Only the frozen development regression packet is permitted")
    if file_digest(directory / "cases.jsonl") != PACKET_CASES_SHA256:
        raise ValueError("Executable regression cases differ from the frozen packet")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or file_digest(directory / name) != digest:
            raise ValueError("Regression packet file hash mismatch")
    if file_digest(ROOT / "scripts/build_executable_cases.py") != manifest["source_sha256"]:
        raise ValueError("Regression source hash mismatch; execution is not rerun by this evaluator")
    cases = [json.loads(line) for line in (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    if (len(cases) != 12 or len({case["id"] for case in cases}) != 12
            or {case["check_id"] for case in cases} != CHECKS
            or len({case["scenario_family"] for case in cases}) != 4
            or any(case["split"] != "development_regression" for case in cases)):
        raise ValueError("Expected twelve cases from four development regression families")
    for check in CHECKS:
        if Counter(case["expected"] for case in cases if case["check_id"] == check) != Counter(LABELS):
            raise ValueError("Each regression check must contain all three labels once")
    return cases, manifest


def render_payload(case, evidence, neutral_labels=False):
    """Whitelist public input fields; identifiers, labels and provenance never enter inference."""
    public, check = case["input"], case["check_id"]
    facts = next((item["facts"] for item in public.get("checks", []) if item["check_id"] == check), {})
    design = public["requirements"] + "\n\n" + "\n\n".join(doc["text"] for doc in public["documents"])
    if facts:
        design += "\n\nExplicit known facts:\n" + json.dumps(facts, ensure_ascii=False, sort_keys=True)
    records = evidence.search(check, facts)
    state = json.dumps({"design": design, "evidence": [{"id": record["id"], "lesson": record["lesson"]}
                                                        for record in records]}, ensure_ascii=False)
    questions = {check: {"type": "choice", "instructions": REGISTRY[check].definition,
                         "criteria": {"supported": "Explicit facts satisfy the requirement.",
                                      "contradicted": "Explicit facts establish a violation.",
                                      "insufficient_information": "Required facts are missing or conflicting."}}}
    questions, mapping = ordered_questions({"questions": questions}, neutral_labels=neutral_labels)
    return {"state": state, "questions": questions, "label_map": mapping}


def prepare_inputs(cases, neutral_labels):
    with tempfile.TemporaryDirectory(prefix="archrivet-regression-evidence-") as directory:
        evidence = PersistentEvidence(directory)
        payloads = [render_payload(case, evidence, neutral_labels) for case in cases]
        version = evidence.version
    return payloads, version


def verified_experiment(base, experiment, helper):
    report = read_json(experiment / "report.json")
    if (report.get("status") != "experimental_encoder_adapter_training_completed"
            or report.get("training_mode") != "encoder_low_rank_adapters"
            or report.get("original_encoder_weights_frozen") is not True
            or report.get("frozen_parameters_unchanged_verified") is not True
            or report.get("state_format") != "narrative"):
        raise ValueError("Require a completed narrative adapter-only experiment with a frozen original encoder")
    plan = helper.checkpoint_plan(base, encoder_adapter_rank=8)
    if file_digest(base / "model.safetensors") != report.get("base_weights_sha256"):
        raise ValueError("Base weight hash mismatch")
    for name, value in helper.configuration_hashes(base).items():
        if report.get(name) != value:
            raise ValueError(f"Base configuration/tokenizer hash mismatch: {name}")
    delta = experiment / "experimental-delta.safetensors"
    if file_digest(delta) != report.get("delta_sha256"):
        raise ValueError("Adapter delta hash mismatch")
    configuration = report.get("adapter_configuration", {})
    if (report.get("trainable_tensor_names") != plan["trainable_tensor_names"]
            or report.get("selected_encoder_layers") != plan["selected_encoder_layers"]
            or configuration.get("rank") != 8 or configuration.get("alpha") != 16
            or configuration.get("targets") != plan["adapter_targets"]
            or configuration.get("merged_into_base") is not False):
        raise ValueError("Adapter scope differs from the verified base checkpoint")
    expected_helper = report.get("implementation_sha256", {}).get("scripts/train_partial_encoder.py")
    if expected_helper != file_digest(ROOT / "scripts/train_partial_encoder.py"):
        raise ValueError("Adapter helper differs from the implementation recorded during training")
    calibration = report.get("calibration", {})
    temperatures = calibration.get("temperature")
    if not isinstance(temperatures, list) or not temperatures:
        raise ValueError("Invalid recorded candidate calibration vector")
    values = [*temperatures, *calibration.get("temperature_by_options", {}).values()]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Invalid recorded candidate calibration")
    if (report.get("context_tokens") not in (1024, 2048, 4096)
            or report.get("head_tokens") != HEAD_TOKENS or type(report.get("neutral_labels")) is not bool):
        raise ValueError("Unsupported recorded inference configuration")
    return report, plan


def check_preprocessing(payloads, tokenizer, context_tokens, head_builder):
    """Check the complete state and question head without attaching any gold labels."""
    lengths = []
    for payload in payloads:
        length = len(tokenizer.encode(payload["state"], add_special_tokens=False))
        if length > context_tokens - HEAD_TOKENS - 8:
            raise ValueError("Regression input exceeds the state budget; truncation is forbidden")
        check, question = next(iter(payload["questions"].items()))
        ids, markers, stats = head_builder(tokenizer, check, question, HEAD_TOKENS)
        if (len(markers) != 3 or stats.get("options_distinct") != 3 or stats.get("options") != 3
                or len(ids) > HEAD_TOKENS or any(marker >= context_tokens for marker in markers)
                or stats.get("truncated")):
            raise ValueError("Regression question preprocessing dropped or collapsed decision options")
        lengths.append(length)
    return {"cases_preprocessed": len(lengths), "max_state_tokens": max(lengths), "truncation_permitted": False}


def score(results):
    def summarize(rows):
        return {**metrics(rows),
                "missed_defects": sum(row["expected"] == "contradicted" and row["predicted"] != "contradicted"
                                      for row in rows),
                "false_approvals": sum(row["predicted"] == "supported" and row["expected"] != "supported"
                                       for row in rows),
                "unknown_predictions": sum(row["predicted"] == "insufficient_information" for row in rows),
                "expected_unknowns": sum(row["expected"] == "insufficient_information" for row in rows),
                "inference_failures": sum(bool(row.get("error")) for row in rows)}
    return {"overall": summarize(results),
            "families": {family: summarize([row for row in results if row["scenario_family"] == family])
                         for family in sorted({row["scenario_family"] for row in results})}, "results": results}


def run_worker(args, helper, report, plan):
    from arch_review.checkpoint import configure_checkpoint_loading

    configure_checkpoint_loading()
    versions = {name: importlib.metadata.version(name) for name in ("laya", "torch", "transformers")}
    if versions != report.get("installed_library_versions"):
        raise ValueError("Inference library versions differ from the recorded adapter experiment")
    import laya
    import torch
    from laya.common import build_head, render_options
    from laya.train import to_internal
    from safetensors.torch import load_file

    frozen = read_json(args.output / "inputs.json")
    preflight = read_json(args.output / "preflight.json")
    if (helper.digest_object(frozen) != preflight["inputs_sha256"]
            or file_digest(args.experiment / "report.json") != preflight["training_report_sha256"]):
        raise ValueError("Frozen input or experiment report changed before inference")
    cases, _ = load_cases(args.data)
    if len(frozen) != len(cases):
        raise ValueError("Frozen regression input count mismatch")
    torch.set_num_threads(args.threads)
    torch.manual_seed(report["seed"])
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        torch.cuda.set_per_process_memory_fraction(.8, device=0)
        torch.cuda.reset_peak_memory_stats(0)
    agent = laya.load(str(args.base), device=args.device)
    agent.amp_enabled, agent.dtype = False, torch.float32
    if any(parameter.dtype != torch.float32 or parameter.device.type != args.device
           for parameter in agent.model.parameters()):
        raise ValueError("Base must load entirely in FP32 on the requested device")
    def complete_head(tok, check, question, budget):
        internal = to_internal(check, question)
        if any(len(tok.encode(" " + option.replace(tok.mask_token, " "), add_special_tokens=False)) > 48
               for option in render_options(internal)):
            raise ValueError("Regression option text exceeds Laya's per-option token cap")
        encoded = build_head(tok, internal, budget)
        if encoded[0] != build_head(tok, internal, 65536)[0]:
            raise ValueError("Regression question head was truncated")
        return encoded

    lengths = check_preprocessing(frozen, agent.tok, report["context_tokens"], complete_head)

    def evaluate(stage):
        agent.model.eval()
        results = []
        for case, payload in zip(cases, frozen, strict=True):
            result = {name: case[name] for name in ("id", "scenario_family", "check_id", "expected")}
            try:
                answer = agent.predict(payload["state"], copy.deepcopy(payload["questions"]),
                                       max_len=report["context_tokens"], head_max_len=HEAD_TOKENS)
                check_laya_usage(answer, case["check_id"])
                decision = answer["answers"][case["check_id"]]
                verdict, probabilities = checked_prediction(decision["choice"], decision["probabilities"],
                                                              payload["label_map"])
                result.update(predicted=verdict, probabilities=probabilities, error=None)
            except (ValueError, KeyError, RuntimeError) as exc:
                result.update(predicted=None, error=f"{type(exc).__name__}: {exc}")
            results.append(result)
            print(json.dumps({"stage": stage, "completed": len(results), "total": len(frozen)}), flush=True)
        return score(results)

    baseline_calibration = {"temperature": copy.deepcopy(agent.temperature),
                            "temperature_by_options": copy.deepcopy(agent.temperature_by_options)}
    before = evaluate("base_regressions")
    helper.configure_adapters(agent.model, plan, report["seed"])
    delta = load_file(str(args.experiment / "experimental-delta.safetensors"), device="cpu")
    parameters = dict(agent.model.named_parameters())
    if (set(delta) != set(plan["trainable_tensor_names"])
            or any(tensor.shape != parameters[name].shape or tensor.dtype != torch.float32
                   or not torch.isfinite(tensor).all() for name, tensor in delta.items())):
        raise ValueError("Delta contains unexpected tensors, shapes, precision or non-finite weights")
    missing, unexpected = agent.model.load_state_dict(delta, strict=False)
    if unexpected or set(delta).intersection(missing):
        raise ValueError("Delta could not be loaded completely")
    agent.temperature = report["calibration"]["temperature"]
    agent.temperature_by_options = report["calibration"].get("temperature_by_options", {})
    after = evaluate("candidate_regressions")
    verified_experiment(args.base, args.experiment, helper)
    result = {"status": "completed", "evaluation_kind": "machine_executed_local_regression",
              "split": "development_regression", "case_count": 12, "scenario_family_count": 4,
              "base": before, "candidate": after, "preprocessing": lengths,
              "baseline_calibration": baseline_calibration, "candidate_calibration": report["calibration"],
              "candidate_calibration_status": report.get("calibration_status"),
              "same_input_and_evidence": True, "precision": "float32", "autocast": False,
              "device": args.device, "inputs_sha256": preflight["inputs_sha256"],
              "installed_library_versions": versions, "evaluator_sha256": preflight["evaluator_sha256"],
              "training_report_sha256": preflight["training_report_sha256"],
              "base_weights_sha256": report["base_weights_sha256"], "delta_sha256": report["delta_sha256"],
              **helper.configuration_hashes(args.base), "test_evaluated": False,
              "checkpoint_selection_performed": False, "release_approved": False,
              "hidden_measurements_used": False, "execution_rerun": False,
              "notice": "Twelve toy SQLite/process-crash/queue regressions, not independent production accuracy. "
                        "Public measured traces are inputs. Missing controls omit them. No checkpoint promotion."}
    if args.device == "cuda":
        result["gpu"] = {"allocator_memory_fraction": .8, "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
    helper.write_json(args.output / "report.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "experiment", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 1800:
        parser.error("timeout-seconds must be between 1 and 1800")
    for name in ("base", "experiment", "data", "output"):
        setattr(args, name, getattr(args, name).resolve())
    helper = trainer_helpers()
    report, plan = verified_experiment(args.base, args.experiment, helper)
    cases, manifest = load_cases(args.data)
    if args.worker:
        if os.environ.get("ARCH_REVIEW_REGRESSION_WORKER") != "1":
            raise ValueError("Use the supervised entry point")
        run_worker(args, helper, report, plan)
        return
    payloads, evidence_version = prepare_inputs(cases, report["neutral_labels"])
    preflight = {"inputs_sha256": helper.digest_object(payloads), "evidence_version": evidence_version,
                 "packet_manifest_sha256": file_digest(args.data / "manifest.json"), "packet": manifest,
                 "training_report_sha256": file_digest(args.experiment / "report.json"),
                 "evaluator_sha256": file_digest(Path(__file__)), "plan": plan, "device": args.device,
                 "test_evaluated": False, "download_performed": False}
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "case_count": len(payloads),
                          "inputs_sha256": preflight["inputs_sha256"], "evidence_version": evidence_version,
                          "training_report_sha256": preflight["training_report_sha256"],
                          "base_weights_sha256": report["base_weights_sha256"],
                          "delta_sha256": report["delta_sha256"], "artifact_hashes_verified": True,
                          "device": args.device, "model_loaded": False, "test_evaluated": False,
                          "torch_imported": "torch" in sys.modules, "download_performed": False}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    helper.write_json(args.output / "inputs.json", payloads)
    helper.write_json(args.output / "preflight.json", preflight)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               USE_TF="0", PYTHONUNBUFFERED="1", ARCH_REVIEW_REGRESSION_WORKER="1")
    if args.device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = "0"
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--device", args.device,
               "--threads", str(args.threads)]
    for name in ("base", "experiment", "data", "output"):
        command.extend(["--" + name, str(getattr(args, name))])
    supervision = helper.supervise(command, env, args.output, args.timeout_seconds,
                                   int(3.5 * helper.GIB), int(.5 * helper.GIB))
    helper.write_json(args.output / "supervision.json", supervision)
    if supervision["status"] != "completed" or not (args.output / "report.json").is_file():
        raise SystemExit("Regression evaluation incomplete; inspect supervision.json and worker.log")
    print(json.dumps({"status": "completed", "case_count": 12, "release_approved": False}))


if __name__ == "__main__":
    main()
