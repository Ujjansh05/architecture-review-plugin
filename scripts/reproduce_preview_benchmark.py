"""Reproduce the public 780-case synthetic benchmark with downloaded Laya checkpoints.

Install the 0.3 preview with its model dependencies. Each invocation evaluates one
version; use --format published with model-v0.1.0 and --format preview with the
new checkpoint. No training or downloads are performed by this script.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--format", choices=["published", "preview"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "docs/benchmarks"
    source = root / "2026-10-final-test.jsonl"
    record = json.loads((root / "2026-10-preview.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != record["final_test_sha256"]:
        raise ValueError("Benchmark inputs changed")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", TOKENIZERS_PARALLELISM="false")
    from arch_review.checkpoint import configure_checkpoint_loading
    from arch_review.decision_input import CRITERIA
    from arch_review.model_contract import format_input, verify_contract
    from arch_review.registry import REGISTRY
    from arch_review.research_fact_input import CACHE_NAMES
    configure_checkpoint_loading()
    import laya
    import torch
    with (args.checkpoint / "model.safetensors").open("rb") as stream:
        weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if weight_hash != record["models"][args.format]["weights_sha256"]:
        raise ValueError("Checkpoint is not the benchmarked release")
    if args.format == "preview":
        verify_contract(args.checkpoint, 2048)
    torch.set_num_threads(args.threads)
    agent = laya.load(str(args.checkpoint), device=args.device)
    agent.amp_enabled, agent.dtype = False, torch.float32
    results = []
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    for index, row in enumerate(rows):
        check = next(iter(row["questions"]))
        state = json.loads(row["state"])
        facts = state["facts"]
        if check == "cache_freshness":
            reverse = {v: k for k, v in CACHE_NAMES.items()}
            facts = {reverse[k]: v for k, v in facts.items()}
        if args.format == "published":
            # Matches v0.2.0a3 LayaBackend.judge, including its omitted context.
            payload = {"state": json.dumps({"facts": facts, "evidence": state["evidence"]}, ensure_ascii=False),
                       "questions": {check: {"type": "choice", "instructions": REGISTRY[check].definition,
                                             "criteria": dict(CRITERIA)}}}
            max_len = 1024
        else:
            payload = format_input(check, facts, state["evidence"], state.get("context", ""))
            max_len = 2048
        expected = row["expected"][check]
        result = {"id": row["id"], "check_id": check, "expected": expected}
        choices = []
        try:
            definition = payload["questions"][check]
            for order in itertools.permutations(definition["criteria"]):
                question = {**definition, "criteria": {k: definition["criteria"][k] for k in order}}
                answer = agent.predict(payload["state"], {check: question}, max_len=max_len, head_max_len=256)
                usage = answer.get("usage", {})
                if usage.get("truncated") or usage.get("state_tokens_dropped", 0):
                    raise ValueError("Truncated input")
                choice = answer["answers"][check]["choice"]
                if choice not in CRITERIA:
                    raise ValueError("Unknown verdict")
                choices.append(choice)
            result.update(predicted=choices[0], order_predictions=choices,
                          changed=len(set(choices)) > 1)
        except Exception as exc:
            result.update(predicted=None, error=f"{type(exc).__name__}: {exc}", order_predictions=choices)
        results.append(result)
        if index % 25 == 0:
            print(f"{index + 1}/{len(rows)}", flush=True)
    correct = sum(r["predicted"] == r["expected"] for r in results)
    report = {"format": args.format, "device": args.device, "weights_sha256": weight_hash,
              "count": len(results), "correct": correct, "accuracy": correct / len(results),
              "incorrect_approvals": sum(r["predicted"] == "supported" and r["expected"] != "supported" for r in results),
              "errors": sum("error" in r for r in results), "results": results,
              "note": "Synthetic cases with shared mechanisms. These inputs are now public and must not be reused as a future untouched test."}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
