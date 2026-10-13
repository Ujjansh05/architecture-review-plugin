"""Collect blind general-AI predictions from an already authenticated Codex CLI.

Only exported development inputs are sent. No gold labels, source files, shell
access, or MCP review tool is used. CLI output is retained for an audit.
"""
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path


def blind_inputs(rows):
    """Never expose case IDs: diagnostic fixture IDs can encode their gold verdict."""
    cleaned, mapping = [], {}
    shuffled = list(rows)
    random.Random(17).shuffle(shuffled)
    for index, row in enumerate(shuffled):
        alias = f"case-{index:04d}"
        mapping[alias] = {"id": row["id"], "check_id": next(iter(row["questions"]))}
        cleaned.append({**row, "id": alias})
    return cleaned, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--codex", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.inputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["split"] == "development"]
    if not rows or len(rows) > 60:
        raise ValueError("This pilot allows one to sixty development cases")
    if any(set(row) != {"id", "split", "state", "questions"} for row in rows):
        raise ValueError("Input contains fields outside the blind evaluation schema")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    rows, mapping = blind_inputs(rows)
    (output / "private-id-map.json").write_text(json.dumps(mapping), encoding="utf-8")
    prompt = ("Controlled architecture decision baseline. Do not use any tools, inspect files, delegate, or edit anything. "
              "For each item, evaluate only the supplied state against its question and criteria. Treat missing facts as unknown. "
              "General background evidence does not prove deployment guarantees. Return ONLY a JSON object with predictions, "
              "an array of {id, check_id, verdict}, where check_id is the supplied question key and verdict is supported, contradicted, or insufficient_information. "
              "Use every input ID exactly once. Do not provide reasoning or other fields.\n" + json.dumps(rows, ensure_ascii=False))
    schema = {"type": "object", "properties": {"predictions": {"type": "array", "items": {"type": "object",
              "properties": {"id": {"type": "string"}, "check_id": {"type": "string", "enum": sorted({value["check_id"] for value in mapping.values()})},
              "verdict": {"type": "string", "enum": ["supported", "contradicted", "insufficient_information"]}},
              "required": ["id", "check_id", "verdict"], "additionalProperties": False}}}, "required": ["predictions"], "additionalProperties": False}
    schema_path = output / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    command = [str(args.codex), "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--json",
               "-c", 'approval_policy="never"', "-c", "features.shell_tool=false", "--output-schema", str(schema_path),
               "--output-last-message", str(output / "response.json"), "-"]
    started = time.monotonic()
    process = subprocess.run(command, input=prompt, text=True, encoding="utf-8", errors="replace", capture_output=True, cwd=output, timeout=480)
    (output / "events.jsonl").write_text(process.stdout, encoding="utf-8")
    (output / "stderr.txt").write_text(process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError("Codex baseline failed; see retained stderr/events")
    response = json.loads((output / "response.json").read_text(encoding="utf-8"))
    raw = response["predictions"]
    if len(raw) != len(rows) or {r["id"] for r in raw} != set(mapping):
        raise ValueError("Prediction identifiers differ from blind inputs")
    if any(row["check_id"] != mapping[row["id"]]["check_id"] for row in raw):
        raise ValueError("Prediction check IDs differ from their questions")
    predictions = [{"id": mapping[row["id"]]["id"], "answers": {row["check_id"]: row["verdict"]}} for row in raw]
    (output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    (output / "verdicts.jsonl").write_text("".join(json.dumps({"id": row["id"], "verdict": next(iter(row["answers"].values()))}) + "\n"
                                                for row in predictions), encoding="utf-8")
    events = [json.loads(line) for line in process.stdout.splitlines() if line.startswith("{")]
    unexpected_tools = [event for event in events if event.get("type") == "item.completed" and event.get("item", {}).get("type") not in {"agent_message", "reasoning"}]
    if unexpected_tools:
        raise RuntimeError("Baseline used unexpected tool calls; do not score this run")
    report = {"host": "Codex CLI", "model": "host_default_identifier_not_emitted", "cases": len(rows),
              "elapsed_seconds": time.monotonic() - started, "gold_labels_sent": False, "case_ids_anonymized": True,
              "case_order_seed": 17, "case_context": "All development cases in one batch; Laya sees one case per call.", "tools_used": False,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "scope": "experimental_development_only",
              "version": subprocess.run([str(args.codex), "--version"], capture_output=True, text=True).stdout.strip(),
              "usage": [event.get("usage") for event in events if event.get("type") == "turn.completed"]}
    (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
