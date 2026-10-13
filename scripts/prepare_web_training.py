"""Automatically turn permitted web passages into source-linked Laya examples."""
import argparse
import json
import shutil
from pathlib import Path

from arch_review.web_corpus import sha, write_json, write_jsonl
from arch_review.web_training import (
    export_families,
    load_corpus,
    read_rows,
    run_teacher,
    select_jobs,
    teacher_prompt,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codex", default=shutil.which("codex"), help="Already installed/authenticated Codex CLI")
    parser.add_argument("--model", help="Optional teacher model identifier; Laya remains the student")
    parser.add_argument("--max-documents", type=int, default=24, help="24 for a pilot; 0 uses all relevant documents")
    parser.add_argument("--batch-size", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--prepare-only", action="store_true", help="Save jobs without invoking a teacher")
    parser.add_argument("--responses", type=Path, help="Import families JSONL from a previous compatible teacher run")
    parser.add_argument("--resume", action="store_true", help="Reuse completed teacher batches after an interrupted run")
    args = parser.parse_args()
    _, docs, chunks = load_corpus(args.corpus)
    jobs = select_jobs(docs, chunks, args.max_documents, args.seed)
    if args.output.exists() and not args.resume:
        raise ValueError("Choose a new output directory; exports are immutable")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    jobs_path = args.output / "jobs.jsonl"
    if args.resume and jobs_path.exists():
        if read_rows(jobs_path) != jobs:
            raise ValueError("Source selection changed; use a new output directory")
    else:
        write_jsonl(jobs_path, jobs)
    if (args.output / "export").exists():
        raise ValueError("An export already exists; use a new output directory to regenerate it")
    write_json(args.output / "scope.json", {"corpus_documents": len(docs), "corpus_chunks": len(chunks),
               "selected_documents": len(jobs), "seed": args.seed, "test_split_assigned_before_generation": True})
    if args.prepare_only:
        print(json.dumps({"status": "jobs_prepared", "jobs": len(jobs)}))
        return
    if args.responses:
        responses = read_rows(args.responses)
    else:
        if not args.codex:
            raise ValueError("Pass --codex with an installed/authenticated CLI, or import --responses")
        responses = []
        for index in range(0, len(jobs), args.batch_size):
            batch = jobs[index:index + args.batch_size]
            directory = args.output / f"teacher-{index:04}"
            if args.resume and (directory / "run.json").exists():
                report = json.loads((directory / "run.json").read_text(encoding="utf-8"))
                if (report.get("prompt_sha256") != sha(teacher_prompt(batch).encode())
                        or report.get("model") != (args.model or "host_default_identifier_not_emitted")):
                    raise ValueError("Teacher configuration changed; use a new output directory")
                response = json.loads((directory / "response.json").read_text(encoding="utf-8"))["families"]
            else:
                attempt = 1
                original = directory
                while directory.exists():
                    directory = original.with_name(original.name + f"-attempt-{attempt}")
                    attempt += 1
                response = run_teacher(batch, directory, args.codex, model=args.model)
            responses.extend(response)
            write_jsonl(args.output / "responses.jsonl", responses)
            print(json.dumps({"documents_generated": len(responses), "documents_selected": len(jobs)}), flush=True)
    result = export_families(args.corpus, jobs, responses, args.output / "export", args.seed)
    print(json.dumps({"status": "exported", "families": result["accepted_families"],
                      "rejected": result["rejected_families"], "files": result["files"]}, indent=2))


if __name__ == "__main__":
    main()
