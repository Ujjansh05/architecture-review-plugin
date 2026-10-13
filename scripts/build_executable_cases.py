"""Original, executable local counterexamples; no rules/model-generated labels.

These are development regressions for bounded SQLite and in-memory scenarios,
not independent expert labels or evidence about distributed production systems.
"""
import argparse
import hashlib
import json
import os
import platform
import queue
import sqlite3
import subprocess
import sys
import tempfile
from collections import deque
from contextlib import closing
from pathlib import Path

CHECKS = ("retry_safety", "ack_durability", "queue_overload", "transaction_atomicity")
SPLIT = "development_regression"
SOURCE = "scripts/build_executable_cases.py"
CRASH_EXIT_CODE = 73
SUBPROCESS_TIMEOUT_SECONDS = 5
SCOPES = {
    "retry_safety": "One SQLite database; two sequential deliveries with the same retained operation ID; "
                    "no external effects, concurrent consumers, or identifier expiry.",
    "ack_durability": "One SQLite database on a local filesystem, DELETE journal, synchronous=FULL; "
                      "abrupt writer-process exit only. The OS and storage remain running; "
                      "host, power, zone and region failures are not tested.",
    "queue_overload": "Three immediate arrivals into an in-memory queue of capacity two, with no consumer "
                      "running during the burst; accepted-message preservation before processing only.",
    "transaction_atomicity": "One SQLite writer and a separate SQLite reader observe a debit/credit transfer "
                             "around an injected exception between writes; no distributed transactions.",
}
REQUIREMENTS = {
    "retry_safety": "Delivering the same operation twice must increment the protected counter only once.",
    "ack_durability": "Every acknowledged append must be present after the writer process exits abruptly.",
    "queue_overload": "Queue size must stay at or below two, and every accepted message must remain queued "
                      "until a consumer processes it. A full queue may reject an arrival before accepting it.",
    "transaction_atomicity": "Readers must see either the original balances (100, 0) or the completed "
                             "transfer balances (90, 10), including when a failure occurs between writes.",
}
MISSING_DESIGNS = {
    "retry_safety": "A worker consumes messages and increments a SQLite counter. Each message carries an "
                    "operation ID and can be delivered twice. The description does not specify whether "
                    "uniqueness is enforced or how duplicate detection and the counter update are committed.",
    "ack_durability": "A writer appends a message to SQLite and sends a success acknowledgement. The "
                      "description does not specify whether commit happens before acknowledgement.",
    "queue_overload": "A producer submits messages to an in-memory queue with capacity two. The "
                      "description does not specify what happens to accepted messages when the queue is full.",
    "transaction_atomicity": "A transfer handler debits one SQLite account and credits another. The "
                             "description does not specify a transaction boundary, failure recovery, "
                             "or what an independent reader sees between writes.",
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def retry_observation(good):
    with closing(sqlite3.connect(":memory:", isolation_level=None)) as connection:
        unique = " PRIMARY KEY" if good else ""
        connection.execute(f"CREATE TABLE processed (operation_id TEXT{unique})")
        connection.execute("CREATE TABLE effects (counter INTEGER NOT NULL)")
        connection.execute("INSERT INTO effects VALUES (0)")
        deliveries = []
        for attempt in (1, 2):
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("INSERT INTO processed VALUES (?)", ("operation-1",))
                connection.execute("UPDATE effects SET counter = counter + 1")
                connection.execute("COMMIT")
                action = "effect_committed"
            except sqlite3.IntegrityError:
                connection.execute("ROLLBACK")
                action = "duplicate_rejected_by_unique_constraint"
            deliveries.append({"attempt": attempt, "operation_id": "operation-1", "action": action,
                               "effect_count": connection.execute("SELECT counter FROM effects").fetchone()[0]})
        effects = connection.execute("SELECT counter FROM effects").fetchone()[0]
        ledger_rows = connection.execute("SELECT COUNT(*) FROM processed").fetchone()[0]
    return {"operation_id_reused": True, "deliveries": deliveries, "effect_count": effects,
            "ledger_rows": ledger_rows, "constraint": "Two deliveries must cause exactly one counter increment",
            "constraint_satisfied": effects == 1,
            "failure_injection": "First commit's queue acknowledgement is withheld; the same ID is redelivered. "
                                 "This retry schedule is simulated; no process is killed in this scenario."}


def crash_writer(database, good):
    connection = sqlite3.connect(database, isolation_level=None)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("INSERT INTO messages VALUES (?)", ("append-1",))
    if good:
        connection.execute("COMMIT")
    print(json.dumps({"event": "acknowledged", "operation_id": "append-1", "commit_before_ack": good}), flush=True)
    # Exit this isolated child immediately without connection cleanup or rollback.
    os._exit(CRASH_EXIT_CODE)


def ack_observation(good):
    with tempfile.TemporaryDirectory(prefix="archrivet-ack-regression-") as root:
        database = Path(root) / "ack.sqlite3"
        with closing(sqlite3.connect(database, isolation_level=None)) as connection:
            connection.execute("CREATE TABLE messages (operation_id TEXT PRIMARY KEY)")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--crash-child", str(database),
                                  "--variant", "good" if good else "flawed"], capture_output=True,
                                 text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS, creationflags=flags)
        if process.returncode != CRASH_EXIT_CODE:
            raise RuntimeError("Isolated crash worker did not reach its expected abrupt exit")
        acknowledgement = json.loads(process.stdout)
        if acknowledgement.get("event") != "acknowledged" or acknowledgement.get("operation_id") != "append-1":
            raise RuntimeError("Crash worker did not emit the expected acknowledgement")
        with closing(sqlite3.connect(database, isolation_level=None)) as recovered:
            persisted = [row[0] for row in recovered.execute("SELECT operation_id FROM messages ORDER BY operation_id")]
    return {"acknowledgement": acknowledgement, "child_exit_code": process.returncode,
            "persisted_after_process_exit": persisted, "constraint": "The acknowledged append must survive process exit",
            "constraint_satisfied": acknowledgement["operation_id"] in persisted,
            "journal_mode": "DELETE", "synchronous": "FULL", "failure_scope": "writer_process_only"}


def queue_observation(good):
    arrivals = ["message-1", "message-2", "message-3"]
    accepted, rejected, depths = [], [], []
    if good:
        backlog = queue.Queue(maxsize=2)
        for message in arrivals:
            try:
                backlog.put_nowait(message)
                accepted.append(message)
            except queue.Full:
                rejected.append(message)
            depths.append(backlog.qsize())
        retained = []
        while not backlog.empty():
            retained.append(backlog.get_nowait())
    else:
        backlog = deque(maxlen=2)
        for message in arrivals:
            backlog.append(message)
            accepted.append(message)
            depths.append(len(backlog))
        retained = list(backlog)
    missing = sorted(set(accepted) - set(retained))
    return {"capacity": 2, "arrivals": arrivals, "accepted": accepted, "rejected_before_accept": rejected,
            "depth_after_each_arrival": depths, "retained_before_consumption": retained,
            "missing_accepted_messages": missing, "consumer_ran_during_burst": False,
            "constraint": "Depth never exceeds two and no accepted message disappears before consumption",
            "constraint_satisfied": max(depths) <= 2 and not missing}


def atomicity_observation(good):
    with tempfile.TemporaryDirectory(prefix="archrivet-atomicity-regression-") as root:
        database = Path(root) / "accounts.sqlite3"
        with closing(sqlite3.connect(database, isolation_level=None)) as writer, \
                closing(sqlite3.connect(database, isolation_level=None)) as reader:
            writer.execute("PRAGMA journal_mode=DELETE")
            writer.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER NOT NULL)")
            writer.executemany("INSERT INTO accounts VALUES (?, ?)", [(1, 100), (2, 0)])

            def balances():
                return [row[0] for row in reader.execute("SELECT balance FROM accounts ORDER BY id")]

            original = balances()
            if good:
                writer.execute("BEGIN IMMEDIATE")
            writer.execute("UPDATE accounts SET balance = balance - 10 WHERE id = 1")
            during = balances()
            # The handler fails after its first write, before executing its second.
            try:
                raise RuntimeError("injected failure between debit and credit")
            except RuntimeError:
                if good:
                    writer.execute("ROLLBACK")
            after_failure = balances()
            complete = None
            if good:
                writer.execute("BEGIN IMMEDIATE")
                writer.execute("UPDATE accounts SET balance = balance - 10 WHERE id = 1")
                writer.execute("UPDATE accounts SET balance = balance + 10 WHERE id = 2")
                writer.execute("COMMIT")
                complete = balances()
    allowed = ([100, 0], [90, 10])
    return {"original_balances": original, "independent_reader_between_writes": during,
            "independent_reader_after_failure": after_failure, "successful_full_transfer": complete,
            "constraint": "Every reader observation is either [100, 0] or [90, 10]",
            "constraint_satisfied": all(value in allowed for value in (original, during, after_failure))
                                    and (complete is None or complete in allowed),
            "failure_injection": "Explicit exception between debit and credit; not a process/host crash"}


RUNNERS = {"retry_safety": retry_observation, "ack_durability": ack_observation,
           "queue_overload": queue_observation, "transaction_atomicity": atomicity_observation}


def architecture(check, good):
    if check == "retry_safety":
        return ("A SQLite worker commits the operation ledger insert and counter increment in one transaction. "
                + ("The ledger operation ID is a database-enforced PRIMARY KEY. " if good else
                   "The ledger operation ID is an ordinary column with no uniqueness constraint. ")
                + "The same operation ID is redelivered once after the first commit is not acknowledged.")
    if check == "ack_durability":
        return ("A SQLite writer inserts one append using a DELETE journal and synchronous=FULL. "
                + ("It commits before emitting success." if good else
                   "It emits success while the insert transaction is still uncommitted.")
                + " Immediately after emitting success, the writer process exits abruptly without cleanup.")
    if check == "queue_overload":
        return ("Three messages arrive while no consumer runs. "
                + ("The ingress uses queue.Queue(maxsize=2) and rejects a put_nowait call that raises Full, "
                   "before telling its producer the message was accepted." if good else
                   "The ingress uses deque(maxlen=2) and tells every producer its message was accepted. "
                   "Appending a third message silently evicts the oldest queued item."))
    return ("A SQLite handler transfers 10 units by debiting one account and crediting another. "
            + ("Both updates share one explicit transaction. Failure between writes triggers rollback. " if good else
               "Each update commits independently. Failure after debit leaves that commit in place. ")
            + "A separate connection reads balances between the writes and again after the failure.")


def explicit_facts(check, good):
    """Only mechanism declarations matching the code; these do not assign labels."""
    if check == "retry_safety":
        return {"requirements": {"prevent_duplicate_effects": True}, "design": {"retry": {
            "redelivery_possible": True, "all_effects_intrinsically_idempotent": False,
            "stable_message_id": True, "dedup_enforced": good, "dedup_atomic_with_effect": True,
            "external_effects": False}}}
    if check == "ack_durability":
        return {"requirements": {"acknowledged_operation": "append", "survive_failures": ["process_crash"]},
                "design": {"acknowledgement": {"acknowledged_operation": "append",
                                               "after_durable_commit": good}}}
    if check == "queue_overload":
        return {"requirements": {"bounded_queue_required": True, "preserve_accepted_messages": True},
                "design": {"queue": {"max_messages": 2, "policy": "reject_before_accept" if good else "drop",
                                     "admission_before_accept": good, "preserves_accepted_messages": good}}}
    return {"atomic_visibility_required": True, "multiple_writes": True,
            "boundary": "single_transaction" if good else "independent_writes", "covers_all_writes": good}


def make_packet():
    cases, observations = [], []
    source_hash = digest(Path(__file__).read_bytes())
    for check in CHECKS:
        family = "executable-local-" + check
        for good in (True, False):
            trace = RUNNERS[check](good)
            # Independent oracle: evaluate the measured scenario's explicit invariant.
            label = "supported" if trace["constraint_satisfied"] else "contradicted"
            variant = "good" if good else "flawed"
            observation_id = family + "-" + variant
            command = f"python {SOURCE} --scenario {check} --variant {variant}"
            observations.append({"id": observation_id, "check_id": check, "variant": variant, "scope": SCOPES[check],
                                 "command": command, "source_sha256": source_hash, "trace": trace})
            visible_trace = {key: value for key, value in trace.items() if key not in {"constraint", "constraint_satisfied"}}
            document = architecture(check, good) + "\n\nMeasured local trace:\n" + json.dumps(visible_trace, sort_keys=True)
            cases.append({"id": observation_id, "scenario_family": family, "split": SPLIT, "check_id": check,
                          "input": {"requirements": REQUIREMENTS[check] + " Scope: " + SCOPES[check],
                                    "documents": [{"id": "local-scenario", "text": document}],
                                    "requested_checks": [check], "checks": [{"check_id": check,
                                        "scope_id": "local-scenario", "facts": explicit_facts(check, good)}]},
                          "expected": label, "label_derivation": "measured_constraint",
                          "observation_id": observation_id, "command": command, "source_sha256": source_hash,
                          "provenance": "executable_local_regression", "expert_reviewed": False})
        cases.append({"id": family + "-missing", "scenario_family": family, "split": SPLIT, "check_id": check,
                      "input": {"requirements": REQUIREMENTS[check] + " Scope: " + SCOPES[check],
                                "documents": [{"id": "local-scenario", "text": MISSING_DESIGNS[check]}],
                                "requested_checks": [check]}, "expected": "insufficient_information",
                      "label_derivation": "information_removal_control", "observation_id": None,
                      "command": None, "source_sha256": source_hash,
                      "provenance": "executable_local_regression", "expert_reviewed": False,
                      "notice": "Decisive mechanism and measured trace are deliberately omitted from input; "
                                "both working and failing implementations remain compatible."})
    return cases, observations


README = """# Executable local backend regressions

This original packet contains 12 cases: four local scenarios, each with a working
implementation, a measured counterexample, and a missing-information control.
Eight runs measure SQLite or bounded in-memory queue behavior. Labels use measured
constraints, not the plugin's rules or a language model. Four incomplete cases omit
the decisive mechanism and trace; their labels indicate insufficient input.

All cases belong to `development_regression`. Do not mix them into training or an
independent final test. Related variants share a scenario family. These are not
expert-reviewed cases or a benchmark of general production architecture quality.

Reproduce all observations and verify the checked-in packet:

    python scripts/build_executable_cases.py --verify datasets/executable-backend-v1

Run one observation:

    python scripts/build_executable_cases.py --scenario ack_durability --variant flawed

Build an immutable new packet:

    python scripts/build_executable_cases.py --output .runtime/executable-backend-next

Every crash subprocess has a five-second deadline. Only the durability scenario
uses an actual abrupt process exit. The retry case simulates the missing queue
acknowledgement by redelivering the same ID; atomicity injects a handler exception.
SQLite tests do not reproduce host/power/zone failures, distributed transactions,
network partitions, concurrency at scale, identifier expiry, or external effects.
Queue tests have no concurrent consumer or process failure. A supported label means
the specific observed local constraint passed, not a universal production guarantee.

The manifest records source and packet hashes plus Python/SQLite runtime versions.
The input objects can be submitted to the V2 review API; observations and expected
labels remain a separate regression oracle. Partial structured facts intentionally
avoid claiming guarantees the small experiments did not measure. A backend may need
to interpret the accompanying prose and trace to reach the local-scenario verdict.
"""


def jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def write_packet(output):
    if output.exists():
        raise ValueError("Packets are immutable; choose a new output directory")
    cases, observations = make_packet()
    output.mkdir(parents=True, exist_ok=False)
    contents = {"cases.jsonl": jsonl(cases), "observations.jsonl": jsonl(observations), "README.md": README.encode()}
    for name, content in contents.items():
        (output / name).write_bytes(content)
    manifest = {"version": "executable-local-regressions-1", "split": SPLIT, "cases": len(cases),
                "measured_runs": len(observations), "information_removal_controls": len(cases) - len(observations),
                "source": SOURCE, "source_sha256": digest(Path(__file__).read_bytes()),
                "files": {name: digest(content) for name, content in contents.items()},
                "runtime": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                            "platform": platform.system()}, "subprocess_timeout_seconds": SUBPROCESS_TIMEOUT_SECONDS,
                "expert_reviewed": False, "training_eligible": False, "final_test_eligible": False,
                "production_quality_claim": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def verify_packet(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != digest(Path(__file__).read_bytes()):
        raise ValueError("Generator source changed; create a new regression packet")
    for name, expected in manifest["files"].items():
        if digest((directory / name).read_bytes()) != expected:
            raise ValueError(f"Packet hash mismatch: {name}")
    cases, observations = make_packet()
    if jsonl(cases) != (directory / "cases.jsonl").read_bytes():
        raise ValueError("Regenerated cases differ from the frozen packet")
    if jsonl(observations) != (directory / "observations.jsonl").read_bytes():
        raise ValueError("Re-executed observations differ from the frozen packet")
    return {"status": "verified", "cases": len(cases), "measured_runs": len(observations), "split": SPLIT}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    action.add_argument("--scenario", choices=CHECKS)
    action.add_argument("--crash-child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=("good", "flawed"), default="good")
    args = parser.parse_args()
    if args.crash_child:
        crash_writer(args.crash_child, args.variant == "good")
    elif args.scenario:
        print(json.dumps(RUNNERS[args.scenario](args.variant == "good"), indent=2))
    elif args.verify:
        print(json.dumps(verify_packet(args.verify), indent=2))
    else:
        print(json.dumps(write_packet(args.output), indent=2))


if __name__ == "__main__":
    main()
