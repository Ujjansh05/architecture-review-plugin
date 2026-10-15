"""Compare completed scoped declarations, without claiming a deployed fix."""
from .documents import flatten


def compare_results(before, after):
    left = {(f["scope_id"], f["check_id"]): f for f in before["checks"]}
    right = {(f["scope_id"], f["check_id"]): f for f in after["checks"]}
    changes = []
    for scope, check in sorted(left.keys() | right.keys()):
        a, b = left.get((scope, check)), right.get((scope, check))
        old = flatten(a["facts"]) if a else {}
        new = flatten(b["facts"]) if b else {}
        facts = [{"fact_path": path, "before": old.get(path), "after": new.get(path)}
                 for path in sorted(old.keys() | new.keys()) if old.get(path) != new.get(path)]
        prior = a.get("verdict") if a else None
        current = b.get("verdict") if b else None
        if facts or a is None or b is None or prior != current or a["missing_information"] != b["missing_information"]:
            changes.append({"scope_id": scope, "check_id": check,
                "change": "added" if a is None else "removed" if b is None else "updated",
                "before_verdict": prior, "after_verdict": current, "changed_facts": facts,
                "missing_information": b["missing_information"] if b else []})
    return {"changes": changes, "remaining_questions": after.get("next_questions", []),
            "unresolved": [{"scope_id": f["scope_id"], "check_id": f["check_id"],
                           "missing_information": f["missing_information"]} for f in after["checks"]
                           if f["verdict"] == "insufficient_information" or f.get("conflict")],
            "capacity": after["capacity"], "warnings": after["warnings"],
            "notice": "Changes describe supplied facts, not proof that code or a deployment changed. Removed checks are not resolved."}
