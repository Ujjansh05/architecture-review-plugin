"""One decision contract for model training, evaluation and live inference."""
import json

from .registry import REGISTRY

INPUT_VERSION = "decision-input-1"
CRITERIA = {
    "supported": "Explicit facts satisfy the required guarantee.",
    "contradicted": "Explicit facts establish a violation of the requirement.",
    "insufficient_information": "Necessary facts are missing, ambiguous or conflicting.",
}


def decision_input(check_id, facts, evidence, context=""):
    definition = REGISTRY[check_id]
    clean = definition.facts_type.model_validate(facts).model_dump(exclude_none=True)
    references = [{"id": r["id"], "conditions": r.get("conditions", []), "lesson": r["lesson"]}
                  for r in evidence]
    return {
        "state": json.dumps({"facts": clean, "evidence": references, "context": context}, ensure_ascii=False, sort_keys=True),
        "questions": {check_id: {"type": "choice", "instructions": definition.definition,
                                 "criteria": dict(CRITERIA)}},
    }
