"""Mechanical source matching, not natural-language inference or deployment verification.

A machine proposal is accepted only when its cited range contains a complete JSON
declaration naming the same check and scope and the exact typed fact value. The
whole source must also be JSON; a JSON-looking fragment embedded in prose is not a
declaration. Sources may contain a declaration, a list, or {"checks": [...]}.
"""
import json
import math


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON constant")


def _load(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_reject_constant)


def _records(value):
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict) and isinstance(value.get("checks"), list):
        return [v for v in value["checks"] if isinstance(v, dict)]
    return [value] if isinstance(value, dict) else []


def _value(record, check_id, scope_id, path):
    if record.get("check_id") != check_id or record.get("scope_id") != scope_id:
        return False, None
    if record.get("path") == path and "value" in record:
        return True, record["value"]
    value = record.get("facts")
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def same_value(left, right):
    # Python considers True == 1; typed JSON declarations must not.
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_value(left[k], right[k]) for k in left)
    return left == right


def matches_type(value, schema):
    """Reject JSON type coercions before the legacy nested Pydantic schemas."""
    if not schema:
        return False
    if "anyOf" in schema:
        return any(matches_type(value, option) for option in schema["anyOf"])
    if "enum" in schema and not any(same_value(value, option) for option in schema["enum"]):
        return False
    kind = schema.get("type")
    if kind == "boolean":
        return type(value) is bool
    if kind == "integer":
        return type(value) is int
    if kind == "number":
        return type(value) in (int, float) and math.isfinite(value)
    if kind == "string":
        return isinstance(value, str)
    if kind == "null":
        return value is None
    if kind == "array":
        return isinstance(value, list) and all(matches_type(item, schema.get("items")) for item in value)
    return False


def parse_sources(documents):
    parsed = {}
    for document_id, text in documents.items():
        try:
            parsed[document_id] = _records(_load(text))
        except (ValueError, TypeError, RecursionError):
            parsed[document_id] = []
    return parsed


def verify_fact(check_id, scope_id, path, value, references, documents, parsed_sources=None):
    parsed_sources = parse_sources(documents) if parsed_sources is None else parsed_sources
    matched = False
    for reference in references:
        text = documents.get(reference.document_id, "")
        try:
            source = parsed_sources[reference.document_id]
            cited = _records(_load(text[reference.start:reference.end]))
        except (ValueError, TypeError, KeyError, RecursionError):
            continue
        declarations = [v for record in source
                        for found, v in [_value(record, check_id, scope_id, path)] if found]
        if any(not same_value(v, value) for v in declarations):
            return "conflicting", "The typed source declaration disagrees with the proposed value."
        for record in cited:
            found, declared = _value(record, check_id, scope_id, path)
            if (found and same_value(declared, value)
                    and any(same_value(record, original) for original in source)):
                matched = True
    if matched:
        for records in parsed_sources.values():
            for record in records:
                found, declared = _value(record, check_id, scope_id, path)
                if found and not same_value(declared, value):
                    return "conflicting", "Typed source declarations disagree for this check and scope."
        return "source_matched", "Exact typed JSON declaration matched; deployment behavior is not verified."
    return "unverified", "A quote proves where text came from, not that the proposed fact follows from it."
