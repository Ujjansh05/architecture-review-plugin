import importlib.util
from pathlib import Path


def test_blind_baseline_removes_label_bearing_ids():
    path = Path(__file__).parents[1] / "scripts" / "run_codex_baseline.py"
    spec = importlib.util.spec_from_file_location("baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = [{"id": "family-contradicted", "split": "development", "state": "same evidence and architecture",
                 "questions": {"cache_freshness": {"type": "choice"}}}]
    rows, mapping = module.blind_inputs(original)
    assert rows[0]["id"] == "case-0000" and "contradicted" not in rows[0]["id"]
    assert rows[0]["state"] == original[0]["state"]
    assert mapping["case-0000"]["id"] == original[0]["id"]
