import copy
import json
from pathlib import Path

import pytest

from arch_review.model import build_state
from arch_review.schema import ReviewRequest
from arch_review.web_corpus import Article, Fetcher, crawl, in_scope, normalize_url, sha, split_text
from arch_review.web_training import load_corpus, make_family, read_rows, select_jobs, validate_web_export


def test_html_extraction_keeps_article_and_removes_scripts():
    page = Article()
    page.feed('<html><head><title>SEO</title></head><body><h1>Queue design</h1>'
              '<p class="author">A. Author</p><nav>ignore</nav><div class="banner">advert</div>'
              '<p>Persist &amp; acknowledge.</p><pre>queue.push(message)</pre>'
              '<script>ignore_secret()</script><a href="next.html">Next</a></body></html>')
    assert "Persist & acknowledge." in page.text()
    assert "queue.push(message)" in page.text()
    assert "ignore" not in page.text() and "advert" not in page.text() and "SEO" not in page.text()
    assert page.title == ["Queue design"] and page.authors == ["A. Author"]
    assert page.links == ["next.html"]


@pytest.mark.parametrize("url", ["http://example.org/a", "https://user:pass@example.org/a",
                               "https://example.org/a?token=secret", "https://example.org/a/%2e%2e/b",
                               "https://example.org:8443/a"])
def test_reject_unsafe_source_urls(url):
    with pytest.raises(ValueError):
        normalize_url(url)


def test_scope_and_nonoverlapping_chunks():
    source = {"allowed_host": "example.org", "allowed_path_prefix": "/en/"}
    assert in_scope("https://example.org/en/chapter.html#part", source)
    assert not in_scope("https://example.org.evil.test/en/chapter.html", source)
    assert not in_scope("https://example.org/private/chapter.html", source)
    text = ("Long architecture paragraph. " * 80 + "\n") * 4
    chunks = list(split_text(text))
    assert "".join(chunk[2] for chunk in chunks) == text
    assert all(text[start:end] == part and len(part) <= 1800 for start, end, part in chunks)


def test_robots_block_before_document_request(tmp_path, monkeypatch):
    fetcher = Fetcher(tmp_path)
    requested = []

    def get(url):
        requested.append(url)
        return b"User-agent: *\nDisallow: /private/\n"

    monkeypatch.setattr(fetcher, "_get", get)
    with pytest.raises(ValueError, match="disallows"):
        fetcher.fetch("https://example.org/private/data.html")
    assert requested == ["https://example.org/robots.txt"]


def test_crawl_deduplicates_and_does_not_leave_scope(tmp_path, monkeypatch):
    root = "https://example.org/en/"
    body = ("Persist the queue before acknowledgement. " * 30).encode()
    pages = {root: b'<html><h1>License marker</h1><a href="v1/a.html">A</a>'
                   b'<a href="v1/copy.html">Copy</a><a href="https://evil.test/en/v1/x.html">X</a></html>',
             root + "v1/a.html": body, root + "v1/copy.html": body}
    requested = []

    def fetch(self, url):
        requested.append(url)
        return pages[url]

    monkeypatch.setattr(Fetcher, "fetch", fetch)
    source = {"id": "demo", "allowed_host": "example.org", "allowed_path_prefix": "/en/",
              "license": "CC-BY-4.0", "license_url": root, "license_marker": "License marker",
              "attribution": "Author", "seeds": [root], "discover_links": True,
              "collect_path_regex": r"/v1/.*\.html$"}
    result = crawl({"sources": [source]}, tmp_path / "out", tmp_path / "cache")
    assert result["documents"] == 1
    assert all(url.startswith(root) for url in requested)
    audit = json.loads((tmp_path / "out" / "crawl-report.json").read_text())
    assert audit[0]["status"] == "duplicate_document"


def fixture_job(check="ack_durability"):
    quote = "Durable writes must be completed before successful acknowledgement of the operation."
    chunk = {"id": "doc:0-100", "source_id": "doc", "text": quote, "url": "https://example.org/doc",
             "title": "Persistence", "authors": "Author", "license": "CC-BY-4.0", "license_url": "https://example.org/license"}
    return {"source_id": "doc", "check_id": check, "chunk_id": chunk["id"], "chunk": chunk,
            "split": "train", "source_group": "doc"}


def fixture_response():
    request = {"requirements": {"acknowledged_operation": "message write", "survive_failures": ["process_crash"]},
               "design": {"components": ["writer", "log"], "flows": ["Writer submits a message"],
                          "acknowledgement": {"acknowledged_operation": "message write", "after_durable_commit": True,
                                              "survives_failures": ["process_crash"]}}}
    return {"source_id": "doc", "check_id": "ack_durability", "relevant": True,
            "supporting_quote": fixture_job()["chunk"]["text"], "lesson": "Commit before reporting success.",
            "condition": "When acknowledged writes must survive a process crash.",
            "request_json": json.dumps(request), "explanation": "Hypothetical write flow inspired by the persistence lesson."}


def test_web_only_minimal_pairs_keep_labels_out_of_state():
    cases, exported = make_family(fixture_job(), fixture_response())
    assert {case["expected"] for case in cases} == {"supported", "contradicted", "insufficient_information"}
    original = cases[0]["request"]
    for case, row in zip(cases, exported):
        modified = copy.deepcopy(case["request"])
        modified["design"]["acknowledgement"]["after_durable_commit"] = True
        assert modified == original
        state = json.loads(row["state"])
        assert state["evidence"] == [case["evidence"]]
        assert "expected" not in state and case["expected"] not in row["state"]
        assert case["expert_reviewed"] is False


def test_fabricated_quote_and_wrong_label_are_rejected():
    response = fixture_response()
    response["supporting_quote"] = "An invented passage that the original document does not actually contain."
    with pytest.raises(ValueError, match="exact"):
        make_family(fixture_job(), response)
    response = fixture_response()
    request = json.loads(response["request_json"])
    request["design"]["acknowledgement"]["survives_failures"] = []
    response["request_json"] = json.dumps(request)
    with pytest.raises(ValueError, match="consistency"):
        make_family(fixture_job(), response)


def test_duplicate_passages_are_grouped_before_split():
    documents = {f"doc-{i}": {} for i in range(12)}
    chunks = [{"id": key + ":0", "source_id": key, "text": f"{key} durable replication acknowledgement commit"}
              for key in documents]
    chunks[1]["text"] = chunks[0]["text"]
    jobs = select_jobs(documents, chunks, max_documents=0)
    indexed = {job["source_id"]: job for job in jobs}
    assert indexed["doc-0"]["source_group"] == indexed["doc-1"]["source_group"]
    assert indexed["doc-0"]["split"] == indexed["doc-1"]["split"]
    assert {job["split"] for job in jobs} == {"train", "development", "calibration", "test"}
    assert jobs == select_jobs(documents, chunks, max_documents=0)


def test_web_runner_rejects_non_web_export(tmp_path, monkeypatch):
    monkeypatch.setattr("arch_review.web_training.validate_export", lambda *a, **kw: ({"version": "pilot"}, {}))
    with pytest.raises(ValueError, match="new web-only"):
        validate_web_export(tmp_path)


def test_scraper_config_is_pinned_and_limited_to_permitted_sources():
    config = json.loads((Path(__file__).parents[1] / "configs" / "web_sources.json").read_text())
    assert {source["license"] for source in config["sources"]} == {"CC-BY-3.0", "CC-BY-4.0"}
    for source in config["sources"]:
        assert all(in_scope(url, source) for url in source["seeds"])
        assert in_scope(source["license_url"], source)
        assert sha(source["id"].encode())


def test_published_web_snapshot_matches_sources_and_training_inputs():
    root = Path(__file__).parents[1] / "datasets" / "web-solo-v1"
    _, docs, _ = load_corpus(root / "corpus")
    manifest, rows = validate_web_export(root / "export")
    assert sha((root / "corpus" / "manifest.json").read_bytes()) == manifest["corpus_manifest_sha256"]
    cases = {case["id"]: case for case in read_rows(root / "export" / "cases.jsonl")}
    assert len(cases) == sum(len(split) for split in rows.values())
    for split, values in rows.items():
        for row in values:
            case = cases[row["id"]]
            assert case["split"] == split
            assert case["supporting_quote"] in docs[case["source"]["source_id"]]["text"]
            assert row["expected"] == {case["check_id"]: case["expected"]}
            req = ReviewRequest.model_validate(case["request"])
            assert row["state"] == build_state(req, case["check_id"], [case["evidence"]])
