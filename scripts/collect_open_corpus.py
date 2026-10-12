"""Collect a bounded, pinned allowlist of openly licensed architecture texts.

No link following, executable dataset loaders, images, or site-wide scraping.
Only original document text is retained; labels are a separate authoring task.
"""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIMER_REVISION = "ae9bbd7b02d90b9866215de185217d33f39ab733"
AOSA_REVISION = "12308deff812eb3a4fa4e210ae5496e154961311"
REPOSITORIES = {
    "primer": {"repository": "donnemartin/system-design-primer", "revision": PRIMER_REVISION,
               "license": "CC-BY-4.0", "license_path": "LICENSE.txt", "license_marker": "Creative Commons Attribution 4.0",
               "license_url": "https://creativecommons.org/licenses/by/4.0/", "authors": "Donne Martin and contributors"},
    "aosa": {"repository": "aosabook/aosabook", "revision": AOSA_REVISION,
             "license": "CC-BY-3.0", "license_path": "LICENSE.md", "license_marker": "Attribution 3.0 Unported",
             "license_url": "https://creativecommons.org/licenses/by/3.0/", "authors": "Named chapter authors; edited by Amy Brown and Greg Wilson"},
}
CHAPTERS = [
    ("v1/hdfs", "The Hadoop Distributed File System", "Robert Chansler; Hairong Kuang; Sanjay Radia; Konstantin Shvachko; Suresh Srinivas", ["distributed_storage", "durability", "replication"]),
    ("v1/riak", "Riak and Erlang/OTP", "Francesco Cesarini; Andy Gross; Justin Sheehy", ["key_value", "replication", "consistency"]),
    ("v1/graphite", "Graphite", "Chris Davis", ["metrics", "time_series", "buffering"]),
    ("v1/nosql", "The NoSQL Ecosystem", "Adam Marcus", ["databases", "consistency", "tradeoffs"]),
    ("v2/distsys", "Scalable Web Architecture and Distributed Systems", "Kate Matsudaira", ["queues", "load_balancing", "caching"]),
    ("v2/nginx", "nginx", "Andrew Alexeev", ["web_server", "event_loop", "connections"]),
    ("v2/zeromq", "ZeroMQ", "Martin Sustrik", ["messaging", "backpressure", "queues"]),
    ("v2/twisted", "Twisted", "Jessica McKellar", ["event_loop", "networking", "asynchronous"]),
    ("v2/mediawiki", "MediaWiki", "Sumana Harihareswara; Guillaume Paumier", ["wiki", "caching", "web_application"]),
]


class ChapterText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "head", "nav"}:
            self.skip += 1
        if tag in {"p", "h1", "h2", "h3", "h4", "div", "li", "br", "tr", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "head", "nav"}:
            self.skip -= 1
        if tag in {"p", "h1", "h2", "h3", "h4", "div", "li", "tr", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self):
        return "\n".join(line for line in (re.sub(r"\s+", " ", line).strip() for line in "".join(self.parts).splitlines()) if line)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def download(url, cache):
    path = cache / sha(url.encode())
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "architecture-review-corpus/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        body = response.read(1_000_001)
    if len(body) > 1_000_000:
        raise ValueError("Source exceeds one-megabyte per-document limit")
    path.write_bytes(body)
    time.sleep(0.3)
    return body


def documents():
    yield {"id": "primer-concepts", "collection": "primer", "path": "README.md", "title": "System Design Primer: concepts",
           "source_kind": "educational_guide", "authors": REPOSITORIES["primer"]["authors"], "publication_year": 2017, "tags": ["system_design", "databases", "caching", "queues"]}
    for name in ["twitter", "mint", "pastebin", "query_cache", "sales_rank", "scaling_aws", "social_graph", "web_crawler"]:
        yield {"id": "primer-" + name, "collection": "primer", "path": f"solutions/system_design/{name}/README.md",
               "title": "Design exercise: " + name.replace("_", " "), "source_kind": "hypothetical_design_exercise",
               "authors": REPOSITORIES["primer"]["authors"], "publication_year": 2017, "tags": ["system_design", name]}
    for path, title, authors, tags in CHAPTERS:
        yield {"id": "aosa-" + path.split("/")[-1], "collection": "aosa", "path": "docs/en/" + path + ".html",
               "title": title, "source_kind": "historical_implementation_account" if path not in ["v1/nosql", "v2/distsys"] else "educational_guide",
               "authors": authors, "publication_year": 2011 if path.startswith("v1/") else 2012, "tags": tags}


def chunk_lines(text, target_chars=1800):
    # Non-overlapping chunks retain exact offsets into normalized source text.
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + target_chars // 2, end)
            if boundary >= 0:
                end = boundary + 1
            else:
                boundary = text.rfind(" ", start + target_chars // 2, end)
                if boundary >= 0:
                    end = boundary + 1
        yield start, end, text[start:end]
        start = end


def collect(output, cache):
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    (output / "licenses").mkdir(exist_ok=True)
    for key, repo in REPOSITORIES.items():
        url = f"https://raw.githubusercontent.com/{repo['repository']}/{repo['revision']}/{repo['license_path']}"
        body = download(url, cache)
        if repo["license_marker"] not in body.decode("utf-8"):
            raise ValueError(f"License verification failed for {key}")
        (output / "licenses" / (key + ".txt")).write_bytes(body)
    sources, chunks, seen = [], [], set()
    accessed_at = datetime.now(timezone.utc).date().isoformat()
    for document in documents():
        repo = REPOSITORIES[document["collection"]]
        url = f"https://raw.githubusercontent.com/{repo['repository']}/{repo['revision']}/{document['path']}"
        body = download(url, cache)
        text = body.decode("utf-8")
        if document["path"].endswith(".html"):
            parser = ChapterText()
            parser.feed(text)
            text = parser.text()
            transformation = "HTML markup, scripts, styles, and navigation removed; whitespace normalized; no paraphrasing"
        else:
            transformation = "Markdown retained; line endings normalized; no paraphrasing"
            text = text.replace("\r\n", "\n")
        source = {**document, "revision": repo["revision"], "url": url,
                  "source_url": f"https://github.com/{repo['repository']}/blob/{repo['revision']}/{document['path']}",
                  "license": repo["license"], "license_url": repo["license_url"], "accessed_at": accessed_at,
                  "raw_sha256": sha(body), "text_sha256": sha(text.encode()), "text_characters": len(text),
                  "transformation": transformation, "origin": "named_author_human_origin_reference",
                  "authorship_caveat": "Historical named authorship is documented; every later contributor edit has not been audited for AI use.",
                  "publication_date": None, "date_precision": "year_of_original_work", "label_status": "unlabeled_reference",
                  "intended_use": "retrieval_and_source_verification", "review_status": "source_checked_by_ai_not_independent_expert"}
        sources.append(source)
        for start, end, content in chunk_lines(text):
            digest = sha(content.encode())
            if digest in seen:
                continue
            seen.add(digest)
            chunks.append({"id": f"{document['id']}:{start}-{end}", "source_id": document["id"], "source_revision": repo["revision"],
                           "source_url": source["source_url"], "source_title": source["title"], "authors": source["authors"],
                           "license": repo["license"], "license_url": repo["license_url"], "text": content, "text_sha256": digest,
                           "start_character": start, "end_character": end, "tags": document["tags"],
                           "usage": "reference_only_not_gold_label", "scenario_split": None})
    (output / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    content = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in chunks)
    (output / "chunks.jsonl").write_text(content, encoding="utf-8", newline="\n")
    manifest = {"version": 1, "documents": len(sources), "chunks": len(chunks), "collections": list(REPOSITORIES),
                "expert_reviewed_decision_labels": 0, "date": accessed_at,
                "files": {str(path.relative_to(output)).replace("\\", "/"): sha(path.read_bytes())
                          for path in [output / "sources.json", output / "chunks.jsonl", *sorted((output / "licenses").glob("*.txt"))]}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "open-reference-v1")
    parser.add_argument("--cache", type=Path, default=ROOT / ".runtime" / "source-cache")
    args = parser.parse_args()
    print(json.dumps(collect(args.output, args.cache), indent=2))
