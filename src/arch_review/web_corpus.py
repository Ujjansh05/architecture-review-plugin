"""Bounded public-document crawler; extracted pages are evidence, never instructions."""
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import deque
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

USER_AGENT = "ArchitectureReviewResearch/1.0"
TOPICS = {
    "retry_safety": r"\b(idempoten\w*|duplicat\w*|retr(?:y|ies)|redeliver\w*|atomic\w*|transactions?)\b",
    "ack_durability": r"\b(durab\w*|acknowledg\w*|persist\w*|replicat\w*|commits?|journals?|write.ahead)\b",
    "queue_overload": r"\b(queues?|queueing|queuing|back.?pressure|overload\w*|bounded|buffers?|rate.limit\w*)\b",
    "architecture_concepts": r"\b(cach\w*|shard\w*|consisten\w*|scalab\w*|partition\w*|concurren\w*|latency|throughput|failover|indexes|indexing)\b",
}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def normalize_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query:
        raise ValueError("Only public HTTPS document URLs without credentials/query strings are accepted")
    if parsed.port not in (None, 443) or ".." in urllib.parse.unquote(parsed.path).split("/"):
        raise ValueError("Unexpected port or traversal in source URL")
    return urllib.parse.urlunsplit(("https", parsed.netloc.lower(), parsed.path or "/", "", ""))


def in_scope(url, source):
    try:
        parsed = urllib.parse.urlsplit(normalize_url(url))
    except ValueError:
        return False
    return parsed.hostname == source["allowed_host"] and parsed.path.startswith(source["allowed_path_prefix"])


class Article(HTMLParser):
    """Keep article paragraphs, headings, code and bylines; discard page chrome."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.links, self.title, self.authors = [], [], [], []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        ignored = tag in {"script", "style", "head", "nav", "footer", "aside"} or bool(classes & {"banner", "navigation", "sidebar"})
        parent = self.stack[-1] if self.stack else ("", False, False, False)
        state = (tag, parent[1] or ignored, parent[2] or tag == "h1", parent[3] or "author" in classes)
        if tag == "a" and attrs.get("href") and not state[1]:
            self.links.append(attrs["href"])
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "li", "pre", "br", "tr"}:
            self.parts.append("\n")
        if tag not in {"br", "img", "hr", "meta", "link", "input", "wbr", "source"}:
            self.stack.append(state)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "li", "pre", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        state = self.stack[-1] if self.stack else ("", False, False, False)
        if not state[1]:
            self.parts.append(data)
            if state[2]:
                self.title.append(data)
            if state[3]:
                self.authors.append(data)

    def text(self):
        return "\n".join(line for line in (" ".join(value.split()) for value in "".join(self.parts).splitlines()) if line)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Avoid requesting a new origin or unapproved path before checking policy.
        raise ValueError("Redirect not followed automatically; add the final approved URL to the source config")


class Fetcher:
    def __init__(self, cache: Path, delay=1.0, max_bytes=1_500_000):
        self.cache, self.delay, self.max_bytes = cache, delay, max_bytes
        self.cache.mkdir(parents=True, exist_ok=True)
        self.robots, self.last_request = {}, {}
        self.opener = urllib.request.build_opener(NoRedirect())

    def _get(self, url):
        key = self.cache / sha(url.encode())
        if key.exists():
            data = key.read_bytes()
            if len(data) > self.max_bytes:
                raise ValueError("Cached document exceeds size limit")
            return data
        host = urllib.parse.urlsplit(url).netloc
        pause = self.delay - (time.monotonic() - self.last_request.get(host, 0))
        if pause > 0:
            time.sleep(pause)
        self.last_request[host] = time.monotonic()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,application/xhtml+xml"})
        with self.opener.open(request, timeout=30) as response:
            mime = response.headers.get_content_type()
            if mime not in {"text/html", "text/plain", "application/xhtml+xml"}:
                raise ValueError("Only text/HTML documents are collected")
            data = response.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise ValueError("Document exceeds download size limit")
        key.write_bytes(data)
        return data

    def fetch(self, url):
        url = normalize_url(url)
        parsed = urllib.parse.urlsplit(url)
        origin = parsed.scheme + "://" + parsed.netloc
        if origin not in self.robots:
            rules = urllib.robotparser.RobotFileParser(origin + "/robots.txt")
            try:
                rules.parse(self._get(origin + "/robots.txt").decode("utf-8").splitlines())
            except urllib.error.HTTPError as exc:
                if exc.code not in {404, 410}:
                    raise ValueError("robots.txt could not be verified") from exc
                rules.parse([])
            self.robots[origin] = rules
        rules = self.robots[origin]
        if not rules.can_fetch(USER_AGENT, url):
            raise ValueError("robots.txt disallows this document")
        self.delay = max(self.delay, rules.crawl_delay(USER_AGENT) or 0)
        rate = rules.request_rate(USER_AGENT)
        if rate and rate.requests:
            self.delay = max(self.delay, rate.seconds / rate.requests)
        return self._get(url)


def split_text(text, size=1800):
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary < 0:
                boundary = text.rfind(" ", start + size // 2, end)
            if boundary >= 0:
                end = boundary + 1
        yield start, end, text[start:end]
        start = end


def crawl(config, output, cache, max_pages=100, max_depth=2, delay=1.0):
    if not 1 <= max_pages <= 1000 or not 0 <= max_depth <= 5 or delay < 0.5:
        raise ValueError("Use 1–1000 pages, depth 0–5, and at least 0.5 seconds between requests")
    output.mkdir(parents=True, exist_ok=False)
    fetcher = Fetcher(cache, delay)
    documents, chunks, events, licenses = [], [], [], []
    visited, hashes = set(), {}
    page_count = 0
    for source in config["sources"]:
        if not source.get("license") or not source.get("license_marker") or not in_scope(source["license_url"], source):
            raise ValueError("Each source needs scoped, documented reuse permission")
        try:
            permission = fetcher.fetch(source["license_url"])
            license_text = " ".join(permission.decode("utf-8").split())
            if source["license_marker"] not in license_text:
                raise ValueError("The declared reuse-permission marker was not found")
        except (ValueError, OSError, UnicodeError) as exc:
            events.append({"source": source["id"], "status": "license_verification_failed", "error": str(exc)})
            continue
        licenses.append({"source": source["id"], "license": source["license"], "url": source["license_url"], "sha256": sha(permission)})
        frontier = deque((normalize_url(url), 0) for url in source["seeds"])
        while frontier and page_count < max_pages:
            url, depth = frontier.popleft()
            if url in visited or not in_scope(url, source):
                continue
            visited.add(url)
            page_count += 1
            try:
                raw = fetcher.fetch(url)
                body = raw.decode("utf-8-sig")
                is_html = "<html" in body[:1500].lower() or "<!doctype html" in body[:1500].lower()
                parser = Article()
                if is_html:
                    parser.feed(body)
                    text = parser.text()
                else:
                    text = body.replace("\r\n", "\n")
                if source.get("discover_links") and depth < max_depth:
                    for href in parser.links:
                        candidate = urllib.parse.urljoin(url, href)
                        if in_scope(candidate, source) and re.search(source["collect_path_regex"], urllib.parse.urlsplit(candidate).path):
                            frontier.append((normalize_url(candidate), depth + 1))
                if not re.search(source["collect_path_regex"], urllib.parse.urlsplit(url).path):
                    continue
                tags = [topic for topic, pattern in TOPICS.items() if re.search(pattern, text, re.I)]
                if len(text) < 500 or not tags:
                    events.append({"url": url, "status": "skipped_short_or_off_topic"})
                    continue
                digest = sha(" ".join(text.split()).encode())
                if digest in hashes:
                    events.append({"url": url, "status": "duplicate_document", "canonical_id": hashes[digest]})
                    continue
                identifier = "web-" + sha(url.encode())[:16]
                hashes[digest] = identifier
                heading = re.search(r"^#\s+(.+)$", text, re.M) if not is_html else None
                title = " ".join(parser.title).strip() or (heading[1] if heading else text.splitlines()[0].lstrip("# "))
                document = {"id": identifier, "url": url, "collection": source["id"], "title": title,
                            "authors": " ".join(" ".join(parser.authors).split()) or source["attribution"],
                            "license": source["license"], "license_url": source["license_url"], "publication_period": source.get("publication_period"),
                            "retrieved_at": datetime.now(timezone.utc).isoformat(), "raw_sha256": sha(raw), "text_sha256": sha(text.encode()),
                            "text": text, "tags": tags, "source_kind": "historical_reference_or_teaching_example",
                            "transformations": "HTML chrome removed, entities decoded, whitespace normalized; no paraphrasing" if is_html else "Markdown retained; line endings normalized"}
                documents.append(document)
                for start, end, content in split_text(text):
                    chunk_tags = [topic for topic, pattern in TOPICS.items() if re.search(pattern, content, re.I)]
                    if not chunk_tags or len(content.split()) < 40:
                        continue
                    chunks.append({"id": f"{identifier}:{start}-{end}", "source_id": identifier, "url": url, "title": document["title"],
                                   "authors": document["authors"], "license": source["license"], "license_url": source["license_url"],
                                   "text": content, "text_sha256": sha(content.encode()), "start": start, "end": end, "tags": chunk_tags})
            except (ValueError, OSError, UnicodeError) as exc:
                events.append({"url": url, "status": "fetch_or_parse_failed", "error": str(exc)})
        if frontier:
            events.append({"source": source["id"], "status": "page_limit_reached", "frontier_entries_remaining": len(frontier)})
    write_jsonl(output / "documents.jsonl", documents)
    write_jsonl(output / "chunks.jsonl", chunks)
    write_json(output / "source-config.json", config)
    write_json(output / "licenses.json", licenses)
    write_json(output / "crawl-report.json", events)
    manifest = {"version": "web-corpus-1", "documents": len(documents), "chunks": len(chunks), "pages_attempted": page_count,
                "max_pages": max_pages, "max_depth": max_depth, "crawl_scope": "configured_permitted_paths_only_not_the_entire_internet",
                "files": {path.name: sha(path.read_bytes()) for path in sorted(output.iterdir()) if path.is_file()}}
    write_json(output / "manifest.json", manifest)
    if not documents:
        raise ValueError("No documents collected; inspect crawl-report.json")
    return manifest
