"""Collect system-design concepts/examples from configured, permitted web sources."""
import argparse
import json
from pathlib import Path

from arch_review.web_corpus import crawl

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/web_sources.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/web-corpus")
    parser.add_argument("--cache", type=Path, default=ROOT / ".runtime/web-cache")
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    result = crawl(json.loads(args.config.read_text(encoding="utf-8")), args.output, args.cache,
                   args.max_pages, args.max_depth, args.delay)
    print(json.dumps(result, indent=2))
