"""Prepare a hash-verified new public-domain corpus without redistribution."""
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen
import numpy as np

ROOT = Path(__file__).resolve().parent
URL = "https://www.gutenberg.org/cache/epub/84/pg84.txt"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def transform(raw):
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    start = re.findall(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK [^\n]+\*\*\*", text)
    end = re.findall(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK [^\n]+\*\*\*", text)
    assert len(start) == len(end) == 1
    assert "FRANKENSTEIN" in start[0].upper()
    return (text.split(start[0])[1].split(end[0])[0].strip()+"\n").encode("utf-8")


def read_data():
    manifest = json.loads((ROOT/"data/manifest.json").read_text())
    raw = (ROOT/"data/source.txt").read_bytes()
    body = transform(raw)
    assert digest(raw) == manifest["source_sha256"] and digest(body) == manifest["body_sha256"]
    assert body == (ROOT/"data/body.bin").read_bytes()
    values = np.frombuffer(body, np.uint8).astype(np.int32)
    edges = manifest["split_offsets"]
    return manifest, tuple(values[a:b] for a, b in zip(edges[:-1], edges[1:]))


if __name__ == "__main__":
    root = ROOT/"data"
    root.mkdir(exist_ok=True)
    with urlopen(URL, timeout=30) as response:
        raw = response.read(1000001)
        modified = response.headers.get("Last-Modified")
    assert len(raw) < 1000000
    body = transform(raw)
    manifest = {"dataset": "Frankenstein; or, the Modern Prometheus", "author": "Mary Wollstonecraft Shelley",
                "source": "https://www.gutenberg.org/ebooks/84", "download_url": URL,
                "source_sha256": digest(raw), "body_sha256": digest(body), "source_bytes": len(raw), "body_bytes": len(body),
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(), "last_modified": modified,
                "split_offsets": [0, int(.8*len(body)), int(.9*len(body)), len(body)],
                "transformation": "UTF-8 BOM decode, CRLF to LF, keep between single Gutenberg start/end markers, strip outer whitespace, append LF, encode UTF-8.",
                "license": "Project Gutenberg identifies this work as public domain in the USA. Full corpus is referenced, not redistributed.",
                "tokenization": "UTF-8 byte IDs 0..255; contiguous 80/10/10 split; no window crosses a split; recurring phrases retained."}
    if (root/"manifest.json").exists():
        old = json.loads((root/"manifest.json").read_text())
        assert old["source_sha256"] == digest(raw) and old["body_sha256"] == digest(body)
    else:
        (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (root/"source.txt").write_bytes(raw)
    (root/"body.bin").write_bytes(body)
    read_data()
    print(json.dumps(manifest))
