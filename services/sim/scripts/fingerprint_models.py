import hashlib
import json
import sys
from pathlib import Path

MODELS = Path.home() / "models"

GRID = {
    "lapa": "lapa/lapa-v0.1.2-instruct-Q8_0.gguf",
    "gemma12": "gemma-3-12b-it/gemma-3-12b-it-Q8_0.gguf",
    "gemma27": "gemma-3-27b-it/gemma-3-27b-it-Q4_K_M.gguf",
    "mamay27": "mamay-3-27b-it/MamayLM-Gemma-3-27B-IT-v2.0-Q4_K_M.gguf",
    "bge": "bge-m3/bge-m3-q8_0.gguf",
}

CHUNK = 4 * 1024 * 1024


def fingerprint(path: Path) -> dict:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        digest.update(fh.read(CHUNK))
        if size > 2 * CHUNK:
            fh.seek(-CHUNK, 2)
            digest.update(fh.read(CHUNK))
    return {
        "file": str(path.relative_to(MODELS)),
        "bytes": size,
        "gib": round(size / 1024**3, 2),
        "quant": path.stem.split("-")[-1],
        "edge_sha256": digest.hexdigest()[:16],
    }


def main():
    out = {}
    for key, rel in GRID.items():
        path = MODELS / rel
        out[key] = fingerprint(path) if path.exists() else {"file": rel, "missing": True}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    width = max(len(k) for k in out)
    print()
    for key, info in out.items():
        if info.get("missing"):
            print(f"{key:<{width}}  ВІДСУТНЯ  {info['file']}")
        else:
            print(f"{key:<{width}}  {info['gib']:>6} GiB  {info['quant']:<8} {info['edge_sha256']}  {info['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
