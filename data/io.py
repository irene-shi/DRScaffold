"""Small, deterministic file helpers; generated artifacts are written atomically."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
                rows.append(row)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def resolve_file(value, root):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty file path")
    path = Path(value).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    return str(path)
