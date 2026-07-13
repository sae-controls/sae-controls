"""Atomic JSON + NPZ writers — write to a temp file, then `os.replace` so
partial writes don't corrupt the artifact directory."""
from __future__ import annotations

import json
import os
from pathlib import Path


def save_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def save_npz_atomic(path: Path, **arrays) -> None:
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
