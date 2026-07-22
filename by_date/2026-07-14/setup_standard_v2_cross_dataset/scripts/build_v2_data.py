#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
EXP = ROOT / "MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset"
SRC_ROOT = Path("/mnt/qjhs-sh-lab-01/wjh/FusionRAG/data")

DATASETS = {
    "musique-v2": "musique-200.jsonl",
    "2wikimqa-v2": "2wikimqa-200.jsonl",
    "hotpotqa-v2": "hotpotqa-260-100-10-doc.jsonl",
    "triviaqa-v2": "triviaqa-270-100-10-doc.jsonl",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_logical_rows(path: Path) -> int:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if len(rows) == 1 and isinstance(rows[0], list):
        return len(rows[0])
    return len(rows)


def main() -> None:
    out_dir = EXP / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, src_name in DATASETS.items():
        src = SRC_ROOT / src_name
        dst = out_dir / f"{name}.jsonl"
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        manifest.append({
            "dataset": name,
            "source": str(src),
            "target": str(dst),
            "physical_jsonl_rows": count_jsonl(dst),
            "logical_examples": count_logical_rows(dst),
            "sha256": sha256(dst),
        })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

