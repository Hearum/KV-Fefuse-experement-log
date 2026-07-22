#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-frac", type=float, required=True)
    p.add_argument("--end-frac", type=float, required=True)
    p.add_argument("--model-path", default="/home/hming/models/Qwen3-8B")
    a = p.parse_args()
    raw = Path(a.text_file).read_text(encoding="utf-8", errors="ignore"); n = len(raw)
    start, end = int(n * a.start_frac), int(n * a.end_frac)
    if start:
        newline = raw.find("\n", start); start = n if newline < 0 else newline + 1
    if end < n:
        newline = raw.rfind("\n", 0, end); end = 0 if newline < 0 else newline
    text = raw[start:end]
    text = "\n".join(line for line in text.splitlines() if line.strip() and not line.strip().startswith("="))
    tokenizer = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.int32)
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True); torch.save(ids, output)
    meta = {"model":Path(a.model_path).name,"tokenizer_path":a.model_path,
            "text_file":a.text_file,"start_frac":a.start_frac,"end_frac":a.end_frac,
            "char_start":start,"char_end":end,"tokens":len(ids),"dtype":"int32"}
    output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))


if __name__ == "__main__": main()
