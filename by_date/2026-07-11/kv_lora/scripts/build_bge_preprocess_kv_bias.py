#!/usr/bin/env python3
from pathlib import Path
import torch

ROOT=Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora")
C=Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique");raw=C/"kv_cache";pp=C/"preprocess_kv_cache_global_topk10_bge";out=ROOT/"results/static_bias_bge_preprocess_minus_raw_kv_50";out.mkdir(parents=True,exist_ok=True)
counts={"k":0,"v":0}
for kind in ["key","value"]:
 for f in sorted(raw.glob(f"*_{kind}.pt")):
  ex=int(f.name.split("_",1)[0]);
  if ex>=50:continue
  q=pp/f.name
  if not q.exists():continue
  r=torch.load(f,map_location="cpu",weights_only=True).float();p=torch.load(q,map_location="cpu",weights_only=True).float()
  torch.save((p-r).half(),out/f.name.replace(f"_{kind}.pt",f"_{'key' if kind=='key' else 'value'}_bias.pt"));counts["k" if kind=="key" else "v"]+=1
print(counts)
