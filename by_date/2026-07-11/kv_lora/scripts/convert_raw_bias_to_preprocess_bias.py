#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument("--raw-bias-dir",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--raw-cache",required=True);p.add_argument("--preprocess-cache",required=True);a=p.parse_args()
src=Path(a.raw_bias_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);raw=Path(a.raw_cache);pp=Path(a.preprocess_cache)
count=0
for f in sorted(src.glob("*_value_bias.pt")):
 stem=f.name.removesuffix("_value_bias.pt");ex,chunk=stem.split("_",1)
 r=raw/f"{ex}_{chunk}_value.pt";q=pp/f"{ex}_{chunk}_value.pt"
 if not q.exists():continue
 b=torch.load(f,map_location="cpu",weights_only=True).float();rv=torch.load(r,map_location="cpu",weights_only=True).float();pv=torch.load(q,map_location="cpu",weights_only=True).float()
 torch.save((b+rv-pv).half(),out/f.name);count+=1
print(count)
