#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument("--raw-cache",required=True);p.add_argument("--preprocess-cache",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--max-examples",type=int,default=50);a=p.parse_args()
raw=Path(a.raw_cache);pp=Path(a.preprocess_cache);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
count=0;missing=[]
for f in sorted(raw.glob("*_value.pt")):
 ex=int(f.name.split("_",1)[0]);
 if ex>=a.max_examples:continue
 q=pp/f.name
 if not q.exists():missing.append(f.name);continue
 rv=torch.load(f,map_location="cpu",weights_only=True).float();pv=torch.load(q,map_location="cpu",weights_only=True).float()
 if rv.shape!=pv.shape:raise ValueError(f"shape mismatch {f.name}: {rv.shape} vs {pv.shape}")
 torch.save((pv-rv).half(),out/f.name.replace("_value.pt","_value_bias.pt"));count+=1
print({"written":count,"missing":len(missing),"missing_examples":missing[:20]})
