#!/usr/bin/env python3
from pathlib import Path
import torch

R=Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora")
src=R/"results/strict_offline_random_prefix_m8k3"
for source in ["raw","preprocess"]:
 out=R/"results"/f"strict_offline_mean_kv_{source}_ex0to4_m6";out.mkdir(parents=True,exist_ok=True);count=0
 for f in sorted(src.glob("ex*_chunk*.pt")):
  z=torch.load(f,map_location="cpu",weights_only=False)
  if source not in z["delta"]:continue
  ex=z["example"];chunk=z["chunk"]
  for kind in ["k","v"]:
   bias=z["delta"][source][kind][:6].float().mean(0).half()
   torch.save(bias,out/f"{ex}_{chunk}_{kind}ey_bias.pt" if kind=="k" else out/f"{ex}_{chunk}_value_bias.pt")
  count+=1
 print(source,count,out)
