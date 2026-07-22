#!/usr/bin/env python3
import json
from pathlib import Path
import torch

R=Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora")
files=sorted((R/"results/strict_offline_random_prefix_m8k3").glob("ex*_chunk*.pt"));device="cuda:0";ranks=[0,1,2,4,5]
acc={(s,k,r):dict(de=0.,ee=0.) for s in ["raw","preprocess"] for k in ["k","v"] for r in ranks};targets={}
for fi,f in enumerate(files):
 z=torch.load(f,map_location="cpu",weights_only=False);targets[f.stem]={}
 for source in z["delta"]:
  for kind in ["k","v"]:
   x=z["delta"][source][kind][:, :, 0].float() # [8,L,H,T,D]
   local={r:dict(de=0.,ee=0.) for r in ranks}
   for layer in range(x.shape[1]):
    q=x[:,layer].flatten(1).to(device);train,test=q[:6],q[6:];mean=train.mean(0);center=train-mean
    _,_,vh=torch.linalg.svd(center,full_matrices=False);truth=test
    for rank in ranks:
     pred=mean.expand_as(truth) if rank==0 else mean+(truth-mean)@vh[:rank].T@vh[:rank]
     de=float(truth.square().sum());ee=float((truth-pred).square().sum());acc[(source,kind,rank)]["de"]+=de;acc[(source,kind,rank)]["ee"]+=ee;local[rank]["de"]+=de;local[rank]["ee"]+=ee
   targets[f.stem][f"{source}_{kind}"]={str(r):1-local[r]["ee"]/local[r]["de"] for r in ranks}
rows=[]
for (s,k,r),z in acc.items():rows.append(dict(source=s,kind=k,rank=r,method="mean" if r==0 else f"oracle_r{r}",remaining_delta=(z["ee"]/z["de"])**.5,explained_delta_energy=1-z["ee"]/z["de"]))
out=R/"results/strict_offline_random_prefix_m8k3_rank_summary.json";out.write_text(json.dumps(dict(n_targets=len(files),split="first6 train / last2 heldout",rows=rows,targets=targets),indent=2)+"\n");print(json.dumps(rows,indent=2))
