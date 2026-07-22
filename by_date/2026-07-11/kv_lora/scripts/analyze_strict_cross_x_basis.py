#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import torch
from transformers import AutoConfig

p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--kind",required=True); p.add_argument("--device",default="cuda:0"); a=p.parse_args()
R=Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora")
ranks=[1,2,4,8,16,32]
cfg=AutoConfig.from_pretrained("/mnt/qjhs-sh-lab-01/models/Qwen3-32B",trust_remote_code=True)
inv=1/(float(cfg.rope_theta)**(torch.arange(0,128,2).float()/128))
def align(x,p):
 if a.kind!="k" or not p:return x
 q=-p*inv;c=torch.cat([q.cos(),q.cos()]).view(1,1,128);s=torch.cat([q.sin(),q.sin()]).view(1,1,128)
 return x*c+torch.cat([-x[...,64:],x[...,:64]],-1)*s
train=[];test=[]
for target in range(1,6):
 D=R/f"results/perdoc_context_deltas/strict_t{target}"
 ps=[torch.load(x,map_location="cpu") for x in sorted(D.glob(f"context_*_{a.source}.pt"))]
 vec=[]
 for z in ps:
  x=z[f"delta_{a.kind}"][:,0].float()
  x=align(x,z["meta"]["prefix_tokens"]).mean(2).flatten()
  vec.append(x)
 train.append(torch.stack(vec[:40]));test.append(torch.stack(vec[40:]))
train=torch.stack(train).to(a.device);test=torch.stack(test).to(a.device)
means=train.mean(1,keepdim=True); tr=train-means; te=test-means
global_tr=tr.flatten(0,1);_,sg,bg=torch.linalg.svd(global_tr,full_matrices=False)
rows=[]
for xi in range(5):
 _,sp,bp=torch.linalg.svd(tr[xi],full_matrices=False)
 for rank in [0]+ranks:
  for mode,basis in [("shared",bg),("per_x",bp)]:
   pred=means[xi,0].expand_as(test[xi]) if rank==0 else means[xi,0]+(te[xi]@basis[:rank].T)@basis[:rank]
   err=test[xi]-pred;delta=test[xi]
   rows.append(dict(target=xi+1,source=a.source,kind=a.kind,mode=mode,rank=rank,
    remaining_delta=float((err.square().sum()/delta.square().sum()).sqrt()),
    explained_delta_energy=float(1-err.square().sum()/delta.square().sum())))
out=R/"results"/f"strict_cross_x_{a.source}_{a.kind}.csv"
with out.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
