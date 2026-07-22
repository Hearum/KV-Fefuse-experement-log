#!/usr/bin/env python3
import json
from pathlib import Path
import torch

ROOT=Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/structured_kv_adapter")
DATA=ROOT/"results/qwen3_8b/wikitext_delta_smoke";OUT=ROOT/"results/qwen3_8b/wiki_basis_50train_16test.json"
L,H,F=36,8,128;RANKS=(8,16,32,64,128)
files=sorted(DATA.glob("sample*.pt"));train=[];test=[]
for p in files:
 z=torch.load(p,map_location="cpu",weights_only=False);(train if z["sample"]<50 else test).append(z)
models={};rows=[]
for ki,kind in enumerate(("k","v")):
 ys=[z["delta_kv"][...,ki*F:(ki+1)*F].double() for z in train];y=torch.cat(ys,2)
 mean=y.mean(2);center=y-mean.unsqueeze(2);cov=torch.einsum("lhti,lhtj->lhij",center,center);_,vec=torch.linalg.eigh(cov);vec=vec.flip(-1);models[kind]=(mean,vec)
 delta_sq={r:0. for r in RANKS};error_sq={r:0. for r in RANKS};dot={r:0. for r in RANKS};pred_sq={r:0. for r in RANKS}
 for z in test:
  target=z["delta_kv"][...,ki*F:(ki+1)*F].double();center=target-mean.unsqueeze(2)
  for rank in RANKS:
   b=vec[...,:rank];pred=mean.unsqueeze(2)+torch.einsum("lhtf,lhfr,lhgr->lhtg",center,b,b)
   delta_sq[rank]+=float(target.square().sum());error_sq[rank]+=float((target-pred).square().sum());dot[rank]+=float((target*pred).sum());pred_sq[rank]+=float(pred.square().sum())
 for rank in RANKS:
  remaining=(error_sq[rank]/delta_sq[rank])**.5;rows.append({"kind":kind,"rank":rank,"remaining_delta":remaining,"explained_delta_energy":1-remaining**2,"cosine":dot[rank]/(delta_sq[rank]*pred_sq[rank])**.5})
result={"model":"Qwen3-8B","train_samples":[z["sample"] for z in train],"test_samples":[z["sample"] for z in test],"tokens_per_sample":8,"basis":"per-layer/head feature PCA, test oracle projection","mean_original_gap":{"k":sum(z["metrics"]["k_original_gap"] for z in test)/len(test),"v":sum(z["metrics"]["v_original_gap"] for z in test)/len(test)},"rows":rows}
OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
