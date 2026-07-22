#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument("--part",required=True);p.add_argument("--source",required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args()
R=Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora")
D=R/"results/perdoc_context_deltas"/a.part;m=json.loads((D/"manifest.json").read_text())
assert set(m["train_pool"]).isdisjoint(m["test_pool"])
all_payload=[torch.load(x,map_location="cpu") for x in sorted(D.glob(f"context_*_{a.source}.pt"))]
P=[x for x in all_payload if x["meta"]["split"]=="train"][:40]
P += [x for x in all_payload if x["meta"]["split"]=="test" and x["meta"]["prefix_ids"]][:10]
assert len(P)==50
assert all(x["meta"]["split"]=="train" for x in P[:40]) and all(x["meta"]["split"]=="test" for x in P[40:])
CR=Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique")/("kv_cache" if a.source=="raw" else "preprocess_kv_cache_global_topk10_bge")
chunk={};lens={}
for i in range(1,11):
 x=torch.load(CR/f"0_{i}_value.pt",map_location="cpu").float();chunk[i]=x[:,0].mean(2).flatten(1);lens[i]=x.shape[3]

# Per layer: [overall prefix mean, last-document mean], plus explicit position features.
F=[];Pos=[]
for z in P:
 ids=z["meta"]["prefix_ids"]
 if ids:
  ll=torch.tensor([lens[i] for i in ids]).view(-1,1,1)
  overall=(torch.stack([chunk[i] for i in ids])*ll).sum(0)/ll.sum()
  last=chunk[ids[-1]]
 else: overall=torch.zeros(64,1024);last=torch.zeros(64,1024)
 F.append(torch.cat([overall,last],1));Pos.append([z["meta"]["prefix_tokens"],len(ids)])
F=torch.stack(F);Pos=torch.tensor(Pos,dtype=torch.float32)

def standard(train,test):
 mu=train.mean(0);sd=train.std(0).clamp_min(1e-5);return (train-mu)/sd,(test-mu)/sd
def feature_pca(train,test,r=8):
 tr,te=standard(train,test);_,_,vh=torch.linalg.svd(tr,full_matrices=False);return tr@vh[:r].T,te@vh[:r].T
def ridge(x,y,xt):
 def fp(a0,b0,z0,alpha):
  aa,zz=standard(a0,z0);ym=b0.mean(0);bb=b0-ym;k=aa@aa.T;scale=float(torch.trace(k)/len(k))
  c=torch.linalg.solve(k+alpha*max(scale,1e-8)*torch.eye(len(k),device=k.device),bb);return zz@aa.T@c+ym
 best=None
 for alpha in [1e-3,1e-2,.1,1.,10.,100.]:
  pr=fp(x[:32],y[:32],x[32:40],alpha);loss=float((pr-y[32:40]).square().mean())
  if best is None or loss<best[0]:best=(loss,alpha)
 return fp(x,y,xt,best[1]),best[1]

methods=["mean","position","prefix_v","prefix_v_position","oracle_r8"]
acc={(i,q):dict(de=0.,be=0.,ee=0.) for i in range(40,50) for q in methods};alphas=[]
for layer in range(64):
 delta=torch.stack([z["delta_v"][layer,0].flatten().float() for z in P]).to(a.device)
 mean=delta[:40].mean(0);center=delta[:40]-mean;_,_,basis=torch.linalg.svd(center,full_matrices=False);basis=basis[:8]
 coef=center@basis.T
 ftr,fte=feature_pca(F[:40,layer].to(a.device),F[40:,layer].to(a.device),8)
 ptr,pte=standard(Pos[:40].to(a.device),Pos[40:].to(a.device))
 cp,ca=ridge(ftr,coef,fte);pp,pa=ridge(ptr,coef,pte);jp,ja=ridge(torch.cat([ftr,ptr],1),coef,torch.cat([fte,pte],1))
 alphas.append(dict(layer=layer,prefix=ca,position=pa,joint=ja))
 for j,i in enumerate(range(40,50)):
  truth=delta[i];preds={"mean":mean,"position":mean+pp[j]@basis,"prefix_v":mean+cp[j]@basis,"prefix_v_position":mean+jp[j]@basis,"oracle_r8":mean+((truth-mean)@basis.T)@basis}
  for q,pred in preds.items():
   z=acc[(i,q)];z["de"]+=float(truth.square().sum());z["be"]+=float(P[i]["base_v_norm2_layer"][layer]);z["ee"]+=float((truth-pred).square().sum())
rows=[]
for (i,q),z in acc.items():rows.append(dict(part=a.part,target=m["target"],source=a.source,context_index=i,method=q,original_gap=(z["de"]/z["be"])**.5,final_kv_error=(z["ee"]/z["be"])**.5,remaining_delta=(z["ee"]/z["de"])**.5,explained_delta_energy=1-z["ee"]/z["de"]))
out=R/"results"/f"strict_predict_{a.part}_{a.source}.csv"
with out.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(R/"results"/f"strict_predict_{a.part}_{a.source}_meta.json").write_text(json.dumps(dict(split="document-disjoint 40/10",basis_rank=8,feature_pca_rank=8,alphas=alphas),indent=2)+"\n")
print(out)
