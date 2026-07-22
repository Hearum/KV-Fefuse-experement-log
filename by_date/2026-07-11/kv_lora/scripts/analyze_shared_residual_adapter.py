#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch

argp=argparse.ArgumentParser();argp.add_argument("--train-cutoff",type=int,default=30);argp.add_argument("--output-tag",default="shared_preprocess_residual_adapter");args=argp.parse_args()
torch.set_num_threads(4);D="cuda:0";R=Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora");DATA=R/"results/formal_preprocess_residual_50";C=Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique/preprocess_kv_cache_global_topk10_bge")
L,H,F=64,8,128;A=F+1;stats={}
for kind in ["k","v"]:stats[kind]=dict(xtx=torch.zeros(L,H,A,A,device=D),xty=torch.zeros(L,H,A,F,device=D),ysum=torch.zeros(L,H,F,device=D),yty=torch.zeros(L,H,F,F,device=D),n=0)
files=sorted(DATA.glob("ex*_sub*.pt"))
for f in files:
 z=torch.load(f,map_location="cpu",weights_only=False)
 if z["example"]>=args.train_cutoff:continue
 for item in z["items"]:
  cid=item["chunk"];ex=z["example"]
  for kind,suffix in [("k","key"),("v","value")]:
   x=torch.load(C/f"{ex}_{cid}_{suffix}.pt",map_location="cpu",weights_only=True)[:,0].float().to(D);y=item[f"delta_{kind}"][:,0].float().to(D)
   # Uniform token cap avoids long documents dominating both fitting and compute.
   if x.shape[2]>32:idx=torch.linspace(0,x.shape[2]-1,32,device=D).long();x=x[:,:,idx];y=y[:,:,idx]
   ones=torch.ones(L,H,x.shape[2],1,device=D);xa=torch.cat([x,ones],-1);s=stats[kind]
   s["xtx"]+=torch.einsum("lhti,lhtj->lhij",xa,xa);s["xty"]+=torch.einsum("lhti,lhtj->lhij",xa,y);s["ysum"]+=y.sum(2);s["yty"]+=torch.einsum("lhti,lhtj->lhij",y,y);s["n"]+=x.shape[2]
models={}
for kind,s in stats.items():
 eye=torch.eye(A,device=D).view(1,1,A,A);scale=s["xtx"].diagonal(dim1=-2,dim2=-1).mean(-1,keepdim=True).unsqueeze(-1);w=torch.linalg.solve(s["xtx"]+1e-3*scale*eye,s["xty"])
 mean=s["ysum"]/s["n"];cov=s["yty"]-s["n"]*torch.einsum("lhi,lhj->lhij",mean,mean);_,basis=torch.linalg.eigh(cov);basis=basis[:,:,:,-8:]
 linear=w[:,:,:F];bias=w[:,:,F];u,sv,vh=torch.linalg.svd(linear);reduced={f"w{rank}":(u[:,:,:,:rank]*sv[:,:,:rank].unsqueeze(-2))@vh[:,:,:rank] for rank in [8,16,32,64]}
 models[kind]=dict(w=w,bias=bias,mean=mean,basis=basis,**reduced)
acc={(k,m):[0.,0.] for k in ["k","v"] for m in ["zero","mean","oracle_basis8","linear","linear_rank8"]}
test_examples=set()
for f in files:
 z=torch.load(f,map_location="cpu",weights_only=False)
 if z["example"]<args.train_cutoff:continue
 test_examples.add(z["example"])
 for item in z["items"]:
  cid=item["chunk"];ex=z["example"]
  for kind,suffix in [("k","key"),("v","value")]:
   x=torch.load(C/f"{ex}_{cid}_{suffix}.pt",map_location="cpu",weights_only=True)[:,0].float().to(D);y=item[f"delta_{kind}"][:,0].float().to(D);m=models[kind];mean=m["mean"].unsqueeze(2);center=y-mean
   preds={"zero":torch.zeros_like(y),"mean":mean.expand_as(y),"oracle_basis8":mean+torch.einsum("lhtf,lhfr,lhgr->lhtg",center,m["basis"],m["basis"]),"linear":torch.einsum("lhtf,lhfg->lhtg",x,m["w"][:,:,:F])+m["bias"].unsqueeze(2),"linear_rank8":torch.einsum("lhtf,lhfg->lhtg",x,m["w8"])+m["bias"].unsqueeze(2)}
   de=float(y.square().sum())
   for name,p in preds.items():acc[(kind,name)][0]+=de;acc[(kind,name)][1]+=float((y-p).square().sum())
rows=[]
for (kind,name),(de,ee) in acc.items():
 if de:rows.append(dict(kind=kind,method=name,remaining_delta=(ee/de)**.5,explained_energy=1-ee/de))
out=dict(train_examples=f"<{args.train_cutoff}",test_examples=sorted(test_examples),train_token_cap_per_doc=32,rank=8,rows=rows);(R/f"results/{args.output_tag}_summary.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
torch.save({kind:{name:t.detach().cpu().half() for name,t in model.items() if name in ["w","w8","w16","w32","w64","bias"]} for kind,model in models.items()},R/f"results/{args.output_tag}.pt")
