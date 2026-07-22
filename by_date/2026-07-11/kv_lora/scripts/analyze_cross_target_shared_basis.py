#!/usr/bin/env python3
import argparse,csv,json,math
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--kind",required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args()
R=Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora");D=R/"results/fixed_prefix_vary_target";ranks=[1,2,4,8,16,32,64]
docs=[]
for mp in sorted(D.glob("ex*/manifest.json")):
 m=json.loads(mp.read_text());runs={(x["target"],x["source"]):x for x in m["runs"]}
 good=[t for t in m["targets"] if (t,"raw") in runs and (t,"preprocess") in runs]
 if len(good)==7:
  for pos,t in enumerate(good):docs.append({"example":m["example"],"target":t,"position":pos,"path":Path("/raid/home/hming/FusionRAG-pca-analysis")/runs[(t,a.source)]["path"]})
payload={}
for d in docs:payload[(d["example"],d["target"])]=torch.load(d["path"],map_location="cpu")
acc={(d["example"],d["target"],r):{"de":0.,"be":0.,"ee":0.} for d in docs for r in ranks}
for fold in range(7):
 train=[d for d in docs if d["position"]!=fold];test=[d for d in docs if d["position"]==fold]
 for layer in range(64):
  for head in range(8):
   cov=torch.zeros((128,128),device=a.device)
   for d in train:
    x=payload[(d["example"],d["target"])]["delta_"+a.kind][layer,0,head].float().to(a.device);cov+=x.T@x
   _,vec=torch.linalg.eigh(cov);basis=vec.flip(1)
   for d in test:
    q=payload[(d["example"],d["target"])];x=q["delta_"+a.kind][layer,0,head].float().to(a.device);de=float(x.square().sum())
    for r in ranks:
     b=basis[:,:r];res=x-(x@b)@b.T;cell=acc[(d["example"],d["target"],r)];cell["de"]+=de;cell["ee"]+=float(res.square().sum())
  for d in test:
   be=float(payload[(d["example"],d["target"])]["base_"+a.kind+"_norm2_layer"][layer])
   for r in ranks:acc[(d["example"],d["target"],r)]["be"]+=be
rows=[]
for (ex,t,r),z in acc.items():rows.append({"example":ex,"target":t,"source":a.source,"kind":a.kind,"rank":r,"original_gap":math.sqrt(z["de"]/z["be"]),"delta_recovery_error":math.sqrt(z["ee"]/z["de"]),"final_kv_error":math.sqrt(z["ee"]/z["be"]),"explained_delta_energy":1-z["ee"]/z["de"]})
tag=f"cross_target_shared_basis_{a.source}_{a.kind}_7fold"
with (R/"results"/f"{tag}.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary=[]
for r in ranks:
 g=[x for x in rows if x["rank"]==r];z={"rank":r,"n":len(g)}
 for m in ["original_gap","delta_recovery_error","final_kv_error","explained_delta_energy"]:
  q=torch.tensor([x[m] for x in g]);z.update({m+"_mean":float(q.mean()),m+"_std":float(q.std()),m+"_median":float(q.median()),m+"_p10":float(torch.quantile(q,.1)),m+"_p90":float(torch.quantile(q,.9))})
 summary.append(z)
(R/"results"/f"{tag}_summary.json").write_text(json.dumps({"source":a.source,"kind":a.kind,"basis":"per-layer/head feature basis trained on other target positions; oracle token coefficients","documents":len(docs),"summary":summary},indent=2)+"\n");print(json.dumps(summary,indent=2))
