#!/usr/bin/env python3
import argparse,csv,json,math
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument("--source",choices=["raw","preprocess"],required=True);p.add_argument("--kind",choices=["k","v"],required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args()
R=Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora");D=R/"results/fixed_prefix_vary_target";ranks=[4,8,16,32,64]
groups={}
for manifest_path in sorted(D.glob("ex*/manifest.json")):
 m=json.loads(manifest_path.read_text()); runs={(x["target"],x["source"]):x for x in m["runs"]}
 good=[t for t in m["targets"] if (t,"raw") in runs and (t,"preprocess") in runs]
 if good:groups[m["example"]]=[(t,Path("/raid/home/hming/FusionRAG-pca-analysis")/runs[(t,a.source)]["path"]) for t in good]

doc_rows=[];group_rows=[]
for ex,items in groups.items():
 reps=[];layer_profiles=[];head_profiles=[]
 for target,path in items:
  x=torch.load(path,map_location="cpu");delta=x["delta_"+a.kind][:,0].float().to(a.device);base=x["base_"+a.kind+"_norm2_layer"].float()
  dn=float(delta.square().sum());bn=float(base.sum());layer_energy=delta.square().sum((1,2,3));head_energy=delta.square().sum((2,3));rep=delta.mean(2).flatten(1)
  row={"example":ex,"target":target,"source":a.source,"kind":a.kind,"tokens":delta.shape[2],"original_gap":math.sqrt(dn/max(bn,1e-30))}
  expl={r:[] for r in ranks}
  for layer in range(delta.shape[0]):
   mat=delta[layer].permute(1,0,2).reshape(delta.shape[2],-1);energy=float(mat.square().sum());s=torch.linalg.svdvals(mat);cum=torch.cumsum(s.square(),0)
   for rank in ranks:expl[rank].append(min(1.,float(cum[min(rank,len(s))-1])/max(energy,1e-30)))
  for rank in ranks:row[f"internal_rank{rank}_explained"]=sum(expl[rank])/len(expl[rank])
  doc_rows.append(row);reps.append(rep);layer_profiles.append(layer_energy);head_profiles.append(head_energy.flatten());del delta
 if len(reps)>=2:
  reps=torch.stack(reps);lp=torch.stack(layer_profiles);hp=torch.stack(head_profiles)
  def pair_cos(z):
   z=z/z.norm(dim=-1,keepdim=True).clamp_min(1e-20);q=z@z.T;idx=torch.triu_indices(len(z),len(z),1,device=z.device);return q[idx[0],idx[1]]
  # Direction similarity is computed per layer and then over all document pairs.
  direction=[]
  for layer in range(reps.shape[1]):direction.extend(pair_cos(reps[:,layer]).cpu().tolist())
  lcos=pair_cos(lp).cpu();hcos=pair_cos(hp).cpu()
  pc=[]
  for layer in range(reps.shape[1]):
   z=reps[:,layer];s=torch.linalg.svdvals(z);e=s.square();tot=float(e.sum());p=e/e.sum().clamp_min(1e-20);er=float(torch.exp(-(p*p.clamp_min(1e-20).log()).sum()))
   pc.append((float(e[:1].sum())/max(tot,1e-30),float(e[:2].sum())/max(tot,1e-30),float(e[:4].sum())/max(tot,1e-30),er))
  group_rows.append({"example":ex,"source":a.source,"kind":a.kind,"documents":len(items),"direction_cos_mean":sum(direction)/len(direction),"direction_cos_min":min(direction),"direction_cos_max":max(direction),"layer_profile_cos_mean":float(lcos.mean()),"head_profile_cos_mean":float(hcos.mean()),"uncentered_pc1_mean":sum(x[0] for x in pc)/len(pc),"uncentered_pc2_mean":sum(x[1] for x in pc)/len(pc),"uncentered_pc4_mean":sum(x[2] for x in pc)/len(pc),"uncentered_effective_rank_mean":sum(x[3] for x in pc)/len(pc)})

tag=f"fixed_prefix_vary_target_{a.source}_{a.kind}"
for name,rows in (("documents",doc_rows),("groups",group_rows)):
 with (R/"results"/f"{tag}_{name}.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def describe(vals):
 t=torch.tensor(vals);return {"mean":float(t.mean()),"std":float(t.std()),"median":float(t.median()),"p10":float(torch.quantile(t,.1)),"p90":float(torch.quantile(t,.9)),"n":len(vals)}
summary={"source":a.source,"kind":a.kind,"paired_documents":len(doc_rows),"groups":len(group_rows),"document_metrics":{},"group_metrics":{}}
for key in ["original_gap"]+[f"internal_rank{r}_explained" for r in ranks]:summary["document_metrics"][key]=describe([x[key] for x in doc_rows])
for key in ["direction_cos_mean","layer_profile_cos_mean","head_profile_cos_mean","uncentered_pc1_mean","uncentered_pc2_mean","uncentered_pc4_mean","uncentered_effective_rank_mean"]:summary["group_metrics"][key]=describe([x[key] for x in group_rows])
(R/"results"/f"{tag}_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
