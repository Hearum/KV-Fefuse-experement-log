#!/usr/bin/env python3
import csv,json,sys
from pathlib import Path
import torch
sys.path.insert(0,"/raid/home/hming/FusionRAG-pca-analysis")
from transformers import AutoConfig,AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import load_kv_and_generate
from test_fusionrag_reflect_preprocess_exp import PreprocessScope,RecallMethod,load_model,prepare_reflect_data
R=Path("/raid/home/hming/FusionRAG-pca-analysis");O=R/"MOTIVATION_EXPERIMENTS/kv_lora/results";M="/mnt/qjhs-sh-lab-01/models/Qwen3-32B";DEV="cuda:0";torch.manual_seed(0)
tok=AutoTokenizer.from_pretrained(M,trust_remote_code=True);cfg=AutoConfig.from_pretrained(M,trust_remote_code=True);cfg._attn_implementation="sdpa";model,_=load_model("qwen3",M,cfg,DEV,False);model.eval();qs,system,_,_=prepare_reflect_data(str(R/"data/result_reflect.json"),tok,"/mnt/qjhs-sh-lab-01/models/bge-m3","qwen3",topk=10,max_main_questions=1,preprocess=False,recall_method=RecallMethod.BGE,preprocess_scope=PreprocessScope.GLOBAL)
data=qs[0];sub=data["sub_questions"][0];ids=sub["chunk_ids"];docs=[data["doc_tensors"][i-1] for i in ids];q=torch.tensor(tok.encode("<|im_end|>\n<|im_start|>user\nQuestion: "+sub["query"]+"\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ",add_special_tokens=False));passages=[system]+docs+[q];start=len(system);n=sum(map(len,docs));root=Path("/raid/home/hming/fusionrag-reflect-qwen3-smoke-cache-preprocess/Qwen3-32B/musique");ranks=[1,2,4,8,16,32,64,128];rows=[];summary=[]
for source,path in (("raw",root/"kv_cache"),("preprocess",root/"preprocess_kv_cache_global_topk10_bge")):
 cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=8192,device=DEV,dtype=model.dtype,passage_len=8192,model=model);st={}
 def snap(phase,past,selected,*_):
  p="b" if phase=="before_reprocess" else "a"
  for kind,caches in (("k",past.key_cache),("v",past.value_cache)):st[p+kind]=torch.stack([x[:,:,start:start+n,:].detach().cpu().float() for x in caches])
  if selected is not None:st["selected"]=[int(x) for x in selected if start<=int(x)<start+n]
 load_kv_and_generate(model,tok,cache,passages,str(path),0,max_new_tokens=1,revert_rope=True,reprocess_method="FusionRAG",rate=1.0,preprocess=source=="preprocess",device=DEV,chunk_ids=[0]+ids,kv_snapshot_callback=snap)
 for kind in ("k","v"):
  B=st["b"+kind];Delta=st["a"+kind]-B;g=float(Delta.norm()/B.norm().clamp_min(1e-20));summary.append({"source":source,"kind":kind,"doc_tokens":n,"selected_tokens":len(st["selected"]),"original_gap":g,"delta_l2":float(Delta.norm()),"base_l2":float(B.norm())})
  for layer in range(64):
   d=Delta[layer,0].permute(1,0,2).reshape(n,-1);b=B[layer].flatten();de=float(d.square().sum());be=float(b.square().sum());qrank=min(128,min(d.shape));_,s,_=torch.svd_lowrank(d,q=qrank,niter=2);capt=torch.cumsum(s.square(),0)
   for rank in ranks:
    used=min(rank,len(s));expl=min(1.0,float(capt[used-1])/max(de,1e-20));tail=max(0.0,de*(1-expl));rows.append({"source":source,"kind":kind,"layer":layer,"rank":rank,"original_gap":(de/max(be,1e-20))**.5,"delta_recovery_error":(tail/max(de,1e-20))**.5,"final_kv_error":(tail/max(be,1e-20))**.5,"delta_cosine":expl**.5,"explained_variance":expl})
 del st,cache
TAG="qwen3_32b_ex0_rate1_adapter_rank_distortion";f=O/f"{TAG}.csv"
with f.open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
agg=[]
for src in ("raw","preprocess"):
 for kind in ("k","v"):
  for rank in ranks:
   x=[z for z in rows if z["source"]==src and z["kind"]==kind and z["rank"]==rank];agg.append({"source":src,"kind":kind,"rank":rank,"mean_original_gap":sum(z["original_gap"] for z in x)/len(x),"mean_delta_recovery_error":sum(z["delta_recovery_error"] for z in x)/len(x),"mean_final_kv_error":sum(z["final_kv_error"] for z in x)/len(x),"mean_explained_variance":sum(z["explained_variance"] for z in x)/len(x)})
(O/f"{TAG}_summary.json").write_text(json.dumps({"global":summary,"layer_mean":agg,"svd":"torch.svd_lowrank q=128 niter=2 seed=0"},indent=2));print(json.dumps(summary,indent=2))
