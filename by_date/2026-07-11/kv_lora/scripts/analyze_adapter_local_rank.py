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
data=qs[0];sub=data["sub_questions"][0];ids=sub["chunk_ids"];docs=[data["doc_tensors"][i-1] for i in ids];lens=[len(x) for x in docs];q=torch.tensor(tok.encode("<|im_end|>\n<|im_start|>user\nQuestion: "+sub["query"]+"\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ",add_special_tokens=False));passages=[system]+docs+[q];start=len(system);n=sum(lens);root=Path("/raid/home/hming/fusionrag-reflect-qwen3-smoke-cache-preprocess/Qwen3-32B/musique");ranks=[1,2,4,8,16,32,64,128];rows=[]
def spectrum(mat,maxq):
 total=float(mat.square().sum());mind=min(mat.shape)
 if mind<=maxq: s=torch.linalg.svdvals(mat)
 else: _,s,_=torch.svd_lowrank(mat,q=maxq,niter=2)
 return s.square(),total,mind
for source,path in (("raw",root/"kv_cache"),("preprocess",root/"preprocess_kv_cache_global_topk10_bge")):
 cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=8192,device=DEV,dtype=model.dtype,passage_len=8192,model=model);st={}
 def snap(phase,past,selected,*_):
  p="b" if phase=="before_reprocess" else "a"
  for kind,caches in (("k",past.key_cache),("v",past.value_cache)):st[p+kind]=torch.stack([x[:,:,start:start+n,:].detach().cpu().float() for x in caches])
 load_kv_and_generate(model,tok,cache,passages,str(path),0,max_new_tokens=1,revert_rope=True,reprocess_method="FusionRAG",rate=1.0,preprocess=source=="preprocess",device=DEV,chunk_ids=[0]+ids,kv_snapshot_callback=snap)
 for kind in ("k","v"):
  B=st["b"+kind];D=st["a"+kind]-B
  for layer in range(64):
   for h in range(8):
    mat=D[layer,0,h];e,total,mind=spectrum(mat,64);cum=torch.cumsum(e,0)
    for rank in ranks:
     expl=1.0 if rank>=mind else min(1.0,float(cum[min(rank,len(e))-1])/max(total,1e-20));rows.append({"source":source,"kind":kind,"granularity":"head","layer":layer,"unit":h,"unit_tokens":n,"rank":rank,"explained_variance":expl,"delta_recovery_error":max(0,1-expl)**.5})
   off=0
   for ci,ln in enumerate(lens):
    mat=D[layer,0,:,off:off+ln,:].permute(1,0,2).reshape(ln,-1);off+=ln;e,total,mind=spectrum(mat,128);cum=torch.cumsum(e,0)
    for rank in ranks:
     expl=1.0 if rank>=mind else min(1.0,float(cum[min(rank,len(e))-1])/max(total,1e-20));rows.append({"source":source,"kind":kind,"granularity":"chunk","layer":layer,"unit":ci,"unit_tokens":ln,"rank":rank,"explained_variance":expl,"delta_recovery_error":max(0,1-expl)**.5})
 del st,cache
TAG="qwen3_32b_ex0_rate1_adapter_local_rank";f=O/f"{TAG}.csv"
with f.open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
agg=[]
for src in ("raw","preprocess"):
 for kind in ("k","v"):
  for gran in ("head","chunk"):
   for rank in ranks:
    x=[z for z in rows if z["source"]==src and z["kind"]==kind and z["granularity"]==gran and z["rank"]==rank];agg.append({"source":src,"kind":kind,"granularity":gran,"rank":rank,"mean_explained_variance":sum(z["explained_variance"] for z in x)/len(x),"mean_delta_recovery_error":sum(z["delta_recovery_error"] for z in x)/len(x),"units":len(x)})
(O/f"{TAG}_summary.json").write_text(json.dumps({"chunk_lengths":lens,"aggregate":agg,"svd":"exact when min_dim<=q, otherwise svd_lowrank niter=2"},indent=2));print("done",len(rows))
