#!/usr/bin/env python3
import argparse,csv,json,sys,traceback
from pathlib import Path
import torch
sys.path.insert(0,"/raid/home/hming/FusionRAG-pca-analysis")
from transformers import AutoConfig,AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import load_kv_and_generate
from test_fusionrag_reflect_preprocess_exp import PreprocessScope,RecallMethod,load_model,prepare_reflect_data
P=argparse.ArgumentParser();P.add_argument("--start",type=int,required=True);P.add_argument("--end",type=int,required=True);P.add_argument("--part",required=True);a=P.parse_args();torch.manual_seed(0);torch.set_num_threads(8)
R=Path("/raid/home/hming/FusionRAG-pca-analysis");O=R/"MOTIVATION_EXPERIMENTS/kv_lora/results/adapter_rank50_parts";O.mkdir(parents=True,exist_ok=True);M="/mnt/qjhs-sh-lab-01/models/Qwen3-32B";DEV="cuda:0";ROOT=Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique");ranks=[4,8,16,32,64]
tok=AutoTokenizer.from_pretrained(M,trust_remote_code=True);cfg=AutoConfig.from_pretrained(M,trust_remote_code=True);cfg._attn_implementation="sdpa";model,_=load_model("qwen3",M,cfg,DEV,False);model.eval();questions,system,_,_=prepare_reflect_data(str(R/"data/result_reflect.json"),tok,"/mnt/qjhs-sh-lab-01/models/bge-m3","qwen3",topk=10,max_main_questions=a.end,preprocess=False,recall_method=RecallMethod.BGE,preprocess_scope=PreprocessScope.GLOBAL)
cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=8192,device=DEV,dtype=model.dtype,passage_len=8192,model=model);global_rows=[];layer_rows=[];fail=[]
for ex in range(a.start,a.end):
 try:
  data=questions[ex];sub=data["sub_questions"][0];ids=sub["chunk_ids"];docs=[data["doc_tensors"][i-1] for i in ids];q=torch.tensor(tok.encode("<|im_end|>\n<|im_start|>user\nQuestion: "+sub["query"]+"\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ",add_special_tokens=False));passages=[system]+docs+[q];start=len(system);n=sum(map(len,docs))
  for source,path in (("raw",ROOT/"kv_cache"),("preprocess",ROOT/"preprocess_kv_cache_global_topk10_bge")):
   missing=[i for i in ids if not (path/f"{ex}_{i}_key.pt").exists()]
   if missing:raise FileNotFoundError(f"{source} missing chunks {missing}")
   st={}
   def snap(phase,past,selected,*_):
    p="b" if phase=="before_reprocess" else "a"
    for kind,caches in (("k",past.key_cache),("v",past.value_cache)):st[p+kind]=torch.stack([x[:,:,start:start+n,:].detach().cpu().float() for x in caches])
    if selected is not None:st["selected"]=sum(start<=int(x)<start+n for x in selected)
   load_kv_and_generate(model,tok,cache,passages,str(path),ex,max_new_tokens=1,revert_rope=True,reprocess_method="FusionRAG",rate=1.0,preprocess=source=="preprocess",device=DEV,chunk_ids=[0]+ids,kv_snapshot_callback=snap)
   for kind in ("k","v"):
    B=st["b"+kind];D=st["a"+kind]-B;global_rows.append({"example":ex,"source":source,"kind":kind,"chunks":len(ids),"tokens":n,"selected":st["selected"],"original_gap":float(D.norm()/B.norm().clamp_min(1e-20))})
    for layer in range(64):
     mat=D[layer,0].permute(1,0,2).reshape(n,-1);de=float(mat.square().sum());be=float(B[layer].square().sum());qrank=min(64,min(mat.shape));_,s,_=torch.svd_lowrank(mat,q=qrank,niter=2);cum=torch.cumsum(s.square(),0)
     for rank in ranks:
      used=min(rank,len(s));expl=min(1.0,float(cum[used-1])/max(de,1e-20));tail=max(0,de*(1-expl));layer_rows.append({"example":ex,"source":source,"kind":kind,"layer":layer,"rank":rank,"original_gap":(de/max(be,1e-20))**.5,"delta_recovery_error":(tail/max(de,1e-20))**.5,"final_kv_error":(tail/max(be,1e-20))**.5,"explained_variance":expl})
   del st
  print(json.dumps({"part":a.part,"example":ex,"status":"done"}),flush=True)
 except Exception as e:
  fail.append({"example":ex,"error":repr(e),"traceback":traceback.format_exc()});print(json.dumps({"part":a.part,"example":ex,"status":"failed","error":repr(e)}),flush=True)
def write(name,rows):
 if rows:
  with (O/f"{a.part}_{name}.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
write("global",global_rows);write("layers",layer_rows);(O/f"{a.part}_manifest.json").write_text(json.dumps({"start":a.start,"end":a.end,"completed":sorted(set(x["example"] for x in global_rows)),"failures":fail},indent=2))
