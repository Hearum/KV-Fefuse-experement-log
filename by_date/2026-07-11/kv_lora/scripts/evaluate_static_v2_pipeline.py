#!/usr/bin/env python3
import csv,json,sys
from pathlib import Path
import torch
sys.path.insert(0,"/raid/home/hming/FusionRAG-pca-analysis")
from transformers import AutoConfig,AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import load_kv_and_generate
from test_fusionrag_reflect_preprocess_exp import PreprocessScope,RecallMethod,load_model,prepare_reflect_data

ROOT=Path("/raid/home/hming/FusionRAG-pca-analysis");R=ROOT/"MOTIVATION_EXPERIMENTS/kv_lora";C=Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique")
DEV="cuda:0";MP="/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
tok=AutoTokenizer.from_pretrained(MP,trust_remote_code=True);cfg=AutoConfig.from_pretrained(MP,trust_remote_code=True);cfg._attn_implementation="sdpa"
model,_=load_model("qwen3",MP,cfg,DEV,False);model.eval()
questions,system,_,_=prepare_reflect_data(str(ROOT/"data/result_reflect.json"),tok,"/mnt/qjhs-sh-lab-01/models/bge-m3","qwen3",topk=10,max_main_questions=1,preprocess=False,recall_method=RecallMethod.BGE,preprocess_scope=PreprocessScope.GLOBAL)
data=questions[0];sub=data["sub_questions"][0];ids=list(sub["chunk_ids"][:10]);docs=[data["doc_tensors"][i-1] for i in ids]
query=torch.tensor(tok.encode("<|im_end|>\n<|im_start|>user\nQuestion: "+sub["query"]+"\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ",add_special_tokens=False))
passages=[system]+docs+[query];chunk_ids=[0]+ids
cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=8192,device=DEV,dtype=model.dtype,passage_len=8192,model=model)

def run(path,rate,preprocess):
 state={}
 def cb(phase,past,*_):
  if phase not in ("before_reprocess","after_reprocess"):return
  n=sum(len(x) for x in passages[:-1]);state[phase]=torch.stack([x[:,:,:n,:].detach().cpu().float() for x in past.value_cache])
 load_kv_and_generate(model,tok,cache,passages,str(path),0,max_new_tokens=1,revert_rope=True,reprocess_method="FusionRAG",rate=rate,preprocess=preprocess,device=DEV,chunk_ids=chunk_ids,kv_snapshot_callback=cb)
 return state["before_reprocess"],state["after_reprocess"]

rows=[]
for source,base_name,v2_name in [("raw","kv_cache","kv_cache_static_v2_ex0_t1to5"),("preprocess","preprocess_kv_cache_global_topk10_bge","preprocess_kv_cache_global_topk10_bge_static_v2_ex0_t1to5")]:
 base,_=run(C/base_name,0,source=="preprocess");v2,_=run(C/v2_name,0,source=="preprocess");_,full=run(C/base_name,1,source=="preprocess")
 start=len(system)
 for order,(cid,doc) in enumerate(zip(ids,docs)):
  sl=slice(start,start+len(doc));b=base[:,:,:,sl,:];q=v2[:,:,:,sl,:];f=full[:,:,:,sl,:]
  for method,pred in [("source_rate0",b),("static_v2_rate0",q)]:
   err=(pred-f).square().sum();den=f.square().sum();dot=(pred*f).sum()
   rows.append(dict(source=source,doc_order=order,chunk_id=cid,bias_available=cid<=5,method=method,tokens=len(doc),relative_l2=float((err/den).sqrt()),cosine=float(dot/(pred.square().sum()*den).sqrt())))
  start+=len(doc)
out=R/"results/static_v2_pipeline_ex0.csv"
with out.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(R/"results/static_v2_pipeline_ex0_meta.json").write_text(json.dumps(dict(example=0,sub_query=0,chunk_ids=ids,query=sub["query"]),indent=2)+"\n")
print(json.dumps(rows,indent=2))
