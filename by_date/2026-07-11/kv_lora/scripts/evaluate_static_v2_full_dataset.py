#!/usr/bin/env python3
import argparse,csv,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,"/raid/home/hming/FusionRAG-pca-analysis")
from transformers import AutoConfig,AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import load_kv_and_generate
from test_fusionrag_reflect_preprocess_exp import PreprocessScope,RecallMethod,load_model,prepare_reflect_data

p=argparse.ArgumentParser();p.add_argument("--shard",type=int,required=True);p.add_argument("--num-shards",type=int,default=8);p.add_argument("--max-new-tokens",type=int,default=32);a=p.parse_args()
ROOT=Path("/raid/home/hming/FusionRAG-pca-analysis");E=ROOT/"MOTIVATION_EXPERIMENTS/kv_lora";CACHE=Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique/kv_cache")
OUT=E/"results/static_v2_full_dataset";OUT.mkdir(parents=True,exist_ok=True);out=OUT/f"shard_{a.shard:02d}.jsonl"
done=set()
if out.exists():
 for line in out.read_text().splitlines():
  z=json.loads(line);done.add((z["example"],z["subquery"]))
DEV="cuda:0";MP="/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
tok=AutoTokenizer.from_pretrained(MP,trust_remote_code=True);cfg=AutoConfig.from_pretrained(MP,trust_remote_code=True);cfg._attn_implementation="sdpa";model,_=load_model("qwen3",MP,cfg,DEV,False);model.eval()
qs,system,_,_=prepare_reflect_data(str(ROOT/"data/result_reflect.json"),tok,"/mnt/qjhs-sh-lab-01/models/bge-m3","qwen3",topk=10,max_main_questions=200,preprocess=False,recall_method=RecallMethod.BGE,preprocess_scope=PreprocessScope.GLOBAL)
cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=8192,device=DEV,dtype=model.dtype,passage_len=8192,model=model)

def run(passages,ids,rate,inject=None):
 state={};t=time.time()
 def cb(phase,past,*_):
  n=sum(len(x) for x in passages[:-1])
  if phase=="after_reprocess" and rate==1:state["full_v"]=torch.stack([x[:,:,:n,:].detach().cpu().half() for x in past.value_cache])
  if phase=="before_reprocess" and inject is not None:
   s=len(passages[0])
   for doc in passages[1:-1]:
    for l,x in enumerate(past.value_cache):x[:,:,s:s+len(doc),:].copy_(inject[l,:,:,s:s+len(doc),:].to(x.device))
    s+=len(doc)
 generated,_,profile=load_kv_and_generate(model,tok,cache,passages,str(CACHE),ids[0],max_new_tokens=a.max_new_tokens,revert_rope=True,reprocess_method="FusionRAG",rate=rate,preprocess=False,device=DEV,chunk_ids=ids[1],kv_snapshot_callback=cb)
 g=generated
 while isinstance(g,torch.Tensor) and g.ndim>1:g=g[0]
 while isinstance(g,(list,tuple)) and g and isinstance(g[0],(list,tuple,torch.Tensor)):g=g[0]
 if isinstance(g,torch.Tensor):g=g.tolist()
 text=tok.decode(g,skip_special_tokens=True)
 return text,state.get("full_v"),time.time()-t,profile

for ex,q in enumerate(qs):
 if ex%a.num_shards!=a.shard or not q.get("should_test",True):continue
 for si,sub in enumerate(q["sub_questions"]):
  if (ex,si) in done:continue
  ids=list(sub["chunk_ids"][:10]);docs=[q["doc_tensors"][i-1] for i in ids]
  qt=torch.tensor(tok.encode("<|im_end|>\n<|im_start|>user\nQuestion: "+sub["query"]+"\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ",add_special_tokens=False))
  passages=[system]+docs+[qt];chunk=[0]+ids
  full_text,full_v,offline_sec,_=run(passages,[ex,chunk],1)
  raw_text,_,raw_sec,_=run(passages,[ex,chunk],0)
  v2_text,_,v2_sec,_=run(passages,[ex,chunk],0,full_v)
  rec=dict(example=ex,subquery=si,question=sub["query"],answer=sub["answer"],chunk_ids=ids,max_new_tokens=a.max_new_tokens,full_rate1=full_text,raw_rate0=raw_text,static_v2_rawK_fullV_rate0=v2_text,offline_fullv_seconds=offline_sec,raw_online_seconds=raw_sec,v2_online_seconds=v2_sec)
  with out.open("a") as f:f.write(json.dumps(rec,ensure_ascii=False)+"\n")
  print(json.dumps(dict(example=ex,subquery=si),ensure_ascii=False),flush=True)
