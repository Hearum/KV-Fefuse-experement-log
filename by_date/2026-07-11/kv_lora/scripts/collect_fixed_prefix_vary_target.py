#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, "/raid/home/hming/FusionRAG-pca-analysis")
from transformers import AutoConfig, AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import load_kv_and_generate
from test_fusionrag_reflect_preprocess_exp import PreprocessScope, RecallMethod, load_model, prepare_reflect_data

p=argparse.ArgumentParser();p.add_argument("--example",type=int,required=True);p.add_argument("--part",required=True);a=p.parse_args()
torch.manual_seed(0);torch.set_num_threads(8)
R=Path("/raid/home/hming/FusionRAG-pca-analysis"); O=R/"MOTIVATION_EXPERIMENTS/kv_lora/results/fixed_prefix_vary_target"/a.part;O.mkdir(parents=True,exist_ok=True)
M="/mnt/qjhs-sh-lab-01/models/Qwen3-32B";DEV="cuda:0";CR=Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique")
tok=AutoTokenizer.from_pretrained(M,trust_remote_code=True);cfg=AutoConfig.from_pretrained(M,trust_remote_code=True);cfg._attn_implementation="sdpa";model,_=load_model("qwen3",M,cfg,DEV,False);model.eval()
qs,system,_,_=prepare_reflect_data(str(R/"data/result_reflect.json"),tok,"/mnt/qjhs-sh-lab-01/models/bge-m3","qwen3",topk=10,max_main_questions=a.example+1,preprocess=False,recall_method=RecallMethod.BGE,preprocess_scope=PreprocessScope.GLOBAL)
data=qs[a.example];sub=data["sub_questions"][0];all_ids=list(range(1,min(10,len(data["doc_tensors"]))+1));prefix=all_ids[:3];targets=all_ids[3:]
q=torch.tensor(tok.encode("<|im_end|>\n<|im_start|>user\nQuestion: "+sub["query"]+"\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ",add_special_tokens=False))
cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=8192,device=DEV,dtype=model.dtype,passage_len=8192,model=model);runs=[];fail=[]
prefix_docs=[data["doc_tensors"][i-1] for i in prefix];start=len(system)+sum(map(len,prefix_docs))
for target in targets:
 target_doc=data["doc_tensors"][target-1];n=len(target_doc);passages=[system]+prefix_docs+[target_doc]+[q];ids=[0]+prefix+[target]
 for source,path in (("raw",CR/"kv_cache"),("preprocess",CR/"preprocess_kv_cache_global_topk10_bge")):
  missing=[i for i in prefix+[target] if not (path/f"{a.example}_{i}_key.pt").exists()]
  if missing: fail.append({"target":target,"source":source,"missing":missing});continue
  st={}
  def snap(phase,past,selected,*_):
   z="b" if phase=="before_reprocess" else "a"
   for kind,layers in (("k",past.key_cache),("v",past.value_cache)):st[z+kind]=torch.stack([x[:,:,start:start+n,:].detach().cpu().float() for x in layers])
   if selected is not None:st["selected"]=sum(start<=int(x)<start+n for x in selected)
  load_kv_and_generate(model,tok,cache,passages,str(path),a.example,max_new_tokens=1,revert_rope=True,reprocess_method="FusionRAG",rate=1.0,preprocess=source=="preprocess",device=DEV,chunk_ids=ids,kv_snapshot_callback=snap)
  if st.get("selected")!=n:raise RuntimeError(f"target {target}: selected {st.get('selected')} != {n}")
  payload={"delta_k":(st["ak"]-st["bk"]).half(),"delta_v":(st["av"]-st["bv"]).half(),"base_k_norm2_layer":st["bk"].square().flatten(1).sum(1),"base_v_norm2_layer":st["bv"].square().flatten(1).sum(1),"meta":{"example":a.example,"prefix":prefix,"prefix_tokens":start-len(system),"target":target,"target_tokens":n,"source":source}}
  f=O/f"target_{target:02d}_{source}.pt";torch.save(payload,f);runs.append({**payload["meta"],"path":str(f.relative_to(R))});print(json.dumps(runs[-1]),flush=True);del st,payload
(O/"manifest.json").write_text(json.dumps({"example":a.example,"prefix":prefix,"prefix_tokens":start-len(system),"targets":targets,"runs":runs,"failures":fail},indent=2)+"\n")
