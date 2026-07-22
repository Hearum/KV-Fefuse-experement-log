#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
HOME=Path("/raid/home/hming") if Path("/raid/home/hming/FusionRAG-pca-analysis").exists() else Path("/home/hming");ROOT=HOME/"FusionRAG-pca-analysis";sys.path.insert(0,str(ROOT))
from transformers import AutoConfig,AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect_preprocess_exp import PreprocessScope,RecallMethod,load_model,prepare_reflect_data
p=argparse.ArgumentParser();p.add_argument("--shard",type=int,required=True);p.add_argument("--num-shards",type=int,default=8);p.add_argument("--max-examples",type=int,default=50);a=p.parse_args()
DEV="cuda:0";MP="/mnt/qjhs-sh-lab-01/models/Qwen3-32B";C=HOME/"fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique/preprocess_kv_cache_global_topk10_bge";OUT=ROOT/"MOTIVATION_EXPERIMENTS/kv_lora/results/formal_preprocess_residual_50";OUT.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained(MP,trust_remote_code=True);cfg=AutoConfig.from_pretrained(MP,trust_remote_code=True);cfg._attn_implementation="sdpa";model,_=load_model("qwen3",MP,cfg,DEV,False);model.eval()
qs,system,_,_=prepare_reflect_data(str(ROOT/"data/result_reflect.json"),tok,"/mnt/qjhs-sh-lab-01/models/bge-m3","qwen3",topk=10,max_main_questions=a.max_examples,preprocess=False,recall_method=RecallMethod.BGE,preprocess_scope=PreprocessScope.GLOBAL)
cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=8192,device=DEV,dtype=model.dtype,passage_len=8192,model=model)
def capture(seq):
 for l in range(len(cache.key_cache)):cache.past_tokens[l]=0
 ids=torch.cat(seq).unsqueeze(0).to(DEV);pos=torch.arange(ids.shape[1],device=DEV)
 with torch.no_grad():model(input_ids=ids,cache_position=pos,past_key_values=cache,use_cache=True,return_dict=False)
 return torch.stack([x[:,:,:ids.shape[1],:].detach().cpu().half() for x in cache.key_cache]),torch.stack([x[:,:,:ids.shape[1],:].detach().cpu().half() for x in cache.value_cache])
def rotate(x,delta):
 if not delta:return x
 z=x.to(DEV);pos=torch.full((1,z.shape[-2]),int(delta),device=DEV)
 try:c,s=model.model.rotary_emb(z[0],pos)
 except AttributeError:c,s=model.model.layers[0].self_attn.rotary_emb(z[0],pos)
 c=c.unsqueeze(0).unsqueeze(2);s=s.unsqueeze(0).unsqueeze(2);h=z.shape[-1]//2;return (z*c+torch.cat([-z[...,h:],z[...,:h]],-1)*s).cpu()
manifest=[]
for ex,q in enumerate(qs):
 if ex%a.num_shards!=a.shard or not q.get("should_test",True):continue
 for si,sub in enumerate(q["sub_questions"]):
  out=OUT/f"ex{ex:03d}_sub{si:02d}.pt"
  if out.exists():continue
  ids=list(sub["chunk_ids"][:10]);docs=[q["doc_tensors"][i-1] for i in ids]
  if not all((C/f"{ex}_{i}_key.pt").exists() and (C/f"{ex}_{i}_value.pt").exists() for i in ids):continue
  fk,fv=capture([system]+docs);start=len(system);items=[];prefix_tokens=0
  for cid,doc in zip(ids,docs):
   n=len(doc);basek=torch.load(C/f"{ex}_{cid}_key.pt",map_location="cpu",weights_only=True);basev=torch.load(C/f"{ex}_{cid}_value.pt",map_location="cpu",weights_only=True)
   fullk=rotate(fk[:,:,:,start:start+n,:],-prefix_tokens);fullv=fv[:,:,:,start:start+n,:]
   items.append(dict(chunk=cid,prefix_tokens=prefix_tokens,tokens=n,delta_k=(fullk.float()-basek.float()).half(),delta_v=(fullv.float()-basev.float()).half()))
   start+=n;prefix_tokens+=n
  torch.save(dict(example=ex,subquery=si,chunk_ids=ids,items=items),out);manifest.append(dict(example=ex,subquery=si,path=out.name,docs=len(items)))
  print(out,flush=True)
(OUT/f"manifest_shard{a.shard}.json").write_text(json.dumps(manifest,indent=2)+"\n")
