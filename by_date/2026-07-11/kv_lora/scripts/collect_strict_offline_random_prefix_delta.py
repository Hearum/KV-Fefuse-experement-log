#!/usr/bin/env python3
import argparse,json,random,sys
from pathlib import Path
import torch

torch.set_num_threads(4);torch.set_num_interop_threads(1)
HOME=Path("/raid/home/hming") if Path("/raid/home/hming/FusionRAG-pca-analysis").exists() else Path("/home/hming")
ROOT=HOME/"FusionRAG-pca-analysis";sys.path.insert(0,str(ROOT))
from transformers import AutoConfig,AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect_preprocess_exp import PreprocessScope,RecallMethod,load_model,prepare_reflect_data

p=argparse.ArgumentParser();p.add_argument("--shard",type=int,required=True);p.add_argument("--num-shards",type=int,default=8);p.add_argument("--target-examples",type=int,default=5);p.add_argument("--samples",type=int,default=8);p.add_argument("--prefix-docs",type=int,default=3);a=p.parse_args()
DEV="cuda:0";MP="/mnt/qjhs-sh-lab-01/models/Qwen3-32B";C=HOME/"fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique";OUT=ROOT/"MOTIVATION_EXPERIMENTS/kv_lora/results/strict_offline_random_prefix_m8k3";OUT.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained(MP,trust_remote_code=True);cfg=AutoConfig.from_pretrained(MP,trust_remote_code=True);cfg._attn_implementation="sdpa";model,_=load_model("qwen3",MP,cfg,DEV,False);model.eval()
qs,system,_,_=prepare_reflect_data(str(ROOT/"data/result_reflect.json"),tok,"/mnt/qjhs-sh-lab-01/models/bge-m3","qwen3",topk=10,max_main_questions=200,preprocess=False,recall_method=RecallMethod.BGE,preprocess_scope=PreprocessScope.GLOBAL)
cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=8192,device=DEV,dtype=model.dtype,passage_len=8192,model=model)
def capture(seq):
 for l in range(len(cache.key_cache)):cache.past_tokens[l]=0
 ids=torch.cat(seq).unsqueeze(0).to(DEV);pos=torch.arange(ids.shape[1],device=DEV)
 with torch.no_grad():model(input_ids=ids,cache_position=pos,past_key_values=cache,use_cache=True,return_dict=False)
 k=torch.stack([x[:,:,:ids.shape[1],:].detach().cpu().half() for x in cache.key_cache]);v=torch.stack([x[:,:,:ids.shape[1],:].detach().cpu().half() for x in cache.value_cache]);return k,v
def rotate(x,delta):
 if delta==0:return x
 pos=torch.full((1,x.shape[-2]),int(delta),device=DEV);z=x.to(DEV)
 try:c,s=model.model.rotary_emb(z[0],pos)
 except AttributeError:c,s=model.model.layers[0].self_attn.rotary_emb(z[0],pos)
 c=c.unsqueeze(0).unsqueeze(2);s=s.unsqueeze(0).unsqueeze(2);h=z.shape[-1]//2;r=torch.cat([-z[... ,h:],z[...,:h]],-1);return (z*c+r*s).cpu()
targets=[(ex,cid,d) for ex in range(a.target_examples) for cid,d in enumerate(qs[ex]["doc_tensors"],1)]
for ti,(ex,cid,xdoc) in enumerate(targets):
 if ti%a.num_shards!=a.shard:continue
 out=OUT/f"ex{ex:03d}_chunk{cid:03d}.pt"
 if out.exists():continue
 bases={}
 for src,dn in [("raw","kv_cache"),("preprocess","preprocess_kv_cache_global_topk10_bge")]:
  kp=C/dn/f"{ex}_{cid}_key.pt";vp=C/dn/f"{ex}_{cid}_value.pt"
  if kp.exists() and vp.exists():bases[src]=(torch.load(kp,map_location="cpu",weights_only=True),torch.load(vp,map_location="cpu",weights_only=True))
 if "raw" not in bases:continue
 deltas={s:{"k":[],"v":[]} for s in bases};meta=[]
 for sample in range(a.samples):
  rng=random.Random(700000+ex*10000+cid*100+sample);pool=list(range(50,len(qs)));rng.shuffle(pool);prefix=[];prefix_meta=[]
  for pe in pool:
   if not qs[pe]["doc_tensors"]:continue
   di=rng.randrange(len(qs[pe]["doc_tensors"]));prefix.append(qs[pe]["doc_tensors"][di]);prefix_meta.append([pe,di+1])
   if len(prefix)==a.prefix_docs:break
  seq=[system]+prefix+[xdoc];fk,fv=capture(seq);start=len(system)+sum(map(len,prefix));ks=fk[:,:,:,start:start+len(xdoc),:];vs=fv[:,:,:,start:start+len(xdoc),:];aligned=rotate(ks,-sum(map(len,prefix)))
  for src,(bk,bv) in bases.items():deltas[src]["k"].append((aligned.float()-bk.float()).half());deltas[src]["v"].append((vs.float()-bv.float()).half())
  meta.append(dict(sample=sample,prefix=prefix_meta,prefix_tokens=sum(map(len,prefix))))
 torch.save(dict(example=ex,chunk=cid,target_tokens=len(xdoc),samples=a.samples,prefix_docs=a.prefix_docs,delta={s:{k:torch.stack(v) for k,v in z.items()} for s,z in deltas.items()},contexts=meta),out)
 print(out,flush=True)
