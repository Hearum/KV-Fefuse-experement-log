#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import torch
from transformers import AutoConfig

ROOT=Path("/raid/home/hming/FusionRAG-pca-analysis");sys.path.insert(0,str(ROOT))
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect_preprocess_exp import load_model

def main():
 p=argparse.ArgumentParser();p.add_argument("--token-cache",required=True);p.add_argument("--start",type=int,default=0);p.add_argument("--count",type=int,default=2);p.add_argument("--stride",type=int,default=64);p.add_argument("--output-dir",required=True);a=p.parse_args()
 mp="/home/hming/models/Qwen3-8B";dev="cuda:0";out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 cfg=AutoConfig.from_pretrained(mp,trust_remote_code=True);cfg._attn_implementation="sdpa";model,_=load_model("qwen3",mp,cfg,dev,False);model.eval()
 cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=512,device=dev,dtype=model.dtype,passage_len=512,model=model)
 def capture(ids):
  for layer in range(len(cache.key_cache)):cache.past_tokens[layer]=0
  ids=ids.unsqueeze(0).to(dev);pos=torch.arange(ids.shape[1],device=dev)
  with torch.no_grad():model(input_ids=ids,cache_position=pos,past_key_values=cache,use_cache=True,return_dict=False)
  return (torch.stack([x[:,:,:ids.shape[1],:].detach().cpu().half() for x in cache.key_cache]),torch.stack([x[:,:,:ids.shape[1],:].detach().cpu().half() for x in cache.value_cache]))
 def rotate(keys,delta):
  z=keys.to(dev);pos=torch.full((1,z.shape[-2]),delta,device=dev)
  try:c,s=model.model.rotary_emb(z[0],pos)
  except AttributeError:c,s=model.model.layers[0].self_attn.rotary_emb(z[0],pos)
  c=c.unsqueeze(0).unsqueeze(2);s=s.unsqueeze(0).unsqueeze(2);half=z.shape[-1]//2
  return (z*c+torch.cat([-z[...,half:],z[...,:half]],-1)*s).cpu()
 ids=torch.load(a.token_cache,map_location="cpu",weights_only=True);manifest=[]
 for sample in range(a.start,a.start+a.count):
  begin=sample*a.stride;span=ids[begin:begin+384].long()
  if len(span)<384:break
  prefix,target=span[:256],span[256:];offline_k,offline_v=capture(target);full_k,full_v=capture(torch.cat([prefix,target]));full_target_k=rotate(full_k[:,:,:,256:,:],-256);full_target_v=full_v[:,:,:,256:]
  dk=full_target_k.float()-offline_k.float();dv=full_target_v.float()-offline_v.float();idx=torch.linspace(0,127,8).long()
  own=torch.cat([offline_k[:,0,:,idx],offline_v[:,0,:,idx]],-1);prefix_k=full_k[:,0,:,:256].float().mean(2);prefix_v=full_v[:,0,:,:256].float().mean(2);prefix_features=torch.cat([prefix_k,prefix_v,prefix_k,prefix_v],-1).half();delta=torch.cat([dk[:,0,:,idx],dv[:,0,:,idx]],-1).half()
  metrics={"k_original_gap":float(dk.square().sum().sqrt()/offline_k.float().square().sum().sqrt()),"v_original_gap":float(dv.square().sum().sqrt()/offline_v.float().square().sum().sqrt())}
  artifact={"model":"Qwen3-8B","sample":sample,"token_start":begin,"prefix_tokens":256,"target_tokens":128,"sampled_positions":idx,"own_kv":own,"prefix_features":prefix_features,"delta_kv":delta,"metrics":metrics}
  path=out/f"sample{sample:07d}.pt";torch.save(artifact,path);row={"sample":sample,"path":path.name,"bytes":path.stat().st_size,**metrics};manifest.append(row);print(json.dumps(row),flush=True)
 (out/f"manifest_{a.start}_{a.start+a.count}.json").write_text(json.dumps(manifest,indent=2)+"\n")
if __name__=="__main__":main()
