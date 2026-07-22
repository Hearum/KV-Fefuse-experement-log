#!/usr/bin/env python3
import argparse,json,random,sys
from pathlib import Path
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
_repo = "/raid/home/hming/FusionRAG-pca-analysis" if Path("/raid/home/hming/FusionRAG-pca-analysis").exists() else "/home/hming/FusionRAG-pca-analysis"
sys.path.insert(0,_repo)
from transformers import AutoConfig,AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect_preprocess_exp import PreprocessScope,RecallMethod,load_model,prepare_reflect_data

p=argparse.ArgumentParser();p.add_argument("--shard",type=int,required=True);p.add_argument("--num-shards",type=int,default=8);p.add_argument("--max-examples",type=int,default=50);p.add_argument("--prefix-docs",type=int,default=3);p.add_argument("--save-bias-dir",type=str,default=None);p.add_argument("--save-bias-ns",type=str,default="16");p.add_argument("--output-tag",type=str,default=None);a=p.parse_args()
HOME_ROOT=Path("/raid/home/hming") if Path("/raid/home/hming/FusionRAG-pca-analysis").exists() else Path("/home/hming")
ROOT=HOME_ROOT/"FusionRAG-pca-analysis";E=ROOT/"MOTIVATION_EXPERIMENTS/kv_lora";CR=HOME_ROOT/"fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique";RAW=CR/"kv_cache";PP=CR/"preprocess_kv_cache_global_topk10_bge"
OUT=E/"results/bias_calibration_scaling"/(a.output_tag or f"m{a.prefix_docs}");OUT.mkdir(parents=True,exist_ok=True);out=OUT/f"shard_{a.shard:02d}.jsonl";done=set()
if out.exists():
 for line in out.read_text().splitlines():z=json.loads(line);done.add((z["example"],z["subquery"]))
DEV="cuda:0";MP="/mnt/qjhs-sh-lab-01/models/Qwen3-32B";tok=AutoTokenizer.from_pretrained(MP,trust_remote_code=True);cfg=AutoConfig.from_pretrained(MP,trust_remote_code=True);cfg._attn_implementation="sdpa";model,_=load_model("qwen3",MP,cfg,DEV,False);model.eval()
qs,system,_,_=prepare_reflect_data(str(ROOT/"data/result_reflect.json"),tok,"/mnt/qjhs-sh-lab-01/models/bge-m3","qwen3",topk=10,max_main_questions=a.max_examples,preprocess=False,recall_method=RecallMethod.BGE,preprocess_scope=PreprocessScope.GLOBAL)
cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=8192,device=DEV,dtype=model.dtype,passage_len=8192,model=model)
def capture(seq):
 for l in range(len(cache.key_cache)):cache.past_tokens[l]=0
 ids=torch.cat(seq).unsqueeze(0).to(DEV);pos=torch.arange(ids.shape[1],device=DEV)
 with torch.no_grad():model(input_ids=ids,cache_position=pos,past_key_values=cache,use_cache=True,return_dict=False)
 return torch.stack([x[:,:,:ids.shape[1],:].detach().cpu().half() for x in cache.value_cache])

Ns=[1,2,4,8,16]
for ex,q in enumerate(qs):
 if ex%a.num_shards!=a.shard or not q.get("should_test",True):continue
 docs=q["doc_tensors"];base={i+1:torch.load(RAW/f"{ex}_{i+1}_value.pt",map_location="cpu",weights_only=True) for i in range(len(docs))}
 sums={i+1:torch.zeros_like(base[i+1],dtype=torch.float32) for i in range(len(docs))};means={n:{} for n in Ns}
 for sample in range(16):
  group=sample//2;other=[qs[(ex+1+max(1,a.prefix_docs)*group+j)%len(qs)]["doc_tensors"][0] for j in range(a.prefix_docs)]
  order=list(range(len(docs)));random.Random(100000+ex*100+sample).shuffle(order)
  if sample%2:order=list(reversed(order))
  v=capture([system]+other+[docs[i] for i in order]);s=len(system)+sum(map(len,other))
  for i in order:sums[i+1]+=(v[:,:,:,s:s+len(docs[i]),:].float()-base[i+1].float());s+=len(docs[i])
  n=sample+1
  if n in Ns:
   for i in range(1,len(docs)+1):means[n][i]=(sums[i]/n).half()
 if a.save_bias_dir:
  for save_n in [int(x) for x in a.save_bias_ns.split(",") if x]:
   if save_n not in means:raise ValueError(f"Unsupported save bias N={save_n}")
   bias_dir=Path(a.save_bias_dir)/f"m{a.prefix_docs}n{save_n}";bias_dir.mkdir(parents=True,exist_ok=True)
   for i in range(1,len(docs)+1):torch.save(means[save_n][i],bias_dir/f"{ex}_{i}_value_bias.pt")
 for si,sub in enumerate(q["sub_questions"]):
  if (ex,si) in done:continue
  ids=list(sub["chunk_ids"][:10]);selected=[docs[i-1] for i in ids];full=capture([system]+selected);s=len(system);energy=0.;errs={"raw":0.,**{f"n{n}":0. for n in Ns},"topk":0.};topk_ok=all((PP/f"{ex}_{i}_value.pt").exists() for i in ids)
  for cid,doc in zip(ids,selected):
   target=full[:,:,:,s:s+len(doc),:].float();b=base[cid].float();energy+=float(target.square().sum());errs["raw"]+=float((b-target).square().sum())
   for n in Ns:errs[f"n{n}"]+=float((b+means[n][cid].float()-target).square().sum())
   if topk_ok:pv=torch.load(PP/f"{ex}_{cid}_value.pt",map_location="cpu",weights_only=True).float();errs["topk"]+=float((pv-target).square().sum())
   s+=len(doc)
  rec=dict(example=ex,subquery=si,prefix_docs=a.prefix_docs,chunk_ids=ids,topk_available=topk_ok,**{k:(v/energy)**.5 for k,v in errs.items() if k!="topk" or topk_ok})
  with out.open("a") as f:f.write(json.dumps(rec)+"\n")
  print(json.dumps(dict(example=ex,subquery=si)),flush=True)
