#!/usr/bin/env python3
import argparse,json,random,sys,time
from pathlib import Path
import torch
sys.path.insert(0,"/raid/home/hming/FusionRAG-pca-analysis")
from transformers import AutoConfig,AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import load_kv_and_generate
from test_fusionrag_reflect_preprocess_exp import PreprocessScope,RecallMethod,load_model,prepare_reflect_data

p=argparse.ArgumentParser();p.add_argument("--shard",type=int,required=True);p.add_argument("--num-shards",type=int,default=8);p.add_argument("--max-examples",type=int,default=50);p.add_argument("--max-new-tokens",type=int,default=32);a=p.parse_args()
ROOT=Path("/raid/home/hming/FusionRAG-pca-analysis");E=ROOT/"MOTIVATION_EXPERIMENTS/kv_lora";CR=Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique");RAW=CR/"kv_cache";PP=CR/"preprocess_kv_cache_global_topk10_bge"
OUT=E/"results/real_static_bias_50";OUT.mkdir(parents=True,exist_ok=True);out=OUT/f"shard_{a.shard:02d}.jsonl";done=set()
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
def decode(g):
 while isinstance(g,torch.Tensor) and g.ndim>1:g=g[0]
 while isinstance(g,(list,tuple)) and g and isinstance(g[0],(list,tuple,torch.Tensor)):g=g[0]
 return tok.decode(g.tolist() if isinstance(g,torch.Tensor) else g,skip_special_tokens=True)
def generate(ex,passages,chunks,path,rate,bias=None,preprocess=False):
 def cb(phase,past,*_):
  if phase=="before_reprocess" and bias is not None:
   s=len(passages[0])
   for cid,doc in zip(chunks[1:],passages[1:-1]):
    b=bias[cid]
    for l,x in enumerate(past.value_cache):x[:,:,s:s+len(doc),:].add_(b[l].to(x.device))
    s+=len(doc)
 t=time.time();g,_,_=load_kv_and_generate(model,tok,cache,passages,str(path),ex,max_new_tokens=a.max_new_tokens,revert_rope=True,reprocess_method="FusionRAG",rate=rate,preprocess=preprocess,device=DEV,chunk_ids=chunks,kv_snapshot_callback=cb)
 return decode(g),time.time()-t

for ex,q in enumerate(qs):
 if ex%a.num_shards!=a.shard or not q.get("should_test",True):continue
 docs=q["doc_tensors"];rng=random.Random(90000+ex)
 other=[]
 for off in range(1,4):other.append(qs[(ex+off)%len(qs)]["doc_tensors"][0])
 orders=[list(range(len(docs))),list(reversed(range(len(docs))))];rng.shuffle(orders[0]);biases={i:[] for i in range(1,len(docs)+1)}
 for order in orders:
  seq=[system]+other+[docs[i] for i in order];v=capture(seq);s=len(system)+sum(map(len,other))
  for i in order:
   full=v[:,:,:,s:s+len(docs[i]),:];base=torch.load(RAW/f"{ex}_{i+1}_value.pt",map_location="cpu",weights_only=True);biases[i+1].append(full-base);s+=len(docs[i])
 bias={i:torch.stack(x).float().mean(0).half() for i,x in biases.items()}
 for si,sub in enumerate(q["sub_questions"]):
  if (ex,si) in done:continue
  ids=list(sub["chunk_ids"][:10]);selected=[docs[i-1] for i in ids];qt=torch.tensor(tok.encode("<|im_end|>\n<|im_start|>user\nQuestion: "+sub["query"]+"\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ",add_special_tokens=False));passages=[system]+selected+[qt];chunks=[0]+ids
  if not all((RAW/f"{ex}_{i}_value.pt").exists() for i in [0]+ids):continue
  full,tf=generate(ex,passages,chunks,RAW,1);raw,tr=generate(ex,passages,chunks,RAW,0);rv2,tv=generate(ex,passages,chunks,RAW,0,bias)
  pp=None;tp=None
  if all((PP/f"{ex}_{i}_value.pt").exists() for i in [0]+ids):pp,tp=generate(ex,passages,chunks,PP,0,preprocess=True)
  rec=dict(example=ex,subquery=si,question=sub["query"],answer=sub["answer"],chunk_ids=ids,full_rate1=full,raw_rate0=raw,random_cross_example_static_bias=rv2,topk_preprocess_rate0=pp,full_seconds=tf,raw_seconds=tr,random_bias_seconds=tv,topk_seconds=tp)
  with out.open("a") as f:f.write(json.dumps(rec,ensure_ascii=False)+"\n")
  print(json.dumps(dict(example=ex,subquery=si),ensure_ascii=False),flush=True)
