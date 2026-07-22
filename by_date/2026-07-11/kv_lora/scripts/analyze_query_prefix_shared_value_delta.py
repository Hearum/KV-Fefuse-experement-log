#!/usr/bin/env python3
import csv,json,sys
from pathlib import Path
import torch
sys.path.insert(0,"/raid/home/hming/FusionRAG-pca-analysis")
from transformers import AutoConfig,AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect_preprocess_exp import PreprocessScope,RecallMethod,load_model,prepare_reflect_data
R=Path("/raid/home/hming/FusionRAG-pca-analysis");O=R/"MOTIVATION_EXPERIMENTS/kv_lora/results";M="/mnt/qjhs-sh-lab-01/models/Qwen3-32B";D="cuda:0"
t=AutoTokenizer.from_pretrained(M,trust_remote_code=True);c=AutoConfig.from_pretrained(M,trust_remote_code=True);c._attn_implementation="sdpa";m,_=load_model("qwen3",M,c,D,False);m.eval();qs,sy,_,_=prepare_reflect_data(str(R/"data/result_reflect.json"),t,"/mnt/qjhs-sh-lab-01/models/bge-m3","qwen3",topk=10,max_main_questions=1,preprocess=False,recall_method=RecallMethod.BGE,preprocess_scope=PreprocessScope.GLOBAL)
data=qs[0];sub=data["sub_questions"][0];ids=sub["chunk_ids"];docs=[data["doc_tensors"][i-1] for i in ids];doc_len=sum(map(len,docs))
queries=[("original",sub["query"]),("short","Which network includes National Cycle Route 57?"),("formal","Identify the larger cycling network containing National Cycle Route 57."),("reasoning",sub["query"]+" Explain step by step."),("related","Where does National Cycle Route 57 start and finish?"),("weather","What will the weather be tomorrow?"),("france","What is the capital of France?"),("math","What is 17 times 23?")]
def prefix_ids(q):return t.encode("<|im_end|>\n<|im_start|>user\nQuestion: "+q+"\n/no_think<|im_end|>\n<|im_start|>assistant\nDocuments:\n",add_special_tokens=False)
raw_ids=[prefix_ids(q) for _,q in queries];fixed=max(map(len,raw_ids));pad=t.encode("\n",add_special_tokens=False)[0];prefixes=[torch.tensor(x+[pad]*(fixed-len(x)),dtype=torch.long) for x in raw_ids]
def capture_value(x,doc_start):
 z=x.unsqueeze(0).to(D);cache=StaticCache(config=m.config,max_batch_size=1,max_cache_len=8192,device=D,dtype=m.dtype,passage_len=8192,model=m);pos=torch.arange(z.shape[1],device=D)
 with torch.no_grad():m(inputs_embeds=m.model.embed_tokens(z),cache_position=pos,past_key_values=cache,return_dict=False,use_cache=True)
 return torch.stack([v[:,:,doc_start:doc_start+doc_len,:].detach().cpu().to(torch.bfloat16) for v in cache.value_cache])
full=[]
for p in prefixes:full.append(capture_value(torch.cat([sy,p]+docs),len(sy)+len(p)))
root=Path("/raid/home/hming/fusionrag-reflect-qwen3-smoke-cache-preprocess/Qwen3-32B/musique")
def load_base(subdir):
 vals=[torch.load(root/subdir/f"0_{i}_value.pt",weights_only=True,map_location="cpu").to(torch.bfloat16) for i in ids]
 return torch.cat(vals,dim=3)
rows=[];pairs=[]
for source,base in (("raw",load_base("kv_cache")),("preprocess",load_base("preprocess_kv_cache_global_topk10_bge"))):
 deltas=[(x-base) for x in full]
 for i in range(len(deltas)):
  for j in range(i+1,len(deltas)):
   a=deltas[i].float().flatten();b=deltas[j].float().flatten();pairs.append({"source":source,"a":queries[i][0],"b":queries[j][0],"cosine":float(torch.nn.functional.cosine_similarity(a,b,dim=0)),"relative_difference":float((a-b).norm()/a.norm().clamp_min(1e-12))})
 for l in range(64):
  X=torch.stack([x[l].float().flatten() for x in deltas]);mean=X.mean(0,keepdim=True);C=X-mean;G=C@C.T;ev=torch.linalg.eigvalsh(G).clamp_min(0).flip(0);tot=ev.sum();frac=ev/tot.clamp_min(1e-24);unc=torch.linalg.eigvalsh(X@X.T).clamp_min(0).flip(0);uf=unc/unc.sum().clamp_min(1e-24)
  rows.append({"source":source,"layer":l,"centered_total_variance":float(tot),"pc1":float(frac[0]),"pc2":float(frac[:2].sum()),"pc4":float(frac[:4].sum()),"effective_rank":float(torch.exp(-(frac*frac.clamp_min(1e-24).log()).sum())),"uncentered_pc1":float(uf[0])})
tag="qwen3_32b_ex0_query_prefix_fixedlen_value_shared";fields=list(rows[0]);
with (O/f"{tag}_layers.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
(O/f"{tag}.json").write_text(json.dumps({"order":"system+fixed_length_query_prefix+all_docs","fixed_prefix_tokens":fixed,"original_prefix_lengths":dict((queries[i][0],len(raw_ids[i])) for i in range(len(queries))),"pairs":pairs},ensure_ascii=False,indent=2));print(json.dumps({"fixed":fixed,"rows":len(rows),"pairs":len(pairs)}))
