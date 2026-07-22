#!/usr/bin/env python3
import csv,json,sys
from pathlib import Path
import torch
sys.path.insert(0,"/raid/home/hming/FusionRAG-pca-analysis")
from transformers import AutoConfig,AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect_preprocess_exp import PreprocessScope,RecallMethod,load_model,prepare_reflect_data
R=Path("/raid/home/hming/FusionRAG-pca-analysis");O=R/"MOTIVATION_EXPERIMENTS/kv_lora/results";M="/mnt/qjhs-sh-lab-01/models/Qwen3-32B";DEV="cuda:0"
tok=AutoTokenizer.from_pretrained(M,trust_remote_code=True);cfg=AutoConfig.from_pretrained(M,trust_remote_code=True);cfg._attn_implementation="sdpa";model,_=load_model("qwen3",M,cfg,DEV,False);model.eval();qs,system,_,_=prepare_reflect_data(str(R/"data/result_reflect.json"),tok,"/mnt/qjhs-sh-lab-01/models/bge-m3","qwen3",topk=10,max_main_questions=1,preprocess=False,recall_method=RecallMethod.BGE,preprocess_scope=PreprocessScope.GLOBAL)
data=qs[0];sub=data["sub_questions"][0];ids=sub["chunk_ids"];docs=[data["doc_tensors"][i-1] for i in ids];doc_len=sum(map(len,docs));orig=sub["query"]
queries=[
("para0","Which broader network contains National Cycle Route 57?","paraphrase",False),("para1","National Cycle Route 57 belongs to which cycling network?","paraphrase",False),("para2","Name the network that includes National Cycle Route 57.","paraphrase",False),("para3","Of what network is National Cycle Route 57 a part?","paraphrase",True),
("reason0",orig+" Answer briefly.","reasoning",False),("reason1",orig+" Explain using the supplied evidence.","reasoning",False),("reason2",orig+" Reason step by step before answering.","reasoning",False),("reason3",orig+" Compare the relevant facts and justify the answer.","reasoning",True),
("rel0","Where does National Cycle Route 57 begin?","related",False),("rel1","Where does National Cycle Route 57 end?","related",False),("rel2","Which places are connected by National Cycle Route 57?","related",False),("rel3","What kind of cycling route is National Cycle Route 57?","related",True),
("neg0","What is the capital of France?","negative",False),("neg1","What will the weather be tomorrow?","negative",False),("neg2","Compute 17 multiplied by 23.","negative",False),("neg3","Who wrote the novel Pride and Prejudice?","negative",True)]
def encode(q):return tok.encode("<|im_end|>\n<|im_start|>user\nQuestion: "+q+"\n/no_think<|im_end|>\n<|im_start|>assistant\nDocuments:\n",add_special_tokens=False)
raw=[encode(q) for _,q,_,_ in queries];fixed=max(map(len,raw));pad=tok.encode("\n",add_special_tokens=False)[0];prefix=[torch.tensor(x+[pad]*(fixed-len(x))) for x in raw]
def capture(p):
 x=torch.cat([system,p]+docs);z=x.unsqueeze(0).to(DEV);cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=8192,device=DEV,dtype=model.dtype,passage_len=8192,model=model);pos=torch.arange(z.shape[1],device=DEV)
 with torch.no_grad():model(inputs_embeds=model.model.embed_tokens(z),cache_position=pos,past_key_values=cache,return_dict=False,use_cache=True)
 start=len(system)+len(p);return torch.stack([v[:,:,start:start+doc_len,:].detach().cpu().to(torch.bfloat16) for v in cache.value_cache])
full=[capture(p) for p in prefix];root=Path("/raid/home/hming/fusionrag-reflect-qwen3-smoke-cache-preprocess/Qwen3-32B/musique")
def base(path):return torch.cat([torch.load(root/path/f"0_{i}_value.pt",weights_only=True,map_location="cpu").to(torch.bfloat16) for i in ids],dim=3)
train=[i for i,q in enumerate(queries) if not q[3]];test=[i for i,q in enumerate(queries) if q[3]];ranks=[0,1,2,4,8];rows=[]
for source,B in (("raw",base("kv_cache")),("preprocess",base("preprocess_kv_cache_global_topk10_bge"))):
 delta=[x-B for x in full]
 for layer in range(64):
  X=torch.stack([delta[i][layer].float().flatten() for i in train]);mean=X.mean(0);C=X-mean;G=C@C.T;ev,Q=torch.linalg.eigh(G);order=torch.argsort(ev,descending=True);ev=ev[order].clamp_min(1e-20);Q=Q[:,order];V=(C.T@Q[:,:8])/torch.sqrt(ev[:8]).unsqueeze(0)
  for i in test:
   d=delta[i][layer].float().flatten();y=d-mean;yn=float(y.square().sum());dn=float(d.square().sum())
   coeff=y@V
   for rank in ranks:
    hat=mean if rank==0 else mean+V[:,:rank]@coeff[:rank];err=float((d-hat).square().sum());cos=float(torch.nn.functional.cosine_similarity(d,hat,dim=0))
    rows.append({"source":source,"layer":layer,"test_query":queries[i][0],"category":queries[i][2],"rank":rank,"relative_l2":(err/max(dn,1e-20))**.5,"cosine":cos,"full_explained_variance":1-err/max(dn,1e-20),"residual_explained_variance":0.0 if yn<1e-20 else 1-float((y-(hat-mean)).square().sum())/yn})
tag="qwen3_32b_ex0_query_prefix_value_heldout_12train4test";f=O/f"{tag}.csv"
with f.open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary=[]
for src in ("raw","preprocess"):
 for rank in ranks:
  x=[r for r in rows if r["source"]==src and r["rank"]==rank];summary.append({"source":src,"rank":rank,"relative_l2":sum(r["relative_l2"] for r in x)/len(x),"cosine":sum(r["cosine"] for r in x)/len(x),"full_explained_variance":sum(r["full_explained_variance"] for r in x)/len(x),"residual_explained_variance":sum(r["residual_explained_variance"] for r in x)/len(x)})
(O/f"{tag}_summary.json").write_text(json.dumps({"fixed_prefix_tokens":fixed,"train":[queries[i][0] for i in train],"test":[queries[i][0] for i in test],"summary":summary},indent=2));print(json.dumps(summary,indent=2))
