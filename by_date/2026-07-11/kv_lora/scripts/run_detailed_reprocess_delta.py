import sys,json,csv
from pathlib import Path
import torch
sys.path.insert(0,"/raid/home/hming/FusionRAG-pca-analysis")
from transformers import AutoConfig,AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import load_kv_and_generate
from test_fusionrag_reflect_preprocess_exp import PreprocessScope,RecallMethod,load_model,prepare_reflect_data
R=Path("/raid/home/hming/FusionRAG-pca-analysis"); M="/mnt/qjhs-sh-lab-01/models/Qwen3-32B"; dev="cuda:0"; out=R/"MOTIVATION_EXPERIMENTS/kv_lora/results"
def put(p,x):
 with open(p,"w",newline="",encoding="utf8") as f:
  w=csv.DictWriter(f,fieldnames=x[0]);w.writeheader();w.writerows(x)
tok=AutoTokenizer.from_pretrained(M,trust_remote_code=True); cfg=AutoConfig.from_pretrained(M,trust_remote_code=True);cfg._attn_implementation="sdpa";model,_=load_model("qwen3",M,cfg,dev,False);model.eval()
qs,sy,_,_=prepare_reflect_data(str(R/"data/result_reflect.json"),tok,"/mnt/qjhs-sh-lab-01/models/bge-m3","qwen3",topk=10,max_main_questions=1,preprocess=False,recall_method=RecallMethod.BGE,preprocess_scope=PreprocessScope.GLOBAL)
d=qs[0]; sub=d["sub_questions"][0]; ids=sub["chunk_ids"]; docs=[d["doc_tensors"][i-1] for i in ids]; q=torch.tensor(tok.encode("<|im_end|>\n<|im_start|>user\nQuestion: "+sub["query"]+"\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ",add_special_tokens=False)); ps=[sy]+docs+[q]; start=len(sy); n=sum(map(len,docs)); root=Path("/raid/home/hming/fusionrag-reflect-qwen3-smoke-cache-preprocess/Qwen3-32B/musique")
S=[];L=[];H=[];T=[]
for source,path in (("raw",root/"kv_cache"),("preprocess",root/"preprocess_kv_cache_global_topk10_bge")):
 c=StaticCache(config=model.config,max_batch_size=1,max_cache_len=8192,device=dev,dtype=model.dtype,passage_len=8192,model=model); st={}
 def cb(phase,past,selected,*_):
  if phase=="before_reprocess":
   for k,x in (("k",past.key_cache),("v",past.value_cache)):st["b"+k]=torch.stack([z[:,:,start:start+n,:].detach().cpu().float() for z in x])
  else:
   for k,x in (("k",past.key_cache),("v",past.value_cache)):st["a"+k]=torch.stack([z[:,:,start:start+n,:].detach().cpu().float() for z in x])
   st["ix"]=sorted(int(z)-start for z in selected if start<=int(z)<start+n)
 load_kv_and_generate(model,tok,c,ps,str(path),0,max_new_tokens=1,revert_rope=True,reprocess_method="FusionRAG",rate=.15,preprocess=source=="preprocess",device=dev,chunk_ids=[0]+ids,kv_snapshot_callback=cb)
 ix=st["ix"]
 for k in "kv":
  D=st["a"+k]-st["b"+k];B=st["b"+k]; ds=D[:,:,:,ix,:];bs=B[:,:,:,ix,:];S.append(dict(source=source,kind=k,selected=len(ix),total=n,relative_l2=float(ds.norm()/bs.norm()),delta_l2=float(ds.norm())))
  tn=torch.sqrt(D.square().sum(dim=(0,1,2,4)));bn=torch.sqrt(B.square().sum(dim=(0,1,2,4)));order=torch.argsort(tn,descending=True)
  for i in ix:T.append(dict(source=source,kind=k,token=i,relative_l2=float(tn[i]/bn[i].clamp_min(1e-12)),energy=float(tn[i]**2/D.square().sum()),rank=int((order==i).nonzero()[0])))
  for l in range(64):
   x=D[l,:,:,ix,:];y=B[l,:,:,ix,:];mat=x[0].permute(1,0,2).reshape(len(ix),-1);s=torch.linalg.svdvals(mat);e=s*s;cum=torch.cumsum(e,0)/e.sum().clamp_min(1e-12);p=e/e.sum().clamp_min(1e-12);L.append(dict(source=source,kind=k,layer=l,relative_l2=float(x.norm()/y.norm().clamp_min(1e-12)),stable_rank=float(e.sum()/e.max().clamp_min(1e-12)),effective_rank=float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum())),rank90=int((cum<.9).sum()+1),rank95=int((cum<.95).sum()+1),rank99=int((cum<.99).sum()+1),nearzero=float((x.abs()<.01*torch.sqrt(x.square().mean())).float().mean())))
   for h in range(8):
    a=x[:,:,h,:];b=y[:,:,h,:];H.append(dict(source=source,kind=k,layer=l,head=h,relative_l2=float(a.norm()/b.norm().clamp_min(1e-12)),energy=float(a.square().sum()/x.square().sum().clamp_min(1e-12))))
tag="qwen3_32b_ex0_fullchunks_fusionrag_rate0p15_detailed";put(out/f"reprocess_summary_{tag}.csv",S);put(out/f"reprocess_layers_{tag}.csv",L);put(out/f"reprocess_heads_{tag}.csv",H);put(out/f"reprocess_tokens_{tag}.csv",T);print(json.dumps(S))
