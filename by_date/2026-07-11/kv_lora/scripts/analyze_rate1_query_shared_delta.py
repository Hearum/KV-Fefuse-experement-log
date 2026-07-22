#!/usr/bin/env python3
import csv, json, sys
from pathlib import Path
import torch
sys.path.insert(0, "/raid/home/hming/FusionRAG-pca-analysis")
from transformers import AutoConfig, AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import load_kv_and_generate
from test_fusionrag_reflect_preprocess_exp import PreprocessScope, RecallMethod, load_model, prepare_reflect_data
R=Path("/raid/home/hming/FusionRAG-pca-analysis"); OUT=R/"MOTIVATION_EXPERIMENTS/kv_lora/results"; M="/mnt/qjhs-sh-lab-01/models/Qwen3-32B"; DEV="cuda:0"
tok=AutoTokenizer.from_pretrained(M,trust_remote_code=True); cfg=AutoConfig.from_pretrained(M,trust_remote_code=True); cfg._attn_implementation="sdpa"; model,_=load_model("qwen3",M,cfg,DEV,False); model.eval()
questions,system,_,_=prepare_reflect_data(str(R/"data/result_reflect.json"),tok,"/mnt/qjhs-sh-lab-01/models/bge-m3","qwen3",topk=10,max_main_questions=1,preprocess=False,recall_method=RecallMethod.BGE,preprocess_scope=PreprocessScope.GLOBAL)
data=questions[0]; sub=data["sub_questions"][0]; chunk_ids=sub["chunk_ids"]; docs=[data["doc_tensors"][i-1] for i in chunk_ids]; doc_start=len(system); doc_len=sum(map(len,docs))
queries=[
 ("original",sub["query"]),
 ("short_paraphrase","Which network includes National Cycle Route 57?"),
 ("formal_paraphrase","Identify the larger cycling network of which National Cycle Route 57 is a constituent route."),
 ("long_reasoning",sub["query"]+" Use all relevant evidence and explain the reasoning step by step before giving the answer."),
 ("related_different","Where does National Cycle Route 57 begin and end?"),
 ("negative_weather","What will the weather be tomorrow?"),
 ("negative_france","What is the capital of France?"),
 ("negative_math","Compute 17 multiplied by 23."),
]
def question_tensor(text):
 s="<|im_end|>\n<|im_start|>user\nQuestion: "+text+"\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: "
 return torch.tensor(tok.encode(s,add_special_tokens=False),dtype=torch.long)
roots={"raw":Path("/raid/home/hming/fusionrag-reflect-qwen3-smoke-cache-preprocess/Qwen3-32B/musique/kv_cache"),"preprocess":Path("/raid/home/hming/fusionrag-reflect-qwen3-smoke-cache-preprocess/Qwen3-32B/musique/preprocess_kv_cache_global_topk10_bge")}
rows=[]; manifest=[]
for source,path in roots.items():
 refs={}; before_ref={}
 for qi,(label,text) in enumerate(queries):
  cache=StaticCache(config=model.config,max_batch_size=1,max_cache_len=8192,device=DEV,dtype=model.dtype,passage_len=8192,model=model); state={}
  def snap(phase,past,selected,*_):
   prefix="before" if phase=="before_reprocess" else "after"
   for kind,caches in (("k",past.key_cache),("v",past.value_cache)):
    state[prefix+kind]=torch.stack([x[:,:,doc_start:doc_start+doc_len,:].detach().cpu().float() for x in caches])
   if selected is not None: state["selected"]=[int(x) for x in selected]
  passages=[system]+docs+[question_tensor(text)]
  _,_,info=load_kv_and_generate(model,tok,cache,passages,str(path),0,max_new_tokens=1,revert_rope=True,reprocess_method="FusionRAG",rate=1.0,preprocess=(source=="preprocess"),device=DEV,chunk_ids=[0]+chunk_ids,kv_snapshot_callback=snap)
  for kind in ("k","v"):
   before=state["before"+kind]; delta=state["after"+kind]-before
   if qi==0: refs[kind]=delta.clone(); before_ref[kind]=before.clone()
   ref=refs[kind]; bref=before_ref[kind]
   for layer in range(delta.shape[0]):
    d=delta[layer].flatten(); r=ref[layer].flatten(); bdiff=(before[layer]-bref[layer]).flatten()
    rows.append({"source":source,"query_index":qi,"query_label":label,"kind":kind,"layer":layer,"delta_l2":float(d.norm()),"delta_vs_original_relative_l2":float((d-r).norm()/r.norm().clamp_min(1e-12)),"delta_vs_original_max_abs":float((d-r).abs().max()),"delta_vs_original_cosine":float(torch.nn.functional.cosine_similarity(d,r,dim=0)),"before_vs_original_max_abs":float(bdiff.abs().max())})
  manifest.append({"source":source,"query_index":qi,"query_label":label,"query":text,"query_tokens":len(passages[-1]),"selected_doc_tokens":sum(doc_start<=x<doc_start+doc_len for x in state["selected"]),"profile":info})
  del state,cache
field=list(rows[0]); tag="qwen3_32b_ex0_rate1_query_shared_delta"
with (OUT/f"{tag}_layers.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=field);w.writeheader();w.writerows(rows)
(OUT/f"{tag}_manifest.json").write_text(json.dumps({"doc_chunks":len(chunk_ids),"doc_tokens":doc_len,"queries":queries,"runs":manifest},ensure_ascii=False,indent=2))
summary=[]
for source in roots:
 for kind in ("k","v"):
  x=[r for r in rows if r["source"]==source and r["kind"]==kind and r["query_index"]>0]
  summary.append({"source":source,"kind":kind,"max_query_delta_relative_l2":max(r["delta_vs_original_relative_l2"] for r in x),"max_query_delta_max_abs":max(r["delta_vs_original_max_abs"] for r in x),"min_query_delta_cosine":min(r["delta_vs_original_cosine"] for r in x),"max_before_difference":max(r["before_vs_original_max_abs"] for r in x)})
(OUT/f"{tag}_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
