#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys,os,subprocess
from pathlib import Path
ROOT=Path("/raid/home/hming/FusionRAG-pca-analysis")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"ktransformers"))
from transformers import AutoTokenizer
from ktransformers.util.utils import _exact_match_score, compute_f1
MODEL="/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
EXP=ROOT/"MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_2wiki_v2_layer_selective_recompute"

def load_glm_summary(path):
 if not path.exists(): return {}
 out={}
 with path.open(newline="",encoding="utf-8") as f:
  for r in csv.DictReader(f):
   out[r["method"]]=r
 return out

def main():
 p=argparse.ArgumentParser(); p.add_argument("--result-root",default=str(EXP/"results")); p.add_argument("--out",default=str(EXP/"results/summary.csv")); p.add_argument("--with-glm",action="store_true"); p.add_argument("--glm-output-dir",default=str(EXP/"rejudge_glm_clean_full_20260715")); p.add_argument("--glm-workers",default=os.environ.get("GLM_REJUDGE_WORKERS","16")); args=p.parse_args()
 if args.with_glm:
  glm_summary=Path(args.glm_output_dir)/"rejudged_summary.csv"
  if not glm_summary.exists():
   env=os.environ.copy(); env["LAYER_SELECTIVE_REJUDGE_OUT_DIR"]=args.glm_output_dir; env["GLM_REJUDGE_WORKERS"]=str(args.glm_workers)
   subprocess.run([sys.executable, str(EXP/"scripts/rejudge_layer_selective_glm_clean.py")], cwd=str(ROOT), env=env, check=True)
 tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
 glm=load_glm_summary(Path(args.glm_output_dir)/"rejudged_summary.csv") if args.with_glm else {}
 rows=[]
 for cond_dir in sorted(Path(args.result_root).iterdir()):
  if not cond_dir.is_dir(): continue
  items=[]
  for csvp in sorted(cond_dir.glob("online_qk/2wikimqa-v2/rate_0p15/*/csv/reprocess_method_*.csv")):
   with csvp.open(newline="",encoding="utf-8") as f: items.extend(csv.DictReader(f))
  dedup={}
  for r in items: dedup.setdefault(r.get("Question", ""), r)
  em=[]; f1=[]
  for r in dedup.values():
   pred=r.get("Pred Answer") or ""; gold=r.get("Real Answer") or ""
   em.append(float(_exact_match_score(pred,gold))); f1.append(float(compute_f1(pred,gold,tok)))
  n=len(dedup)
  row={"condition":cond_dir.name,"rows":n,"raw_rows":len(items),"complete":n>=200,"em":sum(em)/n if n else 0.0,"f1":sum(f1)/n if n else 0.0}
  g=glm.get(cond_dir.name)
  if g:
   row["glm_correct"]=g.get("glm_correct",""); row["glm_acc"]=g.get("glm_acc",""); row["glm_complete"]=g.get("complete","")
  rows.append(row)
 out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
 fields=["condition","rows","raw_rows","complete","em","f1","glm_correct","glm_acc","glm_complete"] if args.with_glm else ["condition","rows","raw_rows","complete","em","f1"]
 with out.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 out.with_suffix(".json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
