#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"ktransformers"))
from transformers import AutoTokenizer
from ktransformers.util.utils import _exact_match_score, compute_f1
JUDGE_PATH=Path(__file__).with_name("rejudge_setup_v2_glm_clean.py")
import importlib.util
spec=importlib.util.spec_from_file_location("sparse_glm",JUDGE_PATH)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class InlineJudge:
    """Judge one completed sample and append metrics before the next sample runs."""
    def __init__(self, output: str, model: str):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.seen: set[tuple[str, str, str]] = set()
        if self.output.exists():
            with self.output.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    self.seen.add((row.get("Question", ""), row.get("Real Answer", ""), row.get("Pred Answer", "")))
        else:
            with self.output.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=["index", "Question", "Real Answer", "Pred Answer", "em", "f1", "glm_correct", "glm_reason", "glm_raw"]).writeheader()

    def __call__(self, *, index: int, question: str, gold: str, prediction: str, em: float) -> None:
        key = (question, gold, prediction)
        if key in self.seen:
            print(f"[inline-glm] skip cached index={index}", flush=True)
            return
        pred_clean = mod.clean_pred(prediction)
        f1 = float(compute_f1(prediction, gold, self.tokenizer))
        ok, reason, raw = mod.call_glm(mod.make_prompt(question, gold, pred_clean))
        fields = ["index", "Question", "Real Answer", "Pred Answer", "em", "f1", "glm_correct", "glm_reason", "glm_raw"]
        with self.output.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({
                "index": index, "Question": question, "Real Answer": gold, "Pred Answer": prediction,
                "em": em, "f1": f1, "glm_correct": ok, "glm_reason": reason, "glm_raw": raw,
            })
        self.seen.add(key)
        print(f"[inline-glm] index={index} em={em:.0f} f1={f1:.6f} glm={ok}", flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--workers",type=int,default=int(os.environ.get("GLM_REJUDGE_WORKERS","8")))
    p.add_argument("--model",default="/mnt/qjhs-sh-lab-01/models/Qwen3-32B")
    a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    with Path(a.csv).open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    def local(row):
        pred=row.get("Pred Answer") or row.get("Answer") or row.get("Predicted") or ""
        gold=row.get("Real Answer") or row.get("Ground Truth") or row.get("Gold") or ""
        return float(_exact_match_score(pred,gold)),float(compute_f1(pred,gold,tok))
    local_scores=[local(r) for r in rows]
    def judge(item):
        i,row=item
        pred=row.get("Pred Answer") or row.get("Answer") or row.get("Predicted") or ""
        gold=row.get("Real Answer") or row.get("Ground Truth") or row.get("Gold") or ""
        q=row.get("Question") or row.get("Main Question") or row.get("Sub Question") or ""
        ok,reason,raw=mod.call_glm(mod.make_prompt(q,gold,mod.clean_pred(pred)))
        return i,ok,reason,raw
    judged={}
    with ThreadPoolExecutor(max_workers=max(1,a.workers)) as ex:
        futures=[ex.submit(judge,x) for x in enumerate(rows)]
        for f in as_completed(futures):
            i,ok,reason,raw=f.result(); judged[i]=(ok,reason,raw)
    fields=list(rows[0].keys()) if rows else []
    fields += ["em","f1","glm_correct","glm_reason"]
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i,row in enumerate(rows):
            x=dict(row); x["em"]=local_scores[i][0]; x["f1"]=local_scores[i][1]; x["glm_correct"]=judged[i][0]; x["glm_reason"]=judged[i][1]; w.writerow(x)
    print(f"[auto-glm] rows={len(rows)} em={sum(x[0] for x in local_scores)/len(rows) if rows else 0:.6f} f1={sum(x[1] for x in local_scores)/len(rows) if rows else 0:.6f} glm={sum(1 for x in judged.values() if x[0])}/{len(rows)}",flush=True)
if __name__=="__main__": main()
