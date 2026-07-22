#!/usr/bin/env python3
from pathlib import Path
import csv

ROOT = Path('/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_uniform_a01_rate_sweep')
OUT = ROOT / 'uniform_a01_rate_sweep_progress.csv'
RATES = ['0.3','0.5','0.8','1.0']
DATASETS = {'2wikimqa':200, 'hotpotqa':260, 'triviaqa':270, 'musique':200}

def csv_name(rate):
    return f'rate_{rate}_draft_Qwen2.5-3B-Instruct_revert_rope.csv'

def segment_done(seg):
    log = seg / 'run.log'
    if not log.exists():
        return False
    try:
        return 'FINAL RESULTS' in log.read_text(errors='ignore')
    except Exception:
        return False

def segment_has_traceback(seg):
    log = seg / 'run.log'
    if not log.exists():
        return False
    try:
        return 'Traceback' in log.read_text(errors='ignore')
    except Exception:
        return False

def summarize(rate, ds):
    base = ROOT / f"uniform_alpha0p1_rate{rate.replace('.', 'p')}" / ds
    done = []
    tracebacks = 0
    files = 0
    rows = correct = 0
    f1 = em = 0.0
    for seg in sorted(base.glob('seg_*')):
        if segment_has_traceback(seg):
            tracebacks += 1
        if not segment_done(seg):
            continue
        done.append(seg.name.replace('seg_', '').replace('_','-'))
        for p in seg.glob(f'**/{csv_name(rate)}'):
            files += 1
            with p.open(newline='', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    rows += 1
                    correct += str(r.get('Correct','')).lower() == 'true'
                    try: f1 += float(r.get('F1') or 0)
                    except Exception: pass
                    try: em += float(r.get('EM') or 0)
                    except Exception: pass
    return {
        'rate': rate,
        'dataset': ds,
        'finished_segments': len(done),
        'csv_files': files,
        'expected_raw_samples': DATASETS[ds],
        'finished_rows': rows,
        'correct': correct,
        'acc_over_finished_rows': correct / rows if rows else 0.0,
        'avg_f1_over_finished_rows': f1 / rows if rows else 0.0,
        'avg_em_over_finished_rows': em / rows if rows else 0.0,
        'traceback_logs': tracebacks,
        'finished_list': ' '.join(done),
    }

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = [summarize(rate, ds) for rate in RATES for ds in DATASETS]
    with OUT.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(OUT)
    with OUT.open(encoding='utf-8') as f:
        print(f.read())

if __name__ == '__main__':
    main()
