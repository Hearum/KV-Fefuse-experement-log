#!/usr/bin/env python3
from pathlib import Path
import csv

ROOT = Path('/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_rate1_same_pipeline')
OUT = ROOT / 'same_pipeline_rate1_summary.csv'
DATASETS = {'2wikimqa': 200, 'hotpotqa': 260, 'triviaqa': 270, 'musique': 200}
CSV_NAME = 'rate_1.0_draft_Qwen2.5-3B-Instruct_revert_rope.csv'

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

def summarize_dataset(ds):
    base = ROOT / 'full_rate1_draft_layout' / ds
    rows = []
    done_segs = []
    tracebacks = 0
    correct = total = 0
    f1 = em = 0.0
    csv_files = 0
    for seg in sorted(base.glob('seg_*')):
        if segment_has_traceback(seg):
            tracebacks += 1
        if not segment_done(seg):
            continue
        done_segs.append(seg.name.replace('seg_', '').replace('_', '-'))
        paths = list(seg.glob(f'**/{CSV_NAME}'))
        csv_files += len(paths)
        for p in paths:
            with p.open(newline='', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    total += 1
                    correct += str(r.get('Correct','')).lower() == 'true'
                    try: f1 += float(r.get('F1') or 0)
                    except Exception: pass
                    try: em += float(r.get('EM') or 0)
                    except Exception: pass
    return {
        'dataset': ds,
        'finished_segments': len(done_segs),
        'csv_files': csv_files,
        'expected_raw_samples': DATASETS[ds],
        'finished_rows': total,
        'correct': correct,
        'acc_over_finished_rows': correct / total if total else 0.0,
        'avg_f1_over_finished_rows': f1 / total if total else 0.0,
        'avg_em_over_finished_rows': em / total if total else 0.0,
        'traceback_logs': tracebacks,
        'finished_list': ' '.join(done_segs),
    }

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = [summarize_dataset(ds) for ds in DATASETS]
    micro_correct = sum(r['correct'] for r in rows)
    micro_total = sum(r['finished_rows'] for r in rows)
    micro_f1_num = sum(r['avg_f1_over_finished_rows'] * r['finished_rows'] for r in rows)
    micro_em_num = sum(r['avg_em_over_finished_rows'] * r['finished_rows'] for r in rows)
    rows.append({
        'dataset': 'micro_all_finished_rows',
        'finished_segments': sum(r['finished_segments'] for r in rows),
        'csv_files': sum(r['csv_files'] for r in rows),
        'expected_raw_samples': sum(DATASETS.values()),
        'finished_rows': micro_total,
        'correct': micro_correct,
        'acc_over_finished_rows': micro_correct / micro_total if micro_total else 0.0,
        'avg_f1_over_finished_rows': micro_f1_num / micro_total if micro_total else 0.0,
        'avg_em_over_finished_rows': micro_em_num / micro_total if micro_total else 0.0,
        'traceback_logs': sum(r['traceback_logs'] for r in rows),
        'finished_list': '',
    })
    with OUT.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(OUT)
    with OUT.open(encoding='utf-8') as f:
        print(f.read())

if __name__ == '__main__':
    main()
