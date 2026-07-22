#!/usr/bin/env python3
import csv
import datetime as dt
from pathlib import Path
import torch

ROOT = Path('/raid/home/hming/FusionRAG-pca-analysis')
EXP = ROOT / 'MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill'
README = EXP / 'README.md'
CKPT = EXP / 'checkpoints'
START = '<!-- AUTO_VAL_HISTORY_START -->'
END = '<!-- AUTO_VAL_HISTORY_END -->'

HEADERS = [
    'run', 'epoch', 'train_loss',
    'Wiki KL', 'Wiki R@5', 'Wiki J@5', 'Wiki R@10', 'Wiki J@10', 'Wiki R@15', 'Wiki J@15', 'Wiki R@30', 'Wiki J@30',
    'MuSiQue KL', 'MuSiQue R@5', 'MuSiQue J@5', 'MuSiQue R@10', 'MuSiQue J@10', 'MuSiQue R@15', 'MuSiQue J@15', 'MuSiQue R@30', 'MuSiQue J@30',
]
ORDER = [
    'train_loss',
    'wiki_val_kl', 'wiki_val_recall_r0p05', 'wiki_val_jaccard_r0p05', 'wiki_val_recall_r0p1', 'wiki_val_jaccard_r0p1',
    'wiki_val_recall_r0p15', 'wiki_val_jaccard_r0p15', 'wiki_val_recall_r0p3', 'wiki_val_jaccard_r0p3',
    'musique_val_kl', 'musique_val_recall_r0p05', 'musique_val_jaccard_r0p05', 'musique_val_recall_r0p1', 'musique_val_jaccard_r0p1',
    'musique_val_recall_r0p15', 'musique_val_jaccard_r0p15', 'musique_val_recall_r0p3', 'musique_val_jaccard_r0p3',
]

def fmt(v):
    if v is None or v == '':
        return '-'
    try:
        return f'{float(v):.4f}'
    except Exception:
        return str(v)

def read_history_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def read_history_pt(path):
    state = torch.load(path, map_location='cpu')
    return list(state.get('history', []))

def read_rows():
    rows = []
    for run_dir in sorted(CKPT.glob('*')):
        if not run_dir.is_dir():
            continue
        run = run_dir.name
        if run.startswith('smoke_'):
            continue
        csv_path = run_dir / 'history.csv'
        if csv_path.exists():
            source = csv_path
            loader = lambda p=csv_path: read_history_csv(p)
        else:
            candidates = [run_dir / 'training_state_latest.pt'] + sorted(run_dir.glob('training_state_epoch*.pt'), reverse=True)
            source = next((p for p in candidates if p.exists()), None)
            if source is None:
                continue
            loader = lambda p=source: read_history_pt(p)
        try:
            for row in loader():
                rows.append((run, int(float(row.get('epoch', 0))), row, source.name))
        except Exception as exc:
            rows.append((run, -1, {'error': f'{source.name}: {exc}'}, source.name))
    rows.sort(key=lambda item: (item[0], item[1]))
    return rows

def make_section():
    now = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rows = read_rows()
    lines = [START, '## 自动同步：每轮 Validation 数字', '']
    lines.append(f'- 更新时间：`{now}`')
    lines.append('- 说明：该区块从 `history.csv` 或训练中的 `training_state_latest.pt` 自动生成；训练未结束时也能看到已完成 epoch 的 validation。')
    lines.append('- R/J 分别表示 student selector top-rate token 相对 full DraftModel teacher top-rate token 的 Recall/Jaccard；rate=5/10/15/30%。')
    lines.append('')
    if not rows:
        lines += ['当前还没有发现 history。', END]
        return '\n'.join(lines) + '\n'
    lines.append('| ' + ' | '.join(HEADERS) + ' |')
    lines.append('| ' + ' | '.join(['---'] + ['---:'] * (len(HEADERS) - 1)) + ' |')
    for run, epoch, row, _source in rows:
        if 'error' in row:
            vals = [run, str(epoch), 'ERROR: ' + row['error']] + ['-'] * (len(HEADERS) - 3)
        else:
            vals = [run, str(epoch)] + [fmt(row.get(k, '')) for k in ORDER]
        lines.append('| ' + ' | '.join(vals) + ' |')
    lines += ['', '### 当前结论记录', '']
    lines.append('- `epoch=0` 是未微调的原生 layer4 selector baseline。')
    lines.append('- 目前已完成的 50k/500k-e1 显示 WikiText validation 有小幅提升，但 MuSiQue validation 基本持平。')
    lines.append('- 长训进行中时，`native_l4_wikitext500k_e20_resume` 的行来自 `training_state_latest.pt`，不是最终 CSV。')
    lines.append(END)
    return '\n'.join(lines) + '\n'

def main():
    old = README.read_text() if README.exists() else '# Native-Initialized Layer4 Draft Selector Distillation\n\n'
    section = make_section()
    if START in old and END in old:
        before = old.split(START)[0].rstrip()
        after = old.split(END, 1)[1].lstrip('\n')
        new = before + '\n\n' + section
        if after:
            new += '\n' + after
    else:
        new = old.rstrip() + '\n\n' + section
    README.write_text(new)

if __name__ == '__main__':
    main()
