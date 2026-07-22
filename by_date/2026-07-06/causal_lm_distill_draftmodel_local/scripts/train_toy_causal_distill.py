#!/usr/bin/env python3
import argparse, json, math, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyCausalBlock(nn.Module):
    def __init__(self, hidden, heads, ff_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden*ff_mult), nn.GELU(), nn.Linear(hidden*ff_mult, hidden))
        self.last_attn = None
    def forward(self, x, causal_mask):
        y, w = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=causal_mask, need_weights=True, average_attn_weights=False)
        self.last_attn = w.detach()
        x = x + y
        x = x + self.ff(self.ln2(x))
        return x

class ToyCausalLM(nn.Module):
    def __init__(self, vocab, hidden, layers, heads, max_len=256):
        super().__init__()
        self.token = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(max_len, hidden)
        self.blocks = nn.ModuleList([ToyCausalBlock(hidden, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
    def forward(self, ids):
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0)
        x = self.token(ids) + self.pos(pos)
        causal_mask = torch.triu(torch.ones(t, t, device=ids.device, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.head(self.ln(x))

def make_data(n, seq_len, vocab):
    g = torch.Generator().manual_seed(7)
    data = []
    for i in range(n):
        base = torch.randint(10, vocab, (seq_len,), generator=g)
        # Inject a simple repeated motif so teacher has non-random structure.
        base[::11] = 3
        base[5::17] = base[1::17][:base[5::17].numel()]
        data.append(base)
    return torch.stack(data)

def kl_loss(student_logits, teacher_logits, temperature):
    s = F.log_softmax(student_logits[:, :-1] / temperature, dim=-1)
    t = F.softmax(teacher_logits[:, :-1] / temperature, dim=-1)
    return F.kl_div(s, t, reduction='batchmean') * (temperature * temperature)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--seq-len', type=int, default=96)
    ap.add_argument('--vocab', type=int, default=512)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)
    teacher = ToyCausalLM(args.vocab, hidden=192, layers=6, heads=6).to(device).eval()
    student = ToyCausalLM(args.vocab, hidden=128, layers=4, heads=4).to(device)
    data = make_data(max(128, args.batch_size * args.steps), args.seq_len, args.vocab)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4)
    hist=[]
    for step in range(1, args.steps+1):
        idx = torch.randint(0, data.shape[0], (args.batch_size,))
        ids = data[idx].to(device)
        with torch.no_grad():
            tlog = teacher(ids)
        slog = student(ids)
        loss_kl = kl_loss(slog, tlog, 2.0)
        loss_ce = F.cross_entropy(slog[:, :-1].reshape(-1, args.vocab), ids[:, 1:].reshape(-1))
        loss = loss_kl + 0.1 * loss_ce
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        row = {'step': step, 'loss': float(loss), 'kl': float(loss_kl), 'ce': float(loss_ce)}
        hist.append(row)
        print(row, flush=True)
    torch.save({'student': student.state_dict(), 'args': vars(args), 'history': hist}, out/'toy_student.pt')
    # Attention importance smoke: use last block attention from final token to all previous tokens.
    ids = data[:1].to(device)
    with torch.no_grad():
        _ = student(ids)
    attn = student.blocks[-1].last_attn[0].mean(0)[-1].detach().cpu()
    top = torch.topk(attn[:-1], k=min(8, attn.numel()-1)).indices.tolist()
    (out/'attention_importance.json').write_text(json.dumps({'top_indices': top, 'scores': attn[top].tolist()}, indent=2))
    (out/'history.jsonl').write_text('\n'.join(json.dumps(x) for x in hist) + '\n')

if __name__ == '__main__':
    main()
