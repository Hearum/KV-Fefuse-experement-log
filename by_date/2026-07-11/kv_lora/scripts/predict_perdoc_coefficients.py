#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument("--part", required=True)
p.add_argument("--source", required=True)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()
R = Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora")
DD = R / "results/perdoc_context_deltas" / a.part
m = json.loads((DD / "manifest.json").read_text())
CR = Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique") / ("kv_cache" if a.source == "raw" else "preprocess_kv_cache_global_topk10_bge")
paths = sorted(DD.glob(f"context_*_{a.source}.pt"))[:50]
payload = [torch.load(x, map_location="cpu") for x in paths]
ranks = [4, 8]
methods = ["mean"] + [f"oracle_r{r}" for r in ranks] + [f"position_r{r}" for r in ranks] + [f"cached_prefix_r{r}" for r in ranks]

chunk = {}
chunk_len = {}
for i in range(1, 11):
    f = CR / f"0_{i}_value.pt"
    if f.exists():
        x = torch.load(f, map_location="cpu").float()
        chunk[i] = x[:, 0].mean(2).flatten(1)
        chunk_len[i] = x.shape[3]
features, positions = [], []
for x in payload:
    ids = x["meta"]["prefix_ids"]
    if ids:
        lens = torch.tensor([chunk_len[i] for i in ids]).view(-1, 1, 1)
        feat = (torch.stack([chunk[i] for i in ids]) * lens).sum(0) / lens.sum()
    else:
        feat = torch.zeros((64, 1024))
    features.append(feat)
    positions.append([x["meta"]["prefix_tokens"], len(ids)])
features = torch.stack(features)
positions = torch.tensor(positions, dtype=torch.float32)


def ridge_predict(x, y, xt):
    def fit_predict(a0, b0, z0, alpha):
        xm, xs = a0.mean(0), a0.std(0).clamp_min(1e-5)
        aa, zz = (a0 - xm) / xs, (z0 - xm) / xs
        ym, bb = b0.mean(0), b0 - b0.mean(0)
        kernel = aa @ aa.T
        scale = float(torch.trace(kernel) / len(kernel))
        coef = torch.linalg.solve(kernel + alpha * max(scale, 1e-8) * torch.eye(len(kernel), device=kernel.device), bb)
        return zz @ aa.T @ coef + ym
    best = None
    for alpha in [1e-3, 1e-2, .1, 1., 10., 100.]:
        pred = fit_predict(x[:32], y[:32], x[32:40], alpha)
        loss = float((pred - y[32:40]).square().mean())
        if best is None or loss < best[0]:
            best = (loss, alpha)
    return fit_predict(x, y, xt, best[1]), best[1]


acc = {(i, method): {"de": 0., "be": 0., "ee": 0., "dot": 0., "pred": 0.} for i in range(40, 50) for method in methods}
alphas = []
for layer in range(64):
    delta = torch.stack([x["delta_v"][layer, 0].flatten().float() for x in payload]).to(a.device)
    mean = delta[:40].mean(0)
    centered = delta[:40] - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    feat = features[:, layer].to(a.device)
    pos = positions.to(a.device)
    for rank in ranks:
        basis = vh[:rank]
        coef = centered @ basis.T
        cp, alpha = ridge_predict(feat[:40], coef, feat[40:])
        pp, palpha = ridge_predict(pos[:40], coef, pos[40:])
        alphas.append({"layer": layer, "rank": rank, "cached_alpha": alpha, "position_alpha": palpha})
        for j, i in enumerate(range(40, 50)):
            target = delta[i]
            predictions = {
                f"oracle_r{rank}": mean + ((target - mean) @ basis.T) @ basis,
                f"cached_prefix_r{rank}": mean + cp[j] @ basis,
                f"position_r{rank}": mean + pp[j] @ basis,
            }
            if rank == 4:
                predictions["mean"] = mean
            for method, pred in predictions.items():
                error = target - pred
                cell = acc[(i, method)]
                cell["de"] += float(target.square().sum())
                cell["be"] += float(payload[i]["base_v_norm2_layer"][layer])
                cell["ee"] += float(error.square().sum())
                cell["dot"] += float(target @ pred)
                cell["pred"] += float(pred.square().sum())

rows = []
for (i, method), z in acc.items():
    rows.append({"part": a.part, "target": m["target"], "source": a.source, "context_index": i, "method": method,
                 "original_gap": (z["de"] / z["be"]) ** .5, "delta_recovery_error": (z["ee"] / z["de"]) ** .5,
                 "final_kv_error": (z["ee"] / z["be"]) ** .5, "explained_delta_energy": 1 - z["ee"] / z["de"],
                 "cosine": z["dot"] / max((z["de"] * z["pred"]) ** .5, 1e-30)})
out = R / "results" / f"predict_perdoc_coefficients_{a.part}_{a.source}.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
(R / "results" / f"predict_perdoc_coefficients_{a.part}_{a.source}_meta.json").write_text(json.dumps({"feature": "per-layer token-weighted mean cached prefix Value", "split": "0:32 fit, 32:40 alpha validation, refit 0:40, test 40:50", "alphas": alphas}, indent=2) + "\n")
print(out)
