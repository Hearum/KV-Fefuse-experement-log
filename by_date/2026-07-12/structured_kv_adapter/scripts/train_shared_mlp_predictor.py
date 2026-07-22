#!/usr/bin/env python3
"""Train a shared nonlinear prefix-conditioned rank-64 Delta predictor."""

import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
DATA_DIRS = [
    ROOT / "MOTIVATION_EXPERIMENTS/kv_lora/results/formal_preprocess_residual_50",
    ROOT / "MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/formal_residual_50_100",
]
CACHE = Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique/preprocess_kv_cache_global_topk10_bge")
RESULTS = ROOT / "MOTIVATION_EXPERIMENTS/structured_kv_adapter/results"
L, H, F, R = 64, 8, 128, 64
SEED = 20260712


def split(example):
    return "train" if example < 70 else "validation" if example < 85 else "test"


def cache_tensor(example, chunk, suffix):
    return torch.load(CACHE / f"{example}_{chunk}_{suffix}.pt", map_location="cpu", weights_only=True)[:, 0].float()


class Predictor(nn.Module):
    def __init__(self, prefix_dim=32):
        super().__init__()
        self.own = nn.Sequential(nn.LayerNorm(2 * F), nn.Linear(2 * F, 128), nn.GELU())
        self.prefix = nn.Sequential(nn.LayerNorm(4 * F), nn.Linear(4 * F, prefix_dim), nn.GELU())
        self.layer_embedding = nn.Embedding(L, 16)
        self.head_embedding = nn.Embedding(H, 8)
        self.trunk = nn.Sequential(
            nn.Linear(128 + prefix_dim + 16 + 8 + 4, 256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 2 * R),
        )

    def forward(self, own, prefix, position, layer, head):
        features = torch.cat(
            [self.own(own), self.prefix(prefix), position,
             self.layer_embedding(layer), self.head_embedding(head)], dim=-1
        )
        return self.trunk(features)


def files():
    return sorted(path for directory in DATA_DIRS for path in directory.glob("ex*_sub*.pt"))


def fit_basis(paths):
    moments = {
        kind: {"sum": torch.zeros(L, H, F, device="cuda"),
               "yty": torch.zeros(L, H, F, F, device="cuda"), "n": 0}
        for kind in ("k", "v")
    }
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=False)
        if split(record["example"]) != "train":
            continue
        for item in record["items"]:
            for kind in ("k", "v"):
                y = item[f"delta_{kind}"][:, 0].float().cuda()
                if y.shape[2] > 32:
                    idx = torch.linspace(0, y.shape[2] - 1, 32, device="cuda").long()
                    y = y[:, :, idx]
                moments[kind]["sum"] += y.sum(2)
                moments[kind]["yty"] += torch.einsum("lhti,lhtj->lhij", y, y)
                moments[kind]["n"] += y.shape[2]
    model = {}
    for kind, stat in moments.items():
        mean = stat["sum"] / stat["n"]
        covariance = stat["yty"] - stat["n"] * torch.einsum("lhi,lhj->lhij", mean, mean)
        _, vectors = torch.linalg.eigh(covariance)
        model[kind] = {"mean": mean.cpu(), "basis": vectors[..., -R:].cpu()}
    return model


def sampled_dataset(paths, wanted_split, basis_model, samples_per_doc=64):
    own_rows, prefix_rows, position_rows, layer_rows, head_rows, target_rows = [], [], [], [], [], []
    examples = set()
    coefficient_squares = torch.zeros(2, L, H, R)
    coefficient_counts = torch.zeros(L, H, 1)
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=False)
        if split(record["example"]) != wanted_split:
            continue
        examples.add(record["example"])
        prefix_k_sum = prefix_v_sum = None
        prefix_tokens = 0
        last_k = last_v = None
        for rank, item in enumerate(record["items"]):
            k = cache_tensor(record["example"], item["chunk"], "key")
            v = cache_tensor(record["example"], item["chunk"], "value")
            tokens = k.shape[2]
            pk = prefix_k_sum / prefix_tokens if prefix_tokens else torch.zeros(L, H, F)
            pv = prefix_v_sum / prefix_tokens if prefix_tokens else torch.zeros(L, H, F)
            lk = last_k if last_k is not None else torch.zeros(L, H, F)
            lv = last_v if last_v is not None else torch.zeros(L, H, F)
            generator = torch.Generator().manual_seed(SEED + record["example"] * 10000 + record["subquery"] * 100 + item["chunk"])
            layers = torch.randint(L, (samples_per_doc,), generator=generator)
            heads = torch.randint(H, (samples_per_doc,), generator=generator)
            token_ids = torch.randint(tokens, (samples_per_doc,), generator=generator)
            own_rows.append(torch.cat([k[layers, heads, token_ids], v[layers, heads, token_ids]], -1).half())
            prefix_rows.append(torch.cat([pk[layers, heads], pv[layers, heads], lk[layers, heads], lv[layers, heads]], -1).half())
            position_rows.append(torch.stack([
                token_ids.float() / max(tokens - 1, 1),
                torch.full((samples_per_doc,), rank / 10.0),
                torch.full((samples_per_doc,), prefix_tokens / 8192.0),
                torch.full((samples_per_doc,), tokens / 512.0),
            ], -1).half())
            layer_rows.append(layers.to(torch.int16)); head_rows.append(heads.to(torch.int8))
            coefficients = []
            for kind in ("k", "v"):
                y = item[f"delta_{kind}"][:, 0].float()
                centered = y[layers, heads, token_ids] - basis_model[kind]["mean"][layers, heads]
                coefficients.append(torch.einsum("nf,nfr->nr", centered, basis_model[kind]["basis"][layers, heads]))
            target = torch.cat(coefficients, -1)
            target_rows.append(target.half())
            for ki, coefficient in enumerate(coefficients):
                coefficient_squares[ki].index_put_((layers, heads), coefficient.square(), accumulate=True)
            coefficient_counts.index_put_((layers, heads), torch.ones(samples_per_doc, 1), accumulate=True)
            prefix_k_sum = k.sum(2) if prefix_k_sum is None else prefix_k_sum + k.sum(2)
            prefix_v_sum = v.sum(2) if prefix_v_sum is None else prefix_v_sum + v.sum(2)
            prefix_tokens += tokens
            last_k, last_v = k.mean(2), v.mean(2)
    tensors = (torch.cat(own_rows), torch.cat(prefix_rows), torch.cat(position_rows),
               torch.cat(layer_rows).long(), torch.cat(head_rows).long(), torch.cat(target_rows))
    std = (coefficient_squares / coefficient_counts.clamp_min(1)).sqrt().clamp_min(1e-4)
    return TensorDataset(*tensors), std, sorted(examples)


@torch.no_grad()
def loss_on(model, loader, coefficient_std):
    model.eval(); total = count = 0
    for own, prefix, position, layer, head, target in loader:
        own, prefix, position = own.cuda().float(), prefix.cuda().float(), position.cuda().float()
        layer, head, target = layer.cuda(), head.cuda(), target.cuda().float()
        scale = torch.cat([coefficient_std[0, layer, head], coefficient_std[1, layer, head]], -1)
        loss = (model(own, prefix, position, layer, head) - target / scale).square().mean()
        total += float(loss) * len(target); count += len(target)
    return total / count


@torch.no_grad()
def strict_test(model, paths, basis_model, coefficient_std):
    model.eval()
    methods = ("mean", "mlp", "oracle_rank64")
    totals = {(kind, method): {key: 0.0 for key in ("delta", "error", "base", "dot", "pred")}
              for kind in ("k", "v") for method in methods}
    examples = set(); documents = 0
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=False)
        if split(record["example"]) != "test": continue
        examples.add(record["example"]); prefix_k_sum = prefix_v_sum = None; prefix_tokens = 0; last_k = last_v = None
        for rank, item in enumerate(record["items"]):
            documents += 1
            k = cache_tensor(record["example"], item["chunk"], "key"); v = cache_tensor(record["example"], item["chunk"], "value"); tokens = k.shape[2]
            pk = prefix_k_sum / prefix_tokens if prefix_tokens else torch.zeros(L,H,F); pv = prefix_v_sum / prefix_tokens if prefix_tokens else torch.zeros(L,H,F)
            lk = last_k if last_k is not None else torch.zeros(L,H,F); lv = last_v if last_v is not None else torch.zeros(L,H,F)
            for start in range(0, tokens, 16):
                stop=min(tokens,start+16); t=stop-start
                own=torch.cat([k[:,:,start:stop],v[:,:,start:stop]],-1).reshape(-1,2*F).cuda()
                prefix=torch.cat([pk,pv,lk,lv],-1).unsqueeze(2).expand(L,H,t,4*F).reshape(-1,4*F).cuda()
                token_pos=torch.arange(start,stop).float()/max(tokens-1,1)
                pos=torch.stack([token_pos,torch.full((t,),rank/10),torch.full((t,),prefix_tokens/8192),torch.full((t,),tokens/512)],-1)
                pos=pos.view(1,1,t,4).expand(L,H,t,4).reshape(-1,4).cuda()
                layers=torch.arange(L).view(L,1,1).expand(L,H,t).reshape(-1).cuda(); heads=torch.arange(H).view(1,H,1).expand(L,H,t).reshape(-1).cuda()
                normalized=model(own,prefix,pos,layers,heads)
                scale=torch.cat([coefficient_std[0,layers,heads],coefficient_std[1,layers,heads]],-1)
                coeff=(normalized*scale).reshape(L,H,t,2*R).cpu()
                for ki,(kind,base) in enumerate((("k",k),("v",v))):
                    y=item[f"delta_{kind}"][:,0,:,start:stop].float(); mean=basis_model[kind]["mean"].unsqueeze(2); basis=basis_model[kind]["basis"]
                    centered=y-mean; mlp=mean+torch.einsum("lhtr,lhfr->lhtf",coeff[...,ki*R:(ki+1)*R],basis)
                    oracle=mean+torch.einsum("lhtf,lhfr,lhgr->lhtg",centered,basis,basis)
                    for method,pred in (("mean",mean.expand_as(y)),("mlp",mlp),("oracle_rank64",oracle)):
                        total=totals[(kind,method)]; total["delta"]+=float(y.square().sum()); total["error"]+=float((y-pred).square().sum()); total["base"]+=float(base[:,:,start:stop].square().sum()); total["dot"]+=float((y*pred).sum()); total["pred"]+=float(pred.square().sum())
            prefix_k_sum=k.sum(2) if prefix_k_sum is None else prefix_k_sum+k.sum(2); prefix_v_sum=v.sum(2) if prefix_v_sum is None else prefix_v_sum+v.sum(2); prefix_tokens+=tokens; last_k,last_v=k.mean(2),v.mean(2)
    rows=[]
    for (kind,method),total in totals.items():
        remaining=math.sqrt(total["error"]/total["delta"]); rows.append({"kind":kind,"method":method,"remaining_delta":remaining,"explained_delta_energy":1-remaining**2,"original_gap":math.sqrt(total["delta"]/total["base"]),"final_kv_error":math.sqrt(total["error"]/total["base"]),"delta_cosine":total["dot"]/math.sqrt(total["delta"]*total["pred"])})
    return rows,sorted(examples),documents


def main():
    random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    paths=files(); basis_model=fit_basis(paths)
    train_data,train_std,train_examples=sampled_dataset(paths,"train",basis_model,64)
    val_data,_,val_examples=sampled_dataset(paths,"validation",basis_model,64)
    coefficient_std=train_std.cuda()
    train_loader=DataLoader(train_data,batch_size=2048,shuffle=True,num_workers=0,pin_memory=True)
    val_loader=DataLoader(val_data,batch_size=4096,shuffle=False,num_workers=0,pin_memory=True)
    model=Predictor(32).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    best=None; history=[]; patience=0
    for epoch in range(1,41):
        model.train(); total=count=0
        for own,prefix,position,layer,head,target in train_loader:
            own,prefix,position=own.cuda().float(),prefix.cuda().float(),position.cuda().float(); layer,head,target=layer.cuda(),head.cuda(),target.cuda().float()
            scale=torch.cat([coefficient_std[0,layer,head],coefficient_std[1,layer,head]],-1)
            loss=(model(own,prefix,position,layer,head)-target/scale).square().mean(); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); total+=float(loss)*len(target); count+=len(target)
        val=loss_on(model,val_loader,coefficient_std); row={"epoch":epoch,"train_loss":total/count,"validation_loss":val}; history.append(row); print(json.dumps(row),flush=True)
        if best is None or val<best[0]: best=(val,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}); patience=0
        else: patience+=1
        if patience>=6: break
    model.load_state_dict(best[2]); rows,test_examples,documents=strict_test(model,paths,basis_model,coefficient_std)
    result={"architecture":"shared MLP, prefix bottleneck32, rank64 K/V coefficients","parameters":sum(p.numel() for p in model.parameters()),"train_examples":train_examples,"validation_examples":val_examples,"test_examples":test_examples,"train_samples":len(train_data),"validation_samples":len(val_data),"best_epoch":best[1],"history":history,"test_document_instances":documents,"rows":rows}
    torch.save({"state_dict":best[2],"coefficient_std":train_std.half(),"basis":basis_model,"config":{"prefix_dim":32,"rank":64}},RESULTS/"shared_mlp_predictor.pt")
    (RESULTS/"shared_mlp_predictor.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
