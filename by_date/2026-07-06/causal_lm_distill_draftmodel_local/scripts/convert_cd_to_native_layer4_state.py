#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cd-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    state = torch.load(args.cd_checkpoint, map_location="cpu")
    student_state = state.get("student_state", state)
    embed = {}
    layers = {}
    for key, value in student_state.items():
        if key.startswith("model.embed_tokens."):
            embed[key[len("model.embed_tokens.") :]] = value
        elif key.startswith("model.layers."):
            layers[key[len("model.layers.") :]] = value

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embed_tokens": embed,
            "layers": layers,
            "epoch": 0,
            "history": [],
            "source_cd_checkpoint": str(args.cd_checkpoint),
            "source_cd_step": int(state.get("step", -1)),
            "source_cd_epoch": int(state.get("epoch", -1)),
        },
        out,
    )
    print(f"wrote {out} with {len(embed)} embed tensors and {len(layers)} layer tensors")


if __name__ == "__main__":
    main()
