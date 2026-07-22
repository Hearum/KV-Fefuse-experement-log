#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

import pandas as pd


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def tmux_alive(session: str) -> bool:
    return run(["tmux", "has-session", "-t", session]).returncode == 0


def as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "correct"}
    return bool(x)


def summarize(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    key_cols = ["Main Question", "Sub Question"]
    unique_df = df.drop_duplicates(key_cols, keep="last")
    correct = unique_df["Correct"].map(as_bool)
    main_ok = unique_df.assign(_correct=correct).groupby("Main Question")["_correct"].all()
    out = {
        "csv": str(csv_path),
        "rows": int(len(df)),
        "unique_rows": int(len(unique_df)),
        "duplicate_rows": int(len(df) - len(unique_df)),
        "sub_correct": int(correct.sum()),
        "sub_total": int(len(unique_df)),
        "sub_acc": float(correct.mean()) if len(unique_df) else 0.0,
        "main_correct": int(main_ok.sum()),
        "main_total": int(len(main_ok)),
        "main_acc": float(main_ok.mean()) if len(main_ok) else 0.0,
        "avg_f1": float(pd.to_numeric(unique_df["F1"], errors="coerce").mean()),
        "avg_em": float(pd.to_numeric(unique_df["EM"], errors="coerce").mean()),
    }
    return out


def append_readme(readme: Path, name: str, result: dict, log_path: str) -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "",
        f"### 自动记录：235B current rerun `{name}`",
        "",
        f"- 完成时间：{now}",
        f"- CSV：`{result['csv']}`",
        f"- run.log：`{log_path}`",
        f"- 原始 CSV 行数：{result['rows']}",
        f"- 唯一 `(Main Question, Sub Question)`：{result['unique_rows']}",
        f"- 重复行：{result['duplicate_rows']}",
        f"- Sub accuracy：{result['sub_correct']}/{result['sub_total']} ({result['sub_acc']:.2%})",
        f"- Main accuracy：{result['main_correct']}/{result['main_total']} ({result['main_acc']:.2%})",
        f"- Avg F1：{result['avg_f1']:.4f}",
        f"- Avg EM：{result['avg_em']:.4f}",
        "",
        "```json",
        json.dumps(result, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    readme.parent.mkdir(parents=True, exist_ok=True)
    with readme.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--readme", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    while tmux_alive(args.session):
        time.sleep(args.interval)
    result = summarize(Path(args.csv))
    append_readme(Path(args.readme), args.name, result, args.log)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
