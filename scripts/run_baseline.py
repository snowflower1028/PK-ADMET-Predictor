"""베이스라인 실행.

예시
  python scripts/run_baseline.py --tasks Caco2_Wang --models rf --features desc --seeds 1
  python scripts/run_baseline.py                       # 세 과제 x 2모델 x 3특징 x 5seed
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import run_task  # noqa: E402
from src.config import RESULTS_DIR, SEEDS, TASKS  # noqa: E402
from src.data import get_group  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--models", nargs="+", default=["rf", "lgbm"])
    parser.add_argument("--features", nargs="+", default=["desc", "fp", "desc+fp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--out", default=str(RESULTS_DIR / "baseline.csv"))
    args = parser.parse_args()

    group = get_group()
    rows = []
    for task in args.tasks:
        metric = TASKS[task]["metric"]
        for model in args.models:
            for feat in args.features:
                started = time.time()
                mean, std = run_task(group, task, model, feat, args.seeds)
                elapsed = time.time() - started
                print(
                    f"{task:24} {model:5} {feat:8} "
                    f"{metric}={mean:.4f} +/- {std:.4f}  ({elapsed:.0f}s)"
                )
                rows.append({
                    "task": task, "model": model, "features": feat,
                    "metric": metric, "mean": mean, "std": std,
                    "n_seeds": len(args.seeds), "seconds": round(elapsed, 1),
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
