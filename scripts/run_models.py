"""2주차 모델 학습 실행.

예시
  # 빠른 점검: 한 과제, MLP, seed 1개
  python scripts/run_models.py --tasks Caco2_Wang --models mlp --seeds 1

  # 전체: 세 과제 x 2모델 x 5seed
  python scripts/run_models.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# cuBLAS는 핸들을 만들 때 이 설정을 읽는다. torch를 import하기 전에
# 세워 두어야 결정적 GEMM이 보장된다. src.utils.set_seed도 같은 값을
# 넣지만, 그쪽은 CUDA가 이미 초기화된 뒤일 수 있다.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR, SEEDS, TASKS  # noqa: E402
from src.data import get_group  # noqa: E402
from src.train import TrainConfig, train_task  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--models", nargs="+",
                        default=["mlp", "gin", "gin+desc"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default=str(RESULTS_DIR / "models.csv"))
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience,
    )

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"gpu   : {torch.cuda.get_device_name(0)}", flush=True)

    group = get_group()
    rows = []
    for task in args.tasks:
        metric = TASKS[task]["metric"]
        for model in args.models:
            started = time.time()
            mean, std, valid_scores, epochs = train_task(
                group, task, model, args.seeds, cfg, device, args.verbose
            )
            elapsed = time.time() - started
            print(
                f"{task:24} {model:5} {metric}={mean:.4f} +/- {std:.4f}  "
                f"({elapsed:.0f}s, epochs {epochs})",
                flush=True,
            )
            rows.append({
                "task": task, "model": model, "metric": metric,
                "mean": round(mean, 4), "std": round(std, 4),
                "n_seeds": len(args.seeds),
                # valid 점수와 멈춘 epoch을 같이 남긴다. test를 보지 않고
                # 모델을 골랐다는 근거이자, 학습이 제대로 됐는지의 단서다.
                "valid_mean": round(float(sum(valid_scores) / len(valid_scores)), 4),
                "best_epochs": "|".join(str(e) for e in epochs),
                "seconds": round(elapsed, 1),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
