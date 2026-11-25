"""4주차 스태킹 실행: RF/LightGBM 예측값 + 원자 그래프 -> GIN.

  # 전체 (세 과제 x 다섯 조건 x 5 seed)
  python scripts/run_stack.py

  # 빠른 점검
  python scripts/run_stack.py --tasks Caco2_Wang --models gin+stack --seeds 1

실행하는 조건(--models로 고를 수 있다)

  base              1단 기반모델 RF/LightGBM 단독. **train만** 쓴 조건에서의
                    점수다. results/baseline.csv(train+valid 학습)와 다르며,
                    스태킹이 보탬이 됐는지는 이쪽과 비교해야 한다.
  gin+meta          그래프 + 기반모델 예측값 (descriptor 없음)
  gin+stack         그래프 + descriptor + 기반모델 예측값  <- 최종 구조
  gin+stack-infold  gin+stack과 같지만 train 특징을 누출된 방식으로 만든 것.
  gin+stack-plain   gin+stack과 같지만 residual 연결을 끈 것.
  gin+stack-plain-infold  residual을 끄고 누출 특징을 쓴 것.

마지막 네 조건이 2x2 설계다.

                    | train 특징 OOF        | train 특징 누출(infold)
  residual head 켬  | gin+stack             | gin+stack-infold
  residual head 끔  | gin+stack-plain       | gin+stack-plain-infold

이렇게 짠 이유: residual을 켜면 "meta를 과신한다"의 귀결이 "출력 = 기반모델
평균"이고, 그 값은 test에서도 쓸 만한 예측이다. 즉 residual head가 누출의
해악을 가려 버린다. 누출이 실제로 무엇을 망치는지는 head가 신뢰도를
**학습해야 하는** plain 쪽에서 드러난다. 한 축만 돌리면 그 교호작용을
볼 수 없다.

2주차 gin / gin+desc는 results/models.csv에 이미 있으므로 다시 돌리지 않는다.
비교 표는 그 CSV와 이 CSV를 같이 읽어서 만든다.

평가는 1·2주차와 똑같이 **TDC group.evaluate_many()** 가 준 숫자만 쓴다.
valid 점수는 early stopping용으로만 쓰고 보고하지 않는다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# cuBLAS는 핸들을 만들 때 이 설정을 읽는다. torch import 전에 세워야 한다.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR, SEEDS, TASKS  # noqa: E402
from src.data import get_group, load_task, xy  # noqa: E402
from src.stacking import META_MODELS, build_meta  # noqa: E402
from src.train import TrainConfig, train_one_seed  # noqa: E402

# (라벨, GIN 모델 종류, 1단 train 특징 생성 방식, residual 연결 사용)
RUN_SPECS = {
    "gin+meta": ("gin+meta", "oof", True),
    "gin+stack": ("gin+stack", "oof", True),
    "gin+stack-infold": ("gin+stack", "infold", True),
    "gin+stack-plain": ("gin+stack", "oof", False),
    "gin+stack-plain-infold": ("gin+stack", "infold", False),
}
ALL_LABELS = ["base"] + list(RUN_SPECS)


def evaluate(group, task_name: str, predictions: list[dict], metric: str):
    """TDC 공식 평가. seed 5개 이상이면 evaluate_many를 쓴다.

    evaluate_many는 리더보드 제출 요건이라 seed가 5개 미만이면 예외를
    던지지 않고 ValueError 객체를 돌려준다. 그래서 개수를 먼저 본다.
    """
    key = task_name.lower()
    if len(predictions) >= 5:
        mean, std = group.evaluate_many(predictions)[key]
        return float(mean), float(std)
    vals = [float(group.evaluate(p)[key][metric]) for p in predictions]
    return float(np.mean(vals)), float(np.std(vals))


def run_base(group, task_name: str, seeds: list[int], cfg: TrainConfig,
             metric: str, verbose: bool):
    """1단 기반모델 단독 점수. GIN을 학습하지 않는다.

    build_meta가 train 전체에 학습한 모델로 test를 예측해 두므로, 그 예측을
    그대로 TDC에 넣으면 같은 train-only 조건의 기반모델 점수가 나온다.
    캐시가 있으면 특징을 다시 만들지 않는다.
    """
    per_model: dict = {m: [] for m in META_MODELS}
    diagnostics = []
    started = time.time()

    for seed in seeds:
        split = load_task(group, task_name, seed)
        tr_smiles, tr_y = xy(split["train"])
        va_smiles, _ = xy(split["valid"])
        te_smiles, _ = xy(split["test"])
        meta = build_meta(
            task_name, seed, tr_smiles, tr_y.to_numpy(), va_smiles, te_smiles,
            mode=cfg.meta_mode, n_folds=cfg.n_folds, verbose=verbose,
        )
        for name, pred in meta.base_test_pred.items():
            per_model[name].append({split["name"]: pred})
        for name, fit in meta.train_fit.items():
            diagnostics.append({
                "task": task_name, "seed": seed, "meta_mode": meta.mode,
                "base_model": name,
                "train_feature_mae": fit["mae"],
                "train_feature_mae_in_sd": fit["mae_in_sd"],
                "from_cache": meta.from_cache,
            })

    elapsed = time.time() - started
    rows = []
    for name, preds in per_model.items():
        mean, std = evaluate(group, task_name, preds, metric)
        rows.append({
            "task": task_name, "model": f"{name}-base", "metric": metric,
            "mean": round(mean, 4), "std": round(std, 4), "n_seeds": len(seeds),
            "valid_mean": None, "valid_at_init": None, "best_epochs": None,
            "meta_mode": "-", "residual_meta": "-",
            "seconds": round(elapsed, 1),
            "note": "train만 학습(1단과 동일 조건). baseline.csv는 train+valid 학습",
        })
        print(f"{task_name:24} {name + '-base':22} {metric}={mean:.4f} "
              f"+/- {std:.4f}", flush=True)
    return rows, diagnostics


def run_stacked(group, task_name: str, label: str, seeds: list[int],
                cfg: TrainConfig, metric: str, device: str, verbose: bool):
    """GIN 한 조건을 seed만큼 학습하고 TDC 평가를 받는다."""
    model_kind, meta_mode, residual = RUN_SPECS[label]
    cfg = TrainConfig(**{**cfg.__dict__, "meta_mode": meta_mode,
                         "residual_meta": residual})

    predictions, valid_scores, epochs, init_scores = [], [], [], []
    diagnostics = []
    started = time.time()

    for seed in seeds:
        name, pred, va, ep, meta, init = train_one_seed(
            group, task_name, model_kind, seed, cfg, device, verbose
        )
        predictions.append({name: pred})
        valid_scores.append(va)
        epochs.append(ep)
        init_scores.append(init)
        if meta is not None:
            for base, fit in meta.train_fit.items():
                diagnostics.append({
                    "task": task_name, "seed": seed, "meta_mode": meta.mode,
                    "base_model": base,
                    "train_feature_mae": fit["mae"],
                    "train_feature_mae_in_sd": fit["mae_in_sd"],
                    "from_cache": meta.from_cache,
                })
        if verbose:
            print(f"  seed {seed}: valid {metric}={va:+.4f}, best epoch {ep}",
                  flush=True)

    elapsed = time.time() - started
    mean, std = evaluate(group, task_name, predictions, metric)
    print(f"{task_name:24} {label:18} {metric}={mean:.4f} +/- {std:.4f}  "
          f"({elapsed:.0f}s, epochs {epochs})", flush=True)

    row = {
        "task": task_name, "model": label, "metric": metric,
        "mean": round(mean, 4), "std": round(std, 4), "n_seeds": len(seeds),
        "valid_mean": round(float(np.mean(valid_scores)), 4),
        # 학습 전 valid 점수. residual 연결에서는 "기반모델 평균"의 점수다.
        # valid_mean이 이것과 같으면 2단 학습이 아무것도 보태지 못한 것이다.
        "valid_at_init": round(float(np.mean(init_scores)), 4),
        "best_epochs": "|".join(str(e) for e in epochs),
        "meta_mode": meta_mode, "residual_meta": residual,
        "seconds": round(elapsed, 1),
        "note": "",
    }
    return [row], diagnostics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--models", nargs="+", default=ALL_LABELS,
                    help="base | " + " | ".join(RUN_SPECS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--patience", type=int, default=TrainConfig.patience)
    ap.add_argument("--n-folds", type=int, default=TrainConfig.n_folds,
                    help="1단 OOF 생성에 쓸 fold 수")
    ap.add_argument("--device", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_DIR / "stack.csv"))
    ap.add_argument("--diagnostics-out",
                    default=str(RESULTS_DIR / "stack_meta_diagnostics.csv"))
    args = ap.parse_args()

    unknown = [m for m in args.models if m not in ALL_LABELS]
    if unknown:
        raise SystemExit(f"알 수 없는 조건: {unknown}. 고를 수 있는 것: {ALL_LABELS}")

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, n_folds=args.n_folds,
    )

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device   : {device}", flush=True)
    if device == "cuda":
        print(f"gpu      : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"seeds    : {args.seeds}", flush=True)
    print(f"1단 모델 : {list(META_MODELS)} (특징 desc+fp, OOF {args.n_folds}-fold)",
          flush=True)
    print("", flush=True)

    group = get_group()
    rows, diagnostics = [], []

    for task in args.tasks:
        metric = TASKS[task]["metric"]
        for label in args.models:
            if label == "base":
                r, d = run_base(group, task, args.seeds, cfg, metric, args.verbose)
            else:
                r, d = run_stacked(group, task, label, args.seeds, cfg,
                                   metric, device, args.verbose)
            rows += r
            diagnostics += d
        print("", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"저장: {out}")

    if diagnostics:
        # 중복 제거: 같은 (task, seed, mode, base_model)이 여러 조건에서 나온다
        dg = pd.DataFrame(diagnostics).drop_duplicates(
            subset=["task", "seed", "meta_mode", "base_model"]
        ).sort_values(["task", "meta_mode", "base_model", "seed"])
        dout = Path(args.diagnostics_out)
        dg.to_csv(dout, index=False)
        print(f"저장: {dout}")
        print("\n1단 train 특징의 정확도 (데이터 표준편차 단위, 작을수록 정답에 가까움)")
        pivot = dg.pivot_table(index=["task", "base_model"], columns="meta_mode",
                               values="train_feature_mae_in_sd", aggfunc="mean")
        print(pivot.round(4).to_string())
        print("\n  oof가 infold보다 크게 나오는 것이 정상이다. infold는 자기가")
        print("  학습한 데이터를 예측한 값이라 실제 예측력보다 좋게 보인다.")


if __name__ == "__main__":
    main()
