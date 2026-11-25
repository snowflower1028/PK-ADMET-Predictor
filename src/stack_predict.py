"""임의의 분자에 2단 스태킹 파이프라인을 적용한다.

리더보드 실행(`scripts/run_stack.py`)은 TDC가 정한 test set만 예측한다.
하지만 "아는 약물 하나를 넣으면 말이 되는 값이 나오는가"를 보려면 아무
SMILES나 넣을 수 있어야 한다. 그 경로가 이 파일이다.

구조는 리더보드 실행과 **같은 함수들**을 쓴다(`build_loaders`,
`build_model`, `fit_model`). 학습 루프나 hyperparameter를 따로 들고 있으면
"데모에서 본 숫자"와 "표에 적은 숫자"가 다른 모델에서 나오게 된다.

달라지는 것은 예측 대상뿐이다. TDC test set 자리에 사용자가 준 분자를
넣는다. 그러면 1단 기반모델도, 2단 GIN도 그 분자들을 예측해 준다.
"""
from __future__ import annotations

import numpy as np

from .config import TASKS
from .train import TrainConfig, _predict, build_loaders, build_model, fit_model
from .utils import set_seed


def fit_predict_stack(
    task_name: str,
    tr_smiles: list[str],
    tr_y: np.ndarray,
    va_smiles: list[str],
    va_y: np.ndarray,
    target_smiles: list[str],
    seed: int,
    model_kind: str = "gin+stack",
    cfg: TrainConfig | None = None,
    device: str | None = None,
    verbose: bool = False,
) -> dict:
    """주어진 train/valid로 2단 파이프라인을 학습하고 target_smiles를 예측한다.

    반환 dict
      "rf", "lgbm"   : 1단 기반모델이 직접 낸 예측 (원래 단위)
      model_kind     : 2단 GIN의 최종 예측 (원래 단위)
      "base_mean"    : 기반모델 예측의 평균. residual 설계의 출발점이라
                       GIN이 이 값을 어느 방향으로 얼마나 움직였는지 볼 수 있다.
      "valid_score"  : early stopping에 쓴 최선 valid 점수
      "best_epoch"   : 그 epoch
      "meta"         : MetaFeatures (1단 진단 정보 포함)

    주의: 누출 점검은 호출하는 쪽 책임이다. target_smiles가 tr/va에 들어
    있으면 모델은 외운 값을 돌려준다. scripts/run_stack_demo.py는 골격
    키로 먼저 빼고 넘긴다.
    """
    import torch

    cfg = cfg or TrainConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    metric = TASKS[task_name]["metric"]

    # train_one_seed와 같은 자리에서 seed를 고정한다. 그래야 같은 입력에
    # 같은 숫자가 나온다.
    set_seed(seed)

    loaders, scaler, n_features, meta = build_loaders(
        task_name, model_kind, seed, cfg,
        tr_smiles, np.asarray(tr_y), va_smiles, np.asarray(va_y),
        target_smiles, verbose,
    )
    model = build_model(model_kind, cfg, n_features, meta, device)
    best_score, best_epoch, init_score = fit_model(
        model, model_kind, loaders, scaler, np.asarray(va_y),
        metric, cfg, device, verbose,
    )

    # fit_model이 최선 valid 가중치를 되돌려 놓은 뒤에 예측한다
    pred = scaler.inverse(_predict(model, loaders["test"], model_kind, device))

    out = {model_kind: np.asarray(pred, dtype=np.float64),
           "valid_score": best_score, "best_epoch": best_epoch,
           "valid_at_init": init_score, "meta": meta}
    if meta is not None:
        for name, col in zip(meta.columns, meta.test.T):
            # 열 이름은 "pred_rf" 형태로 들어 있다
            out[name.removeprefix("pred_")] = np.asarray(col, dtype=np.float64)
        out["base_mean"] = np.asarray(meta.test.mean(axis=1), dtype=np.float64)
    return out
