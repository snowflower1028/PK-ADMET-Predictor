"""2주차 학습 루프.

규칙이 하나 있고, 이 파일 전체가 그 규칙을 지키기 위한 구조다.

    **test set은 마지막 예측에만 쓴다.**

epoch 선택(early stopping)도, 최선 가중치 선택도 전부 TDC가 준 valid로
한다. test 점수를 보고 뭔가를 고르면 리더보드와 비교할 수 없는 숫자가 된다.

valid 점수는 모델 선택용이라 scipy/sklearn으로 직접 계산한다. 반면
**보고하는 숫자는 전부 `group.evaluate_many()`가 준 것**이다. 두 경로를
섞지 않는다.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TASKS
from .data import load_task, xy
from .models import make_model
from .stacking import build_meta
from .torch_data import (
    FeatureScaler,
    GraphDataset,
    TargetScaler,
    VectorDataset,
    collate_graphs,
)
from .utils import seeded_generator, set_seed

# 모델 종류를 세 축으로 가른다. 조합이 늘어도 분기가 늘지 않게.
GRAPH_KINDS = ("gin", "gin+desc", "gin+meta", "gin+stack")  # 그래프를 받는다
DESC_KINDS = ("gin+desc", "gin+stack")                      # descriptor를 받는다
META_KINDS = ("gin+meta", "gin+stack")                      # 1단 예측값을 받는다


@dataclass
class TrainConfig:
    """기본값은 test를 보지 않고 미리 정했다.

    고르는 근거: 데이터가 637~1952개로 작다. 그래서 (1) batch는 작게,
    (2) epoch 상한은 넉넉히 두고 early stopping으로 끊고, (3) weight
    decay와 dropout으로 과적합을 막는다.
    """

    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 40  # valid가 이만큼 좋아지지 않으면 멈춘다
    min_delta: float = 1e-5
    hidden: int = 512  # MLP
    dim: int = 128  # GIN
    n_layers: int = 3  # GIN
    dropout: float = 0.3

    # --- 4주차 스태킹 (gin+meta, gin+stack에서만 쓴다) --------------------
    # meta_mode="oof"가 올바른 방식이고 기본값이다. "infold"는 누출
    # 대조군으로만 쓴다. 자세한 이유는 src/stacking.py 머리말 참고.
    meta_mode: str = "oof"
    n_folds: int = 5
    # 최종 출력을 "기반모델 평균 + 보정량"으로 쓸지. 끄면 그냥 concat 후
    # head가 전부 처음부터 학습한다(대조군).
    residual_meta: bool = True


def _valid_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """valid에서 모델을 고르기 위한 점수. 클수록 좋게 부호를 맞춘다."""
    if metric == "mae":
        return -float(np.mean(np.abs(y_true - y_pred)))
    if metric == "spearman":
        from scipy.stats import spearmanr

        rho = spearmanr(y_true, y_pred).correlation
        # 예측이 상수면 nan이 된다. 그건 최악으로 취급한다.
        return float(rho) if np.isfinite(rho) else -1.0
    raise ValueError(f"알 수 없는 지표: {metric}")


def build_loaders(
    task_name,
    model_kind,
    seed,
    cfg,
    tr_smiles,
    tr_y,
    va_smiles,
    va_y,
    te_smiles,
    verbose=False,
):
    """주어진 split을 Dataset/DataLoader로 바꾼다. scaler는 train만 보고 만든다.

    te_smiles는 "마지막에 예측할 분자들"이다. 리더보드 실행에서는 TDC test
    set이고, scripts/run_stack_demo.py에서는 사용자가 넣은 예시 분자다.
    타깃이 필요 없으므로 라벨 자리에는 0을 채운다.

    반환: (loaders, scaler, n_features, meta)
    """
    tr_y = np.asarray(tr_y)
    va_y = np.asarray(va_y)

    # 타깃 정규화 통계는 train에서만 구한다. valid/test를 보면 정보 누출이다.
    scaler = TargetScaler(tr_y)

    meta = None
    if model_kind == "mlp":
        train_ds = VectorDataset(tr_smiles, scaler.transform(tr_y))
        valid_ds = VectorDataset(va_smiles, scaler.transform(va_y))
        test_ds = VectorDataset(te_smiles, np.zeros(len(te_smiles)))
        collate = None
        n_features = train_ds.n_features
    elif model_kind in GRAPH_KINDS:
        if model_kind in META_KINDS:
            # 1단: RF/LightGBM을 돌려 예측값을 받아 온다. train 구간은 OOF로
            # 만든다(mode="oof"). 1단은 train만 쓴다 — valid는 아래 early
            # stopping이 쓰므로 1단에 넣으면 모델 선택이 오염된다.
            meta = build_meta(
                task_name, seed, tr_smiles, tr_y, va_smiles, te_smiles,
                mode=cfg.meta_mode, n_folds=cfg.n_folds, verbose=verbose,
            )
            # 기반모델 예측값을 **타깃과 같은 스케일러로** 정규화한다.
            # 같은 공간에 두면 residual 연결(기반모델 평균 + 보정량)이
            # 그대로 성립하고, 세 과제의 단위 차이(log, %, uL/min/mg)도
            # 타깃과 똑같이 흡수된다.
            tr_meta = scaler.transform(meta.train)
            va_meta = scaler.transform(meta.valid)
            te_meta = scaler.transform(meta.test)
        else:
            tr_meta = va_meta = te_meta = None

        train_ds = GraphDataset(tr_smiles, scaler.transform(tr_y), tr_meta)
        valid_ds = GraphDataset(va_smiles, scaler.transform(va_y), va_meta)
        test_ds = GraphDataset(te_smiles, np.zeros(len(te_smiles)), te_meta)
        for name, ds in (("train", train_ds), ("valid", valid_ds), ("test", test_ds)):
            if ds.n_failed:
                # 1주차에 0건을 확인했다. 생기면 데이터가 바뀐 것이므로 알린다.
                raise RuntimeError(
                    f"{task_name} {name}에서 SMILES {ds.n_failed}개가 파싱되지 "
                    "않았다. test 예측 개수가 어긋나므로 그냥 넘기지 않는다."
                )
        # descriptor 정규화 통계도 train에서만 구해 세 split에 같이 적용한다
        desc_scaler = (
            FeatureScaler(train_ds.desc_raw) if model_kind in DESC_KINDS else None
        )
        for ds in (train_ds, valid_ds, test_ds):
            ds.set_desc_scaler(desc_scaler)
        collate = collate_graphs
        n_features = train_ds.n_desc if model_kind in DESC_KINDS else None
    else:
        raise ValueError(f"알 수 없는 모델: {model_kind}")

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate, drop_last=len(train_ds) > cfg.batch_size,
            # 전역 RNG가 아니라 seed에서 만든 생성기로 섞는다
            generator=seeded_generator(seed),
        ),
        # valid/test는 순서를 섞지 않는다. test는 순서가 곧 제출 순서다.
        "valid": DataLoader(
            valid_ds, batch_size=256, shuffle=False, collate_fn=collate
        ),
        "test": DataLoader(
            test_ds, batch_size=256, shuffle=False, collate_fn=collate
        ),
    }
    return loaders, scaler, n_features, meta


def _make_loaders(task_name, model_kind, seed, group, cfg, verbose=False):
    """TDC 공식 split을 받아 build_loaders에 넘긴다."""
    split = load_task(group, task_name, seed)
    tr_smiles, tr_y = xy(split["train"])
    va_smiles, va_y = xy(split["valid"])
    te_smiles, _ = xy(split["test"])

    tr_y = tr_y.to_numpy()
    va_y = va_y.to_numpy()

    loaders, scaler, n_features, meta = build_loaders(
        task_name, model_kind, seed, cfg,
        tr_smiles, tr_y, va_smiles, va_y, te_smiles, verbose,
    )
    return split, loaders, scaler, va_y, n_features, meta


def _forward(model, batch, model_kind, device):
    """모델 종류에 따라 배치를 풀어 넣는다."""
    if model_kind == "mlp":
        x, y = batch
        return model(x.to(device)), y.to(device)
    atom_x, edge_index, edge_attr, batch_idx, y, desc, meta = batch
    pred = model(
        atom_x.to(device), edge_index.to(device),
        edge_attr.to(device), batch_idx.to(device),
        desc.to(device) if desc.shape[1] > 0 else None,
        meta.to(device) if meta.shape[1] > 0 else None,
    )
    return pred, y.to(device)


def build_model(model_kind: str, cfg: "TrainConfig", n_features, meta, device: str):
    """TrainConfig와 1단 특징 정보로 모델을 만든다.

    hyperparameter를 한 군데서만 읽게 하려고 떼어 둔 함수다. train_one_seed와
    stack_predict가 서로 다른 dropout으로 학습하면 비교가 성립하지 않는다.
    """
    if model_kind == "mlp":
        model = make_model("mlp", n_features=n_features,
                           hidden=cfg.hidden, dropout=cfg.dropout)
    else:
        # 그래프 계열은 n_features(descriptor 개수)와 n_meta(기반모델 개수)로
        # 갈린다. 둘 다 0이면 순수 GIN이다.
        kwargs = dict(dim=cfg.dim, n_layers=cfg.n_layers,
                      dropout=min(cfg.dropout, 0.2))
        if model_kind in META_KINDS:
            kwargs["n_meta"] = meta.n_meta
            kwargs["residual_meta"] = cfg.residual_meta
        model = make_model(model_kind, n_features=n_features, **kwargs)
    return model.to(device)


def fit_model(
    model,
    model_kind: str,
    loaders: dict,
    scaler: TargetScaler,
    va_y_raw: np.ndarray,
    metric: str,
    cfg: "TrainConfig",
    device: str,
    verbose: bool = False,
):
    """epoch 루프를 돌고 **최선 valid 가중치를 model에 되돌려 놓는다.**

    반환: (최선 valid 점수, 그 epoch, 학습 전 valid 점수)

    세 번째 값은 **진단용으로만** 쓴다(모델 선택에 쓰지 않는다). residual
    연결을 켠 스태킹 모델은 마지막 층이 0으로 초기화돼 있어서 학습 전
    예측이 정확히 "기반모델 평균"이다. 그 점수를 적어 두면 학습이 기반모델
    대비 실제로 무엇을 보탰는지 바로 보인다 — 최선 valid가 학습 전과
    같으면 2단은 아무것도 보태지 못한 것이다.

    일반 모델에서는 학습 전 가중치가 무작위라 이 값이 아주 나쁘게 나오고,
    최선으로 뽑힐 일이 없다. 그래서 2주차 숫자는 영향받지 않는다.

    train_one_seed와 stack_predict가 같은 루프를 쓰도록 떼어 둔 함수다.
    학습 루프가 두 벌이 되면 한쪽만 고치는 사고가 난다.

    규칙은 그대로다: **epoch 선택은 valid로만 한다.** test는 이 함수가
    아예 보지 않는다(loaders["test"]를 쓰지 않는다).
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=cfg.patience // 4
    )
    # Huber는 이상치에 덜 끌린다. Clearance는 3~150으로 꼬리가 길다.
    loss_fn = nn.HuberLoss(delta=1.0)

    # 학습 전 valid 점수. eval 모드 forward는 난수를 뽑지도, BatchNorm의
    # running 통계를 바꾸지도 않으므로 학습 경로에 영향이 없다.
    init_pred = scaler.inverse(_predict(model, loaders["valid"], model_kind, device))
    init_score = _valid_score(va_y_raw, init_pred, metric)

    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            pred, y = _forward(model, batch, model_kind, device)
            loss = loss_fn(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 그래프 모델에서 초반 gradient가 튀는 걸 막는다
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # valid로만 판단한다
        va_pred = scaler.inverse(_predict(model, loaders["valid"], model_kind, device))
        score = _valid_score(va_y_raw, va_pred, metric)
        scheduler.step(score)

        if score > best_score + cfg.min_delta:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

        if verbose and epoch % 20 == 0:
            print(f"    epoch {epoch:3d}  valid {metric}={score:+.4f} "
                  f"(best {best_score:+.4f} @ {best_epoch})", flush=True)

    model.load_state_dict(best_state)
    return float(best_score), best_epoch, float(init_score)


@torch.no_grad()
def _predict(model, loader, model_kind, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        pred, _ = _forward(model, batch, model_kind, device)
        out.append(pred.detach().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_one_seed(
    group,
    task_name: str,
    model_kind: str,
    seed: int,
    cfg: TrainConfig | None = None,
    device: str | None = None,
    verbose: bool = False,
):
    """한 seed를 학습하고 test 예측을 돌려준다.

    반환: (TDC 과제명, test 예측 ndarray, 최선 valid 점수, 멈춘 epoch,
          1단 특징 정보 또는 None, 학습 전 valid 점수)
    """
    cfg = cfg or TrainConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    metric = TASKS[task_name]["metric"]
    split, loaders, scaler, va_y_raw, n_features, meta = _make_loaders(
        task_name, model_kind, seed, group, cfg, verbose
    )

    model = build_model(model_kind, cfg, n_features, meta, device)

    best_score, best_epoch, init_score = fit_model(
        model, model_kind, loaders, scaler, va_y_raw, metric, cfg, device, verbose
    )

    # 최선 가중치로 되돌린 뒤에야 test를 본다(fit_model이 되돌려 놓는다)
    test_pred = scaler.inverse(_predict(model, loaders["test"], model_kind, device))

    return split["name"], test_pred, float(best_score), best_epoch, meta, init_score


def train_task(
    group,
    task_name: str,
    model_kind: str,
    seeds,
    cfg: TrainConfig | None = None,
    device: str | None = None,
    verbose: bool = False,
):
    """seed를 돌려 모으고 TDC 공식 평가를 받는다.

    반환: (평균, 표준편차, seed별 valid 점수, seed별 멈춘 epoch)
    """
    seeds = list(seeds)
    metric = TASKS[task_name]["metric"]
    key = task_name.lower()

    predictions, valid_scores, epochs = [], [], []
    for seed in seeds:
        name, pred, va, ep, _meta, _init = train_one_seed(
            group, task_name, model_kind, seed, cfg, device, verbose
        )
        predictions.append({name: pred})
        valid_scores.append(va)
        epochs.append(ep)
        if verbose:
            print(f"  seed {seed}: valid {metric}={va:+.4f}, best epoch {ep}",
                  flush=True)

    if len(predictions) >= 5:
        # evaluate_many는 {과제명: [평균, 표준편차]}를 준다 (지표명 키가 없다)
        mean, std = group.evaluate_many(predictions)[key]
    else:
        # seed 5개 미만은 리더보드 형식이 아니다. 참고용으로만 쓴다.
        vals = [float(group.evaluate(p)[key][metric]) for p in predictions]
        mean, std = float(np.mean(vals)), float(np.std(vals))

    return float(mean), float(std), valid_scores, epochs
