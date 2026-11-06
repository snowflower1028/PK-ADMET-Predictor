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
from .torch_data import (
    FeatureScaler,
    GraphDataset,
    TargetScaler,
    VectorDataset,
    collate_graphs,
)
from .utils import seeded_generator, set_seed


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


def _make_loaders(task_name, model_kind, seed, group, cfg):
    """TDC split을 Dataset/DataLoader로 바꾼다. scaler는 train만 보고 만든다."""
    split = load_task(group, task_name, seed)
    tr_smiles, tr_y = xy(split["train"])
    va_smiles, va_y = xy(split["valid"])
    te_smiles, _ = xy(split["test"])

    tr_y = tr_y.to_numpy()
    va_y = va_y.to_numpy()

    # 타깃 정규화 통계는 train에서만 구한다. valid/test를 보면 정보 누출이다.
    scaler = TargetScaler(tr_y)

    if model_kind == "mlp":
        train_ds = VectorDataset(tr_smiles, scaler.transform(tr_y))
        valid_ds = VectorDataset(va_smiles, scaler.transform(va_y))
        test_ds = VectorDataset(te_smiles, np.zeros(len(te_smiles)))
        collate = None
        n_features = train_ds.n_features
    elif model_kind in ("gin", "gin+desc"):
        train_ds = GraphDataset(tr_smiles, scaler.transform(tr_y))
        valid_ds = GraphDataset(va_smiles, scaler.transform(va_y))
        test_ds = GraphDataset(te_smiles, np.zeros(len(te_smiles)))
        for name, ds in (("train", train_ds), ("valid", valid_ds), ("test", test_ds)):
            if ds.n_failed:
                # 1주차에 0건을 확인했다. 생기면 데이터가 바뀐 것이므로 알린다.
                raise RuntimeError(
                    f"{task_name} {name}에서 SMILES {ds.n_failed}개가 파싱되지 "
                    "않았다. test 예측 개수가 어긋나므로 그냥 넘기지 않는다."
                )
        # descriptor 정규화 통계도 train에서만 구해 세 split에 같이 적용한다
        desc_scaler = (
            FeatureScaler(train_ds.desc_raw) if model_kind == "gin+desc" else None
        )
        for ds in (train_ds, valid_ds, test_ds):
            ds.set_desc_scaler(desc_scaler)
        collate = collate_graphs
        n_features = train_ds.n_desc if model_kind == "gin+desc" else None
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
    return split, loaders, scaler, va_y, n_features


def _forward(model, batch, model_kind, device):
    """모델 종류에 따라 배치를 풀어 넣는다."""
    if model_kind == "mlp":
        x, y = batch
        return model(x.to(device)), y.to(device)
    atom_x, edge_index, edge_attr, batch_idx, y, desc = batch
    pred = model(
        atom_x.to(device), edge_index.to(device),
        edge_attr.to(device), batch_idx.to(device),
        desc.to(device) if desc.shape[1] > 0 else None,
    )
    return pred, y.to(device)


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

    반환: (TDC 과제명, test 예측 ndarray, 최선 valid 점수, 멈춘 epoch)
    """
    cfg = cfg or TrainConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    metric = TASKS[task_name]["metric"]
    split, loaders, scaler, va_y_raw, n_features = _make_loaders(
        task_name, model_kind, seed, group, cfg
    )

    if model_kind == "mlp":
        model = make_model("mlp", n_features=n_features,
                           hidden=cfg.hidden, dropout=cfg.dropout)
    else:
        # gin과 gin+desc는 n_features(descriptor 개수)로만 갈린다
        model = make_model(model_kind, n_features=n_features,
                           dim=cfg.dim, n_layers=cfg.n_layers,
                           dropout=min(cfg.dropout, 0.2))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=cfg.patience // 4
    )
    # Huber는 이상치에 덜 끌린다. Clearance는 3~150으로 꼬리가 길다.
    loss_fn = nn.HuberLoss(delta=1.0)

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

    # 최선 가중치로 되돌린 뒤에야 test를 본다
    model.load_state_dict(best_state)
    test_pred = scaler.inverse(_predict(model, loaders["test"], model_kind, device))

    return split["name"], test_pred, float(best_score), best_epoch


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
        name, pred, va, ep = train_one_seed(
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
