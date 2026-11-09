"""1주차 베이스라인: RandomForest와 LightGBM.

2주차 PyTorch 모델이 이 숫자를 넘는지로 개선을 판단한다.
베이스라인 없이 딥러닝 숫자만 보고하면 좋은지 나쁜지 알 수 없다.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .config import SEEDS, TASKS
from .data import load_task, xy
from .features import featurize
from .utils import set_seed


def make_model(kind: str, seed: int):
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    if kind == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=800, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbose=-1,
        )
    raise ValueError(f"알 수 없는 모델: {kind}")


def run_one_seed(group, task_name: str, model_kind: str, feat_kind: str, seed: int):
    """seed 하나로 학습하고 test 예측을 돌려준다."""
    set_seed(seed)
    split = load_task(group, task_name, seed)

    # train과 valid를 합쳐 학습한다. 트리 모델은 early stopping을 쓰지 않으므로
    # valid를 따로 둘 이유가 없고, 2주차 신경망에서는 valid를 쓰게 된다.
    train_smiles, train_y = xy(split["train"])
    valid_smiles, valid_y = xy(split["valid"])
    fit_smiles = train_smiles + valid_smiles
    fit_y = np.concatenate([train_y.to_numpy(), valid_y.to_numpy()])

    test_smiles, _ = xy(split["test"])

    x_fit = featurize(fit_smiles, feat_kind)
    x_test = featurize(test_smiles, feat_kind)

    model = make_model(model_kind, seed)
    model.fit(x_fit, fit_y)
    return split["name"], model.predict(x_test)


def run_task(group, task_name: str, model_kind: str, feat_kind: str, seeds=None):
    """seed를 돌려가며 예측을 모으고 TDC 공식 평가를 받는다.

    반환: (평균, 표준편차)

    주의: group.evaluate_many()는 리더보드 제출 요건이라 seed 5개 미만이면
    평가를 거부하고 ValueError 객체를 돌려준다(예외를 던지지 않는다).
    그래서 빠른 점검용으로 seed가 적을 때는 evaluate()를 직접 쓴다.
    """
    seeds = list(seeds or SEEDS)
    metric = TASKS[task_name]["metric"]
    key = task_name.lower()  # TDC 결과 dict의 키는 소문자다

    predictions_list = []
    for seed in seeds:
        name, y_pred = run_one_seed(group, task_name, model_kind, feat_kind, seed)
        predictions_list.append({name: y_pred})

    if len(predictions_list) >= 5:
        # evaluate_many는 {과제명: [평균, 표준편차]}를 준다. evaluate()와 달리
        # 지표 이름이 키에 들어가지 않는다(지표는 TDC가 과제별로 고정해 둔다).
        scores = group.evaluate_many(predictions_list)
        mean, std = scores[key]
        return float(mean), float(std)

    # {과제명: {지표명: 값}}
    values = [float(group.evaluate(p)[key][metric]) for p in predictions_list]
    return float(np.mean(values)), float(np.std(values))


def describe_task(task_name: str) -> str:
    info = TASKS[task_name]
    return f"{task_name} — {info['label']}, 지표 {info['metric']}"
