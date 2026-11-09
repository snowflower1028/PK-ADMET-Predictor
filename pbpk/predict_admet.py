"""한 분자의 세 ADMET 물성을 예측한다. PBPK 입력을 만들기 위한 단계다.

**누출 처리가 이 파일의 요점이다.**

프로프라놀롤이 TDC 데이터셋에 들어 있는지 확인했다(InChIKey 골격 비교로
입체이성질체까지 잡았다).

    Caco2_Wang              없음  -> 그냥 예측하면 된다
    Clearance_Microsome_AZ  없음  -> 그냥 예측하면 된다
    PPBR_AZ                 train에 4건 있음 (R체·S체 각 2건)

더 중요한 점: `src/features.py`의 Morgan fingerprint는
`includeChirality=False`(기본값)이고 descriptor도 전부 비카이랄이다.
**모델 입장에서 R체·S체·라세미체는 완전히 같은 입력이다.** 그래서
PPBR을 그대로 예측하면 예측이 아니라 train 값을 외워 돌려주는 것이 된다.

그래서 PPBR만 **해당 분자를 train에서 빼고 다시 학습해서** 예측한다.
리더보드 숫자(`results/models.csv`)는 공식 split 결과이므로 손대지 않는다.
이쪽은 PBPK 입력용 별도 학습이고, 그 사실을 결과에 함께 적는다.

내 석사 연구가 입체선택적 대사라 이 한계가 특히 눈에 걸린다.
프로프라놀롤은 실제로 거울상 간 단백결합·대사가 다른데, 지금 특징
추출은 원리적으로 그 차이를 볼 수 없다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor

from src.config import SEEDS, TASKS
from src.data import load_task, xy
from src.features import featurize
from src.utils import set_seed

# 1주차 격자에서 세 과제 모두 가장 좋았던 조합
BEST_MODEL = "rf"
BEST_FEATURES = "desc+fp"


def skeleton_key(smiles: str) -> str | None:
    """입체화학을 무시한 분자 식별자.

    InChIKey의 첫 블록은 원자 연결만 반영하고 입체는 두 번째 블록에
    들어간다. 그래서 첫 블록만 비교하면 거울상끼리 같게 잡힌다.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol).split("-")[0]


@dataclass
class Prediction:
    task: str
    value: float
    std: float
    leaked: bool          # 원래 train에 있었는지
    excluded_n: int       # 학습에서 뺀 행 수
    train_values: list    # 뺐다면 그 값들 (보고용)
    n_seeds: int

    def __repr__(self) -> str:
        tag = f", train에서 {self.excluded_n}건 제외" if self.leaked else ""
        return (f"{self.task}: {self.value:.3f} +/- {self.std:.3f} "
                f"(seed {self.n_seeds}개{tag})")


def find_in_task(group, task_name: str, target_smiles: str) -> dict:
    """한 과제의 세 split에서 해당 분자를 찾는다."""
    key = skeleton_key(target_smiles)
    split = load_task(group, task_name, seed=SEEDS[0])
    hits = {"train": [], "valid": [], "test": []}
    for name in hits:
        for smi, y in zip(split[name]["Drug"], split[name]["Y"]):
            if skeleton_key(smi) == key:
                hits[name].append((smi, float(y)))
    return hits


def predict_one(
    group,
    task_name: str,
    target_smiles: str,
    seeds=None,
    exclude_target: bool = True,
) -> Prediction:
    """대상 분자의 물성을 예측한다.

    exclude_target=True면 학습 전에 대상 분자(입체이성질체 포함)를
    train/valid에서 빼낸다. train에 없으면 아무 일도 일어나지 않는다.

    seed마다 따로 학습해 평균±표준편차를 돌려준다. 단일 숫자로 보고하면
    seed 하나에서 운 좋게 나온 값일 수 있다.
    """
    seeds = list(seeds or SEEDS)
    key = skeleton_key(target_smiles)
    if key is None:
        raise ValueError(f"SMILES를 파싱할 수 없다: {target_smiles}")

    hits = find_in_task(group, task_name, target_smiles)
    leaked = bool(hits["train"] or hits["valid"])
    train_values = [v for _, v in hits["train"] + hits["valid"]]

    x_target = featurize([target_smiles], BEST_FEATURES)

    preds, excluded_n = [], 0
    for seed in seeds:
        set_seed(seed)
        split = load_task(group, task_name, seed)
        smiles, ys = [], []
        for part in ("train", "valid"):
            s, y = xy(split[part])
            smiles += list(s)
            ys += list(y.to_numpy())

        if exclude_target:
            keep = [i for i, s in enumerate(smiles) if skeleton_key(s) != key]
            excluded_n = len(smiles) - len(keep)
            smiles = [smiles[i] for i in keep]
            ys = [ys[i] for i in keep]

        x = featurize(smiles, BEST_FEATURES)
        model = RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
        model.fit(x, np.asarray(ys, dtype=np.float64))
        preds.append(float(model.predict(x_target)[0]))

    return Prediction(
        task=task_name,
        value=float(np.mean(preds)),
        std=float(np.std(preds)),
        leaked=leaked,
        excluded_n=excluded_n,
        train_values=train_values,
        n_seeds=len(seeds),
    )


def predict_all(group, target_smiles: str, seeds=None) -> dict[str, Prediction]:
    """세 과제를 모두 예측한다. 누출된 과제는 자동으로 제외 학습이 된다."""
    return {
        task: predict_one(group, task, target_smiles, seeds, exclude_target=True)
        for task in TASKS
    }
