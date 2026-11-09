"""PyTorch Dataset과 collate_fn.

두 가지를 만든다.
- VectorDataset : 1주차와 같은 descriptor+fingerprint 벡터. MLP용.
- GraphDataset  : 분자 그래프. GNN용. 분자마다 원자 수가 다르므로
                  collate_fn을 직접 써야 한다.

그래프 배치를 만드는 방법은 두 가지다. 하나는 가장 큰 분자에 맞춰 0으로
패딩하는 것이고, 다른 하나는 여러 분자를 **연결 성분이 떨어진 하나의 큰
그래프**로 이어 붙이는 것이다. 후자를 쓴다. 패딩이 없어서 낭비가 없고,
메시지 전달이 인접 행렬 곱이 아니라 엣지 단위 scatter로 끝난다.
핵심은 분자 k의 원자 인덱스에 '앞선 분자들의 원자 수 합'을 더해 주는 것이다.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import descriptors, featurize
from .graph import BOND_DIM, MolGraph, smiles_to_graph


class VectorDataset(Dataset):
    """SMILES를 고정 길이 벡터로 바꿔 들고 있는다.

    특징 추출은 __init__에서 한 번에 끝낸다. 데이터가 최대 2790개라
    전부 메모리에 올라가고, 매 epoch마다 RDKit을 다시 부르는 것이
    학습보다 느려지는 걸 막는다.
    """

    def __init__(self, smiles: list[str], y: np.ndarray, kind: str = "desc+fp"):
        self.x = torch.from_numpy(featurize(smiles, kind).astype(np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])


class GraphDataset(Dataset):
    """SMILES를 MolGraph로 바꿔 들고 있는다.

    descriptor도 같이 계산해 둔다(`desc_raw`). 그래프만 보는 GIN은
    이걸 쓰지 않고, descriptor를 함께 받는 변형(gin+desc)만 쓴다.
    27차원이라 계산·메모리 부담이 없어서 항상 만들어 둔다.

    descriptor를 쓸 때는 반드시 `set_desc_scaler()`로 정규화한다.
    MolWt는 수백, qed는 0~1이라 스케일을 안 맞추면 큰 값이 gradient를
    독차지한다.
    """

    def __init__(self, smiles: list[str], y: np.ndarray):
        y = np.asarray(y, dtype=np.float32)
        graphs: list[MolGraph] = []
        targets: list[float] = []
        kept_smiles: list[str] = []
        self.n_failed = 0
        for smi, target in zip(smiles, y):
            g = smiles_to_graph(smi)
            if g is None:
                # 1주차 점검에서 세 데이터셋 모두 0건이었다. 그래도 세어 둔다.
                self.n_failed += 1
                continue
            graphs.append(g)
            targets.append(float(target))
            kept_smiles.append(smi)
        self.graphs = graphs
        self.y = np.asarray(targets, dtype=np.float32)
        # 파싱에 성공한 분자만 가지고 계산해야 그래프와 행이 어긋나지 않는다
        self.desc_raw = descriptors(kept_smiles).astype(np.float32)
        self.desc = np.zeros((len(graphs), 0), dtype=np.float32)

    def set_desc_scaler(self, scaler: "FeatureScaler | None") -> None:
        """descriptor를 정규화해 모델에 넘길 형태로 만든다.

        scaler는 **train에서만** 만든 것을 세 split에 모두 적용한다.
        split별로 따로 만들면 그 자체가 정보 누출이다.
        """
        if scaler is None:
            self.desc = np.zeros((len(self.graphs), 0), dtype=np.float32)
        else:
            self.desc = scaler.transform(self.desc_raw).astype(np.float32)

    @property
    def n_desc(self) -> int:
        return int(self.desc.shape[1])

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, i: int):
        return self.graphs[i], self.y[i], self.desc[i]


def collate_graphs(batch):
    """여러 MolGraph를 떨어진 성분 하나의 큰 그래프로 이어 붙인다.

    돌려주는 것
      atom_x     : (배치 전체 원자 수, ATOM_DIM)
      edge_index : (2, 배치 전체 엣지 수)   — 오프셋이 더해진 인덱스
      edge_attr  : (배치 전체 엣지 수, BOND_DIM)
      batch_idx  : (배치 전체 원자 수,)     — 각 원자가 몇 번째 분자인지
      y          : (분자 수,)
      desc       : (분자 수, descriptor 수) — 안 쓰면 열이 0개

    batch_idx가 핵심이다. 마지막에 원자별 벡터를 분자별로 모을 때
    이 인덱스로 scatter 합을 하면 분자 단위 표현이 나온다.
    """
    graphs, ys, descs = zip(*batch)

    atom_x = np.concatenate([g.atom_x for g in graphs], axis=0)

    edge_indices = []
    offset = 0
    for g in graphs:
        if g.edge_index.shape[1] > 0:
            edge_indices.append(g.edge_index + offset)
        offset += g.n_atoms  # 다음 분자의 원자 번호는 여기서부터 시작한다
    if edge_indices:
        edge_index = np.concatenate(edge_indices, axis=1)
        edge_attr = np.concatenate([g.edge_attr for g in graphs], axis=0)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, BOND_DIM), dtype=np.float32)

    batch_idx = np.concatenate(
        [np.full(g.n_atoms, k, dtype=np.int64) for k, g in enumerate(graphs)]
    )

    return (
        torch.from_numpy(atom_x),
        torch.from_numpy(edge_index),
        torch.from_numpy(edge_attr),
        torch.from_numpy(batch_idx),
        torch.tensor(ys, dtype=torch.float32),
        torch.from_numpy(np.stack(descs).astype(np.float32)),
    )


class FeatureScaler:
    """descriptor 열마다 평균 0, 표준편차 1로 맞춘다.

    descriptor는 열끼리 스케일이 전혀 다르다. MolWt는 수백, qed는 0~1,
    BertzCT는 수천까지 간다. 정규화 없이 신경망에 넣으면 큰 열이
    gradient를 독차지해서 작은 열은 사실상 무시된다. (트리 모델은
    분할 기준만 보기 때문에 1주차에는 이 문제가 없었다.)

    통계는 **train에서만** 구한다.
    """

    def __init__(self, x_train: np.ndarray):
        x_train = np.asarray(x_train, dtype=np.float64)
        self.mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        # 상수 열(모든 분자가 같은 값)은 나누기 0이 된다. 1로 두면
        # 정규화 후 전부 0이 되어 모델이 무시한다. 그게 맞는 동작이다.
        self.std = np.where(std < 1e-8, 1.0, std)

    def transform(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = (x - self.mean) / self.std
        # descriptor 계산에서 간혹 극단값이 나온다. 꼬리를 잘라 학습이
        # 한 분자 때문에 흔들리지 않게 한다.
        return np.clip(out, -10.0, 10.0)


class TargetScaler:
    """타깃을 평균 0, 표준편차 1로 맞춘다.

    세 과제의 타깃 범위가 전혀 다르다(Caco2 -7.8~-3.5, PPBR 10~100,
    Clearance 3~150). 스케일을 안 맞추면 같은 learning rate로 세 과제를
    돌릴 수 없다.

    중요: 평균·표준편차는 **train에서만** 구한다. valid나 test의 통계를
    쓰면 그 자체가 정보 누출이다.
    """

    def __init__(self, y_train: np.ndarray):
        y_train = np.asarray(y_train, dtype=np.float64)
        self.mean = float(y_train.mean())
        self.std = float(y_train.std())
        if self.std < 1e-8:  # 상수 타깃이면 나누기 0이 된다
            self.std = 1.0

    def transform(self, y):
        return (np.asarray(y, dtype=np.float64) - self.mean) / self.std

    def inverse(self, y):
        """예측값을 원래 단위로 되돌린다. TDC 평가는 원래 단위로 해야 한다."""
        return np.asarray(y, dtype=np.float64) * self.std + self.mean
