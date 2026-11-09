"""SMILES를 그래프(원자 노드 + 결합 엣지)로 바꾼다.

1주차 descriptor/fingerprint는 분자 하나를 고정 길이 벡터로 눌러 버린다.
그래프는 어떤 원자가 어떤 원자에 붙어 있는지를 그대로 들고 간다.
GNN이 descriptor보다 나을 수 있다면 그 차이는 여기서 나온다.

torch_geometric을 쓰지 않고 직접 만든다. 이 정도 규모에서는 의존성을
하나 더 들이는 값이 크고, 배치 묶는 방식(패딩이 아니라 떨어진 성분 하나의
큰 그래프로 이어 붙이고 원자 인덱스에 오프셋을 더한다)을 한 줄씩
설명할 수 있어야 하기 때문이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# 원자 종류는 ADMET 데이터에 실제로 나오는 것만 둔다. 목록에 없으면 "기타"로
# 간다. 희귀 원소까지 one-hot을 늘리면 거의 안 쓰이는 차원만 늘어난다.
ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _one_hot(value, choices: list) -> list[float]:
    """choices에 있으면 그 자리를 1로, 없으면 마지막 '기타' 자리를 1로."""
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    """원자 하나를 벡터로. 결합수·전하·수소수·혼성·방향족·고리 여부."""
    return (
        _one_hot(atom.GetSymbol(), ATOM_TYPES)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    """결합 하나를 벡터로. 결합 차수·고리 여부·공액 여부."""
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [float(bond.IsInRing()), float(bond.GetIsConjugated())]
    )


ATOM_DIM = (
    len(ATOM_TYPES) + 1 + len(DEGREES) + 1 + len(FORMAL_CHARGES) + 1
    + len(NUM_HS) + 1 + len(HYBRIDIZATIONS) + 1 + 2
)
BOND_DIM = len(BOND_TYPES) + 1 + 2


@dataclass
class MolGraph:
    """분자 하나의 그래프.

    atom_x    : (원자 수, ATOM_DIM)
    edge_index: (2, 엣지 수)  — 무향 결합을 양방향 두 줄로 넣는다
    edge_attr : (엣지 수, BOND_DIM)
    """

    atom_x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray

    @property
    def n_atoms(self) -> int:
        return self.atom_x.shape[0]


def smiles_to_graph(smiles: str) -> MolGraph | None:
    """SMILES 하나를 MolGraph로. 파싱 실패하면 None을 돌려준다.

    None을 조용히 0으로 바꾸지 않는 이유: 1주차에서 세 데이터셋 모두 파싱
    실패가 0건이라는 걸 확인했다. 그래도 실패가 생기면 그건 데이터가 바뀐
    것이므로 호출하는 쪽에서 알아차려야 한다.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_x = np.asarray(
        [atom_features(a) for a in mol.GetAtoms()], dtype=np.float32
    )
    if atom_x.ndim != 2 or atom_x.shape[0] == 0:
        return None

    srcs: list[int] = []
    dsts: list[int] = []
    attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        # 양방향으로 넣어야 메시지가 두 방향 다 흐른다
        srcs += [i, j]
        dsts += [j, i]
        attrs += [feat, feat]

    if srcs:
        edge_index = np.asarray([srcs, dsts], dtype=np.int64)
        edge_attr = np.asarray(attrs, dtype=np.float32)
    else:
        # 결합이 없는 분자(단일 원자 이온 등). 빈 엣지로 두면 GNN은
        # 메시지 없이 노드 자신만 보게 되는데, 그게 맞는 동작이다.
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, BOND_DIM), dtype=np.float32)

    return MolGraph(atom_x=atom_x, edge_index=edge_index, edge_attr=edge_attr)
