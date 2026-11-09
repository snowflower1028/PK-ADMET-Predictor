"""RDKit으로 SMILES를 숫자 특징으로 바꾼다.

두 가지를 쓴다.
- descriptor: 물리화학적 성질(분자량, logP, TPSA 등). 해석이 쉽다.
- Morgan fingerprint: 부분 구조의 존재 여부를 비트로 적은 것. 표현력이 크다.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from .config import MORGAN_BITS, MORGAN_RADIUS

RDLogger.DisableLog("rdApp.*")  # 파싱 경고가 로그를 덮지 않게 한다

# ADMET 예측에서 널리 쓰이는 descriptor만 고른다.
# 전체 200여 개를 다 쓰면 결측·무한값 처리가 번거롭고 과적합도 쉽다.
DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "MolMR", "TPSA", "LabuteASA",
    "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "NumHeteroatoms",
    "HeavyAtomCount", "RingCount", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticRings", "FractionCSP3", "NHOHCount", "NOCount",
    "BertzCT", "BalabanJ", "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
    "Chi0v", "Chi1v", "Chi2v", "qed",
]

_DESCRIPTOR_FUNCS = [(n, getattr(Descriptors, n)) for n in DESCRIPTOR_NAMES]
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)


def to_mol(smiles: str):
    """SMILES를 RDKit 분자 객체로 바꾼다. 실패하면 None."""
    return Chem.MolFromSmiles(smiles)


def descriptors(smiles_list: list[str]) -> np.ndarray:
    """(분자 수, descriptor 수) 행렬. 파싱 실패나 계산 실패는 0으로 채운다."""
    rows = []
    for smi in smiles_list:
        mol = to_mol(smi)
        if mol is None:
            rows.append([0.0] * len(_DESCRIPTOR_FUNCS))
            continue
        row = []
        for _, func in _DESCRIPTOR_FUNCS:
            try:
                row.append(float(func(mol)))
            except Exception:
                row.append(0.0)
        rows.append(row)
    arr = np.asarray(rows, dtype=np.float64)
    # 무한값·결측은 트리 모델에서도 문제가 되므로 여기서 막는다
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def morgan(smiles_list: list[str]) -> np.ndarray:
    """(분자 수, MORGAN_BITS) 이진 행렬."""
    rows = []
    for smi in smiles_list:
        mol = to_mol(smi)
        if mol is None:
            rows.append(np.zeros(MORGAN_BITS, dtype=np.uint8))
            continue
        rows.append(_MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.uint8))
    return np.vstack(rows)


def featurize(smiles_list: list[str], kind: str = "desc+fp") -> np.ndarray:
    """kind: 'desc' | 'fp' | 'desc+fp'."""
    if kind == "desc":
        return descriptors(smiles_list)
    if kind == "fp":
        return morgan(smiles_list).astype(np.float64)
    if kind == "desc+fp":
        return np.hstack([descriptors(smiles_list), morgan(smiles_list)])
    raise ValueError(f"알 수 없는 특징 종류: {kind}")


def n_invalid(smiles_list: list[str]) -> int:
    """파싱되지 않는 SMILES 개수. 데이터 점검용."""
    return sum(1 for s in smiles_list if to_mol(s) is None)
