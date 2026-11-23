"""입체화학 관련 공용 도구.

"입체를 무시하면 같은 분자"라는 개념이 이 프로젝트에서 두 군데 쓰인다.

- `pbpk/predict_admet.py`: 프로프라놀롤이 train에 있는지 확인할 때.
  거울상이 들어 있어도 같은 분자로 잡아야 누출을 놓치지 않는다.
- `scripts/run_chirality.py`: 카이랄 fingerprint가 **원리적으로** 도움이
  될 수 있는 데이터가 얼마나 되는지 셀 때.

정의가 두 벌 있으면 한쪽만 고쳐질 수 있으므로 여기 모아 둔다.
"""
from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def skeleton_key(smiles: str) -> str | None:
    """입체화학을 무시한 분자 식별자. 파싱 실패면 None.

    InChIKey의 첫 블록은 원자 연결만 반영하고 입체는 두 번째 블록에
    들어간다. 그래서 첫 블록만 비교하면 거울상끼리 같게 잡힌다.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol).split("-")[0]


def has_stereo(smiles: str) -> bool:
    """지정된 입체중심이나 이중결합 입체가 하나라도 있는지.

    `includeUnassigned=False`가 핵심이다. SMILES에 `@`가 적혀 있지 않은
    입체중심은 "미지정"이고, 그 분자는 카이랄 fingerprint를 켜도 비트가
    달라지지 않는다. 즉 여기서 세는 것은 '입체중심을 가질 수 있는 분자'가
    아니라 '입체가 실제로 적혀 있는 분자'다.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=False, useLegacyImplementation=False
    )
    if centers:
        return True
    return any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())


def canonical(smiles: str) -> str | None:
    """입체를 포함한 정규 SMILES. 같은 분자의 다른 표기를 합치는 데 쓴다."""
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def stereo_groups(smiles_list: list[str]) -> dict[str, set[str]]:
    """골격(입체 무시) -> 그 골격으로 등장한 서로 다른 입체 SMILES 집합.

    값의 크기가 2 이상인 항목이, 카이랄 fingerprint가 비로소 구분할 수
    있게 되는 분자들이다. 1이면 입체정보가 있어도 비교 대상이 없어서
    모델이 그 비트에서 배울 것이 없다.
    """
    groups: dict[str, set[str]] = {}
    for smi in smiles_list:
        key = skeleton_key(smi)
        if key is None:
            continue
        can = canonical(smi)
        if can is None:
            continue
        groups.setdefault(key, set()).add(can)
    return groups
