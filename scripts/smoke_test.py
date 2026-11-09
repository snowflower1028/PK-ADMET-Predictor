"""네트워크 없이 돌아가는 점검.

TDC 데이터를 내려받지 않고도 RDKit 특징 추출과 모델 학습이 되는지 본다.
설치가 끝났는지 확인할 때 가장 먼저 쓰면 좋다 (check_env.py 다음).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features  # noqa: E402
from src.baseline import make_model  # noqa: E402
from src.utils import set_seed  # noqa: E402

SMILES = [
    "CC(C)NCC(O)COc1cccc2ccccc12",                      # propranolol
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                       # caffeine
    "CC(=O)Nc1ccc(O)cc1",                               # acetaminophen
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",                       # ibuprofen
    "CN1CCC[C@H]1c1cccnc1",                             # nicotine
    "OC(=O)c1ccccc1OC(C)=O",                            # aspirin
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",                       # procainamide
    "Clc1ccccc1C1=NCC(=O)Nc2ccc(Cl)cc21",               # 벤조디아제핀 계열
    "CC12CCC(=O)C=C1CCC1C2CCC2(C)C1CCC2=O",             # 스테로이드 골격
    "C1CC1C(",                                          # 일부러 깨뜨린 SMILES
]


def main() -> int:
    set_seed(1)
    ok = True

    n_bad = features.n_invalid(SMILES)
    print(f"파싱 실패 개수 : {n_bad} (1이어야 정상 — 마지막 SMILES를 일부러 깨뜨렸다)")
    ok &= n_bad == 1

    desc = features.descriptors(SMILES)
    print(f"descriptor 모양 : {desc.shape}")
    ok &= desc.shape == (len(SMILES), len(features.DESCRIPTOR_NAMES))
    ok &= bool(np.isfinite(desc).all())
    print(f"무한값·결측 없음 : {bool(np.isfinite(desc).all())}")

    fp = features.morgan(SMILES)
    print(f"fingerprint 모양 : {fp.shape}, 켜진 비트 합 {int(fp.sum())}")
    ok &= fp.shape[1] == 2048

    both = features.featurize(SMILES, "desc+fp")
    print(f"desc+fp 모양     : {both.shape}")
    ok &= both.shape[1] == desc.shape[1] + fp.shape[1]

    # 모델이 학습되고 예측 모양이 맞는지만 본다 (정확도는 의미 없다)
    y = np.arange(len(SMILES), dtype=float)
    for kind in ("rf", "lgbm"):
        try:
            model = make_model(kind, seed=1)
            model.fit(both, y)
            pred = model.predict(both)
            print(f"{kind:5} 학습·예측  : 예측 {pred.shape}, 평균 {pred.mean():.2f}")
            ok &= pred.shape == y.shape
        except ImportError as exc:
            print(f"{kind:5} 건너뜀      : {exc}")

    print("\n통과" if ok else "\n실패 — 위 항목을 확인한다")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
