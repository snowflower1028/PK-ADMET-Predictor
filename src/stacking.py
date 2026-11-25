"""4주차 스태킹: RF/LightGBM의 예측값을 GIN의 입력 특징으로 쓴다.

흐름은 2단이다.

    1단: SMILES -> descriptor(27) + Morgan fingerprint(2048) -> RF, LightGBM
    2단: 1단의 예측값 2개 + 원자/결합 그래프 + descriptor -> GIN -> 최종 예측

이 파일은 1단만 담당한다. 돌려주는 것은 train/valid/test 각 구간에 대한
(분자 수, 기반모델 수) 행렬이고, 열은 **원래 타깃 단위의 예측값**이다.
타깃 정규화는 2단(train.py)에서 타깃과 같은 스케일러로 한다.

-----------------------------------------------------------------------------
이 파일 전체가 존재하는 이유: **정보 누출**
-----------------------------------------------------------------------------

RF를 train 전체에 학습시킨 뒤 그 train을 다시 예측해 특징으로 쓰면 안 된다.
RF(min_samples_leaf=2)는 train을 거의 외우므로 그 열은 정답에 아주 가깝다.
GIN은 "그 열만 베끼면 손실이 0"을 배우고, 그래프와 descriptor는 쳐다보지
않게 된다. 그런데 test에서는 같은 열이 훨씬 부정확하다. 학습할 때 본 특징과
예측할 때 받는 특징의 **분포가 다르다**(train에서는 거의 정답, test에서는
노이즈). 결과는 test 성능 붕괴다.

막는 방법이 out-of-fold(OOF)다. train을 K조각으로 나누고, 조각 k의 특징은
**조각 k를 보지 않은** 모델로 예측한다. 그러면 train 구간의 특징도 test
구간과 같은 "처음 보는 분자에 대한 예측" 성격을 갖는다.

    mode="oof"    : train 특징 = K-fold out-of-fold 예측      (올바른 방식)
    mode="infold" : train 특징 = train 전체 학습 후 자기 예측 (누출, 대조군)

누출 버전을 지우지 않고 남겨 둔 이유는, 둘을 같은 조건으로 돌려 숫자 차이를
보여 주는 것이 "누출을 막았다"는 말보다 강한 증거이기 때문이다.

valid/test 특징은 두 모드 모두 **train 전체에 학습한 모델**로 예측한다.
valid/test는 1단 학습에 쓰이지 않으므로 여기에는 누출이 없다.

-----------------------------------------------------------------------------
1주차 베이스라인과 학습 데이터가 다르다 (비교할 때 반드시 같이 말해야 함)
-----------------------------------------------------------------------------

results/baseline.csv의 RF/LGBM은 train+valid에 학습했다. 트리 모델은 early
stopping을 쓰지 않아 valid를 남겨 둘 이유가 없었기 때문이다.

여기서는 valid를 쓸 수 없다. 2단 GIN이 valid로 early stopping을 하므로,
1단이 valid를 학습에 쓰면 GIN의 모델 선택 자체가 오염된다. 그래서 1단은
**train만** 쓴다. 데이터가 적으니 1주차보다 조금 약한 기반모델이 되고,
그래서 `base_test_pred`를 같이 돌려준다 — 같은 train-only 조건에서 기반모델
혼자 받는 점수. 스태킹이 보탬이 됐는지는 1주차 숫자가 아니라 이것과 비교해야
한다.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import KFold

from .baseline import make_model as make_tree_model
from .config import RESULTS_DIR
from .features import featurize
from .utils import set_seed

# 1단에 쓸 기반모델. 1주차 그리드에서 세 과제 모두 1·2위였던 조합이다
# (results/baseline.csv: rf desc+fp가 세 과제 전부 1위, lgbm desc+fp가
# Caco2/PPBR 2위). 그래서 특징도 desc+fp로 고정한다.
META_MODELS = ("rf", "lgbm")
META_FEATURES = "desc+fp"
N_FOLDS = 5

CACHE_DIR = RESULTS_DIR / "meta_cache"


@dataclass
class MetaFeatures:
    """1단 출력. 열 값은 전부 **원래 타깃 단위**다."""

    train: np.ndarray  # (train 분자 수, 기반모델 수)
    valid: np.ndarray
    test: np.ndarray
    columns: list[str]
    mode: str
    # 진단용. train 구간 특징이 실제로 얼마나 정확한지. oof와 infold의
    # 차이가 여기서 먼저 드러난다(infold는 비현실적으로 좋게 나온다).
    train_fit: dict = field(default_factory=dict)
    # 같은 train-only 조건에서 기반모델 혼자 낸 test 예측. 모델별로 하나씩.
    base_test_pred: dict = field(default_factory=dict)
    seconds: float = 0.0
    from_cache: bool = False

    @property
    def n_meta(self) -> int:
        return int(self.train.shape[1])


def _data_key(tr_smiles, tr_y, va_smiles, te_smiles) -> str:
    """캐시가 다른 데이터에 붙는 것을 막는 해시.

    (task, seed)만으로 키를 만들면 TDC 데이터가 갱신됐을 때 조용히 옛 특징을
    쓰게 된다. 입력 자체를 해시해 두면 그 경우 캐시가 빗나가 다시 계산한다.
    """
    h = hashlib.md5()
    for part in (tr_smiles, va_smiles, te_smiles):
        h.update("|".join(part).encode("utf-8"))
        h.update(b"\x00")
    h.update(np.asarray(tr_y, dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


def _fit_predict(model_kind: str, seed: int, x_fit, y_fit, *x_predict):
    """기반모델 하나를 학습하고 주어진 행렬들을 차례로 예측한다."""
    model = make_tree_model(model_kind, seed)
    model.fit(x_fit, y_fit)
    return [np.asarray(model.predict(x), dtype=np.float64) for x in x_predict]


def _score_fit(y_true, y_pred) -> dict:
    """train 구간 특징의 정확도. 단위에 안 끌리게 표준편차로도 재 둔다."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    sd = float(y_true.std())
    return {
        "mae": round(mae, 4),
        # 데이터 표준편차로 나눈 오차. 0에 가까우면 거의 외운 것이다.
        "mae_in_sd": round(mae / sd, 4) if sd > 0 else float("nan"),
    }


def build_meta(
    task_name: str,
    seed: int,
    tr_smiles: list[str],
    tr_y: np.ndarray,
    va_smiles: list[str],
    te_smiles: list[str],
    mode: str = "oof",
    n_folds: int = N_FOLDS,
    models=META_MODELS,
    cache: bool = True,
    verbose: bool = False,
) -> MetaFeatures:
    """1단 특징을 만든다.

    mode="oof"    : train 특징을 K-fold out-of-fold 예측으로 만든다.
    mode="infold" : train 전체로 학습해 그 train을 예측한다(누출 대조군).

    두 모드 모두 valid/test는 train 전체에 학습한 모델로 예측한다.
    """
    if mode not in ("oof", "infold"):
        raise ValueError(f"알 수 없는 mode: {mode}")
    models = tuple(models)
    tr_y = np.asarray(tr_y, dtype=np.float64)

    key = _data_key(tr_smiles, tr_y, va_smiles, te_smiles)
    cache_path = (
        CACHE_DIR
        / f"{task_name}_seed{seed}_{mode}_k{n_folds}_{'-'.join(models)}_{key}.npz"
    )
    if cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return MetaFeatures(
            train=z["train"], valid=z["valid"], test=z["test"],
            columns=list(z["columns"]), mode=mode,
            train_fit=z["train_fit"].item(),
            base_test_pred=z["base_test_pred"].item(),
            seconds=float(z["seconds"]), from_cache=True,
        )

    started = time.time()
    # 특징 추출은 한 번만. RDKit 호출이 fold 수만큼 반복되면 그게 더 느리다.
    x_tr = featurize(tr_smiles, META_FEATURES)
    x_va = featurize(va_smiles, META_FEATURES)
    x_te = featurize(te_smiles, META_FEATURES)

    train_meta = np.zeros((x_tr.shape[0], len(models)), dtype=np.float64)
    valid_meta = np.zeros((x_va.shape[0], len(models)), dtype=np.float64)
    test_meta = np.zeros((x_te.shape[0], len(models)), dtype=np.float64)
    train_fit: dict = {}
    base_test_pred: dict = {}

    for j, kind in enumerate(models):
        # 기반모델도 seed를 고정한다. 같은 seed면 같은 특징이 나와야 한다.
        set_seed(seed)

        # --- valid/test: train 전체에 학습한 모델로 (누출 없음) -------------
        va_pred, te_pred = _fit_predict(kind, seed, x_tr, tr_y, x_va, x_te)
        valid_meta[:, j] = va_pred
        test_meta[:, j] = te_pred
        base_test_pred[kind] = te_pred

        # --- train: 모드에 따라 갈린다 --------------------------------------
        if mode == "infold":
            # 누출 대조군. 자기가 학습한 데이터를 자기가 예측한다.
            (tr_pred,) = _fit_predict(kind, seed, x_tr, tr_y, x_tr)
            train_meta[:, j] = tr_pred
        else:
            # KFold도 seed로 고정한다. 모델별로 같은 분할을 써야 두 열이
            # 같은 조건에서 나온 값이 된다.
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold, (fit_idx, out_idx) in enumerate(kf.split(x_tr), start=1):
                (pred,) = _fit_predict(
                    kind, seed, x_tr[fit_idx], tr_y[fit_idx], x_tr[out_idx]
                )
                train_meta[out_idx, j] = pred
                if verbose:
                    print(f"      {kind} fold {fold}/{n_folds} "
                          f"({len(fit_idx)} fit -> {len(out_idx)} pred)", flush=True)

        train_fit[kind] = _score_fit(tr_y, train_meta[:, j])
        if verbose:
            print(f"    {kind:5} train 특징 MAE={train_fit[kind]['mae']:.4f} "
                  f"({train_fit[kind]['mae_in_sd']:.3f} SD)", flush=True)

    out = MetaFeatures(
        train=train_meta, valid=valid_meta, test=test_meta,
        columns=[f"pred_{m}" for m in models], mode=mode,
        train_fit=train_fit, base_test_pred=base_test_pred,
        seconds=time.time() - started, from_cache=False,
    )

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            train=out.train, valid=out.valid, test=out.test,
            columns=np.asarray(out.columns),
            train_fit=np.asarray(out.train_fit, dtype=object),
            base_test_pred=np.asarray(out.base_test_pred, dtype=object),
            seconds=np.asarray(out.seconds),
        )
    return out
