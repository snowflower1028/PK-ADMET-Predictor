"""알려진 약물을 넣어 예측이 합당한지 빠르게 보는 점검 스크립트.

  python scripts/run_sanity.py
  python scripts/run_sanity.py --smiles "Aspirin=CC(=O)Oc1ccccc1C(=O)O"

**왜 필요한가.** 지표(MAE, Spearman)는 test set 전체의 평균이라 "이 모델이
아는 약물 하나를 넣으면 말이 되는 값을 내놓는가"를 말해 주지 않는다.
리뷰나 면접에서 가장 먼저 받는 질문이 그것이고, 3주차에 프로프라놀롤
하나를 넣어 본 것이 실제로 문제 두 개를 잡아냈다(train 누출 4건,
a priori Cmax 5배 오차).

**이 점검의 핵심은 누출 검사와 한 쌍이라는 것이다.**

대상 분자가 train에 들어 있으면 모델은 외운 값을 돌려준다. 그걸 "잘
맞혔다"고 읽으면 아무 의미가 없다. 그래서 여기서는

  1. 먼저 입체화학을 무시한 골격(InChIKey 첫 블록)으로 데이터셋을 뒤져
     대상 분자가 어디에 있는지 확인하고,
  2. **sanity set 전체를 train/valid에서 빼고** 다시 학습한 뒤,
  3. 데이터셋이 들고 있던 그 분자의 라벨을 **정답으로 삼아** 비교한다.

즉 leave-set-out 방식이다. 문헌값을 따로 들고 오지 않는다. 외부 숫자를
손으로 적으면 그 숫자가 맞는지부터 증명해야 하는데, 데이터셋 자신의
라벨을 쓰면 그 문제가 사라진다. 분자가 데이터셋에 아예 없으면 정답이
없으므로 '참조 없음'으로 표시하고 예측값의 분포상 위치만 보고한다.

seed마다 sanity set 전체를 한 번만 빼고 학습한다(분자마다 따로 빼지
않는다). 분자 8개를 빼는 것은 900~2800개 중 수십 행이라 영향이 작고,
개별 leave-one-out보다 오히려 보수적이다. 대신 학습 횟수가
3과제 x 5seed = 15회로 줄어 몇 분 안에 끝난다.

SMILES 출처: ChEMBL (아래 CHEMBL ID). 이름과 구조를 기억으로 적지
않는다 — 구조를 잘못 라벨링하면 점검 자체가 거짓이 된다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem, RDLogger  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from pbpk.predict_admet import BEST_FEATURES  # noqa: E402
from src.config import RESULTS_DIR, SEEDS, TASKS  # noqa: E402
from src.data import get_group, load_task, xy  # noqa: E402
from src.features import featurize  # noqa: E402
from src.stereo import skeleton_key  # noqa: E402
from src.utils import set_seed  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# (이름, SMILES, 출처, 고른 이유)
# 단백결합이 낮은 쪽부터 높은 쪽까지 깔리도록 골랐다. 전부 비슷한 약물이면
# 모델이 평균만 내놓아도 다 맞는 것처럼 보인다.
SANITY_SET = [
    ("Atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
     "CHEMBL24", "친수성 베타차단제 — 투과도 낮고 단백결합 낮음"),
    ("Metoprolol", "COCCc1ccc(OCC(O)CNC(C)C)cc1",
     "CHEMBL13", "아테놀롤과 같은 계열이지만 더 친유성 — 쌍 비교용"),
    ("Propranolol", "CC(C)NCC(O)COc1cccc2ccccc12",
     "CHEMBL27", "3주차 PBPK 대상 — PPBR train에 누출이 있던 분자"),
    ("Theophylline", "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
     "CHEMBL190", "청소율 낮은 약물 — 대사 과제의 아래쪽 끝"),
    ("Caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
     "CHEMBL113", "테오필린과 메틸기 하나 차이 — 구조 민감도 확인"),
    ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
     "CHEMBL12", "단백결합 매우 높음"),
    ("Warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
     "CHEMBL1464", "단백결합 극단값 — 위쪽 끝"),
    ("Verapamil", "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
     "CHEMBL6966", "고청소율·고초회통과 — 대사 과제의 위쪽 끝"),
]


def locate(group, task_name: str, keys: dict) -> dict:
    """각 분자가 데이터셋 어디에 있는지, 라벨이 무엇인지 찾는다.

    seed 1의 split을 쓴다. train/valid를 합치면 전체 train_val이 되고
    test는 고정이므로, '어디에 있는가'는 seed에 의존하지 않는다.
    """
    split = load_task(group, task_name, seed=SEEDS[0])
    found: dict = {}
    for part in ("train", "valid", "test"):
        smiles, ys = xy(split[part])
        for smi, y in zip(smiles, ys.to_numpy()):
            k = skeleton_key(smi)
            if k is None:
                continue
            for name, key in keys.items():
                if k == key:
                    rec = found.setdefault(
                        name, {"where": set(), "values": [], "rows": 0}
                    )
                    rec["where"].add("train_valid" if part != "test" else "test")
                    rec["values"].append(float(y))
                    rec["rows"] += 1
    return found


def run_task(group, task_name: str, mols: list, keys: dict, seeds: list,
             leave_one_out: bool = False):
    """sanity set 전체를 빼고 학습한 뒤 전원을 예측한다.

    leave_one_out=True면 분자마다 **그 분자만** 빼고 따로 학습한다.

    이 구분이 실제로 중요했다. 카페인과 테오필린은 메틸기 하나 차이라
    서로의 최근접 이웃인데, 기본 모드는 둘을 같이 빼므로 "이 계열을 못
    맞힌다"인지 "최근접 유사체에만 기대고 있었다"인지 섞인다. 하나씩만
    빼고 보면 갈린다(실제로 갈렸다 — docs/05 참고).

    비용: 분자 수 x seed 수만큼 학습하므로 기본 모드보다 분자 수배 느리다.

    반환: (이름 -> 예측 리스트, 데이터셋 Y 통계, 제외된 행 수)
    """
    if leave_one_out:
        return _run_task_loo(group, task_name, mols, keys, seeds)

    target_keys = set(keys.values())
    x_target = featurize([m[1] for m in mols], BEST_FEATURES)

    preds: dict = {m[0]: [] for m in mols}
    excluded = 0
    y_all: list = []

    for seed in seeds:
        set_seed(seed)
        split = load_task(group, task_name, seed)
        smiles, ys = [], []
        for part in ("train", "valid"):
            s, y = xy(split[part])
            smiles += list(s)
            ys += list(y.to_numpy())

        keep = [i for i, s in enumerate(smiles) if skeleton_key(s) not in target_keys]
        excluded = len(smiles) - len(keep)
        fit_smiles = [smiles[i] for i in keep]
        fit_y = np.asarray([ys[i] for i in keep], dtype=np.float64)
        if not y_all:
            y_all = list(ys)

        x = featurize(fit_smiles, BEST_FEATURES)
        model = RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
        model.fit(x, fit_y)
        for m, v in zip(mols, model.predict(x_target)):
            preds[m[0]].append(float(v))

    arr = np.asarray(y_all, dtype=np.float64)
    stats = {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "max": float(arr.max()),
        "values": arr,
    }
    return preds, stats, excluded


def _run_task_loo(group, task_name: str, mols: list, keys: dict, seeds: list):
    """분자마다 그 분자만 빼고 학습한다. run_task(leave_one_out=True)가 쓴다."""
    preds: dict = {m[0]: [] for m in mols}
    excluded_total, y_all = 0, []

    for name, smi, _, _ in mols:
        key = keys[name]
        x_target = featurize([smi], BEST_FEATURES)
        for seed in seeds:
            set_seed(seed)
            split = load_task(group, task_name, seed)
            smiles, ys = [], []
            for part in ("train", "valid"):
                s, y = xy(split[part])
                smiles += list(s)
                ys += list(y.to_numpy())
            if not y_all:
                y_all = list(ys)

            keep = [i for i, s in enumerate(smiles) if skeleton_key(s) != key]
            excluded_total = max(excluded_total, len(smiles) - len(keep))
            x = featurize([smiles[i] for i in keep], BEST_FEATURES)
            model = RandomForestRegressor(
                n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed
            )
            model.fit(x, np.asarray([ys[i] for i in keep], dtype=np.float64))
            preds[name].append(float(model.predict(x_target)[0]))

    arr = np.asarray(y_all, dtype=np.float64)
    stats = {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "min": float(arr.min()), "max": float(arr.max()),
        "values": arr,
    }
    return preds, stats, excluded_total


def verdict(err, sd: float) -> str:
    """정답 대비 오차를 데이터셋 표준편차로 재서 판정한다.

    과제마다 단위가 달라(log 단위, %, uL/min/mg) 절대 오차를 그대로
    비교할 수 없다. 표준편차로 나누면 '이 과제 기준으로 큰 오차인가'를
    같은 눈으로 볼 수 있다.
    """
    if err is None:
        return "참조 없음"
    if sd <= 0:
        return "판정 불가"
    z = err / sd
    if z <= 1.0:
        return "합당"
    if z <= 2.0:
        return "경계"
    return "확인 필요"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--smiles", action="append", default=[],
                    help='추가 분자. "이름=SMILES" 형식. 여러 번 쓸 수 있다')
    ap.add_argument("--leave-one-out", action="store_true",
                    help="분자마다 그 분자만 빼고 따로 학습한다 (분자 수배 느리다). "
                         "최근접 유사체에 기대고 있는지 가리려면 이쪽을 쓴다")
    ap.add_argument("--out", default=str(RESULTS_DIR / "sanity_check.csv"))
    args = ap.parse_args()

    mols = list(SANITY_SET)
    for item in args.smiles:
        if "=" not in item:
            raise SystemExit('--smiles 는 "이름=SMILES" 형식이어야 한다: ' + item)
        name, smi = item.split("=", 1)
        mols.append((name.strip(), smi.strip(), "사용자 입력", "명령행에서 추가"))

    # --- 0. 구조 확인 -------------------------------------------------------
    print("=" * 78)
    print("0. 구조 확인 (SMILES가 파싱되는지, 어떤 분자인지)")
    print("=" * 78)
    keys: dict = {}
    for name, smi, src, why in mols:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise SystemExit("SMILES를 파싱할 수 없다: " + name + " " + smi)
        keys[name] = skeleton_key(smi)
        print("  {:14} {:12} {}".format(name, src, Chem.MolToInchiKey(mol)))
        print("  {:14} {}".format("", why))

    group = get_group()
    rows = []

    for task in args.tasks:
        print("\n" + "=" * 78)
        print("{}  —  {}".format(task, TASKS[task]["label"]))
        print("=" * 78)

        where = locate(group, task, keys)
        started = time.time()
        preds, stats, excluded = run_task(
            group, task, mols, keys, args.seeds, args.leave_one_out)
        elapsed = time.time() - started

        print("  데이터셋 Y: 평균 {:.3f}, 표준편차 {:.3f}, 범위 {:.3f} ~ {:.3f}".format(
            stats["mean"], stats["std"], stats["min"], stats["max"]))
        mode = "분자마다 따로" if args.leave_one_out else "sanity set 전체"
        print("  학습에서 제외한 행: {}개 ({})".format(excluded, mode))
        print("  seed {}개 학습 {:.0f}s\n".format(len(args.seeds), elapsed))

        head = "  {:13}{:13}{:>9}{:>10}{:>8}{:>9}{:>7}  판정".format(
            "분자", "위치", "정답", "예측", "±", "오차", "/SD")
        print(head)
        print("  " + "-" * 76)

        for name, smi, src, _ in mols:
            p = np.asarray(preds[name])
            rec = where.get(name)
            if rec:
                ref = float(np.mean(rec["values"]))
                loc = "+".join(sorted(rec["where"]))
                if rec["rows"] > 1:
                    loc = loc + "({})".format(rec["rows"])
                err = abs(float(p.mean()) - ref)
                z = err / stats["std"] if stats["std"] > 0 else float("nan")
                ref_s, err_s, z_s = "{:9.3f}".format(ref), "{:9.3f}".format(err), "{:7.2f}".format(z)
            else:
                ref, loc, err, z = None, "없음", None, None
                ref_s, err_s, z_s = "        -", "        -", "      -"

            v = verdict(err, stats["std"])
            pct = float((stats["values"] < p.mean()).mean() * 100)

            print("  {:13}{:13}{}{:10.3f}{:8.3f}{}{}  {}".format(
                name, loc, ref_s, p.mean(), p.std(), err_s, z_s, v))

            rows.append({
                "task": task, "molecule": name, "source": src, "smiles": smi,
                "located_in": loc,
                "reference_from_dataset": None if ref is None else round(ref, 4),
                "n_dataset_rows": 0 if not rec else rec["rows"],
                "prediction": round(float(p.mean()), 4),
                "prediction_std": round(float(p.std()), 4),
                "abs_error": None if err is None else round(err, 4),
                "error_in_dataset_sd": None if z is None else round(z, 3),
                "pred_percentile_in_dataset": round(pct, 1),
                "dataset_mean": round(stats["mean"], 4),
                "dataset_std": round(stats["std"], 4),
                "verdict": v, "n_seeds": len(args.seeds),
                "excluded_rows": excluded,
                "exclusion_mode": ("leave_one_out" if args.leave_one_out
                                   else "leave_set_out"),
            })

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("요약")
    print("=" * 78)
    scored = df[df["abs_error"].notna()]
    print("  정답이 있는 경우 {} / {}".format(len(scored), len(df)))
    if len(scored):
        counts = scored["verdict"].value_counts()
        for k in ("합당", "경계", "확인 필요"):
            print("    {:8} {}".format(k, int(counts.get(k, 0))))
        print("  오차 중앙값 {:.2f} SD".format(scored["error_in_dataset_sd"].median()))
    print("\n저장: {}".format(out))
    print("\n주의: 모든 예측은 해당 분자를 train/valid에서 빼고 다시 학습해")
    print("      얻은 값이다. 리더보드 숫자(results/baseline.csv, models.csv)와는")
    print("      다른 학습이며, 그 CSV들은 건드리지 않는다.")


if __name__ == "__main__":
    main()
