"""카이랄 fingerprint 재실험: Morgan `includeChirality` False vs True.

  python scripts/run_chirality.py

**왜 이 실험을 하는가.** 3주차에서 프로프라놀롤 PPBR 예측을 하다가, 지금
특징 추출이 R체와 S체를 **완전히 같은 입력**으로 본다는 것이 드러났다
(`pbpk/predict_admet.py` 참고). Morgan fingerprint가 `includeChirality=False`
이고 descriptor 27개도 전부 비카이랄이기 때문이다. 입체선택적 단백결합·
대사는 실제로 존재하므로, 이건 모델이 못 맞히는 게 아니라 **원리적으로 볼
수 없는** 종류의 한계다. 그래서 카이랼 비트를 켜면 나아지는지 확인한다.

**두 가지를 같이 본다.**

1. 지표가 움직이는가 (`results/chirality.csv`)
2. 움직일 수 있는 데이터가 애초에 얼마나 있는가 (`results/chirality_audit.csv`)

2번이 없으면 1번을 해석할 수 없다. 카이랄 비트가 구분력을 얻는 것은 같은
골격의 입체이성질체가 **둘 이상 데이터에 들어 있을 때**뿐이다. 하나만
있으면 입체정보가 적혀 있어도 비교 대상이 없어서 배울 것이 없고, 비트만
흩어진다(같은 부분구조가 R/S로 쪼개져 비트 공유가 줄어든다).

리더보드 숫자(`results/baseline.csv`, `results/models.csv`)는 건드리지
않는다. config의 `MORGAN_CHIRALITY` 기본값도 False로 남겨 두고, 여기서만
명시적으로 True를 넘긴다. 공정한 비교를 위해 False 쪽도 이 실행에서 다시
돌린다(저장된 CSV를 옮겨 적지 않는다).

특징 조합을 둘 다 보는 이유: `desc+fp`는 1주차 최적 조합이지만 비카이랄
descriptor 27개가 섞여 효과가 희석된다. `fp`만 쓰면 스위치의 효과가
그대로 보인다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import run_task  # noqa: E402
from src.config import RESULTS_DIR, SEEDS, TASKS  # noqa: E402
from src.data import get_group, load_task, xy  # noqa: E402
from src.stereo import (  # noqa: E402
    canonical, has_stereo, skeleton_key, stereo_groups,
)


def audit_task(group, task_name: str) -> dict:
    """카이랄 비트가 원리적으로 쓸 수 있는 데이터가 얼마나 되는지 센다.

    seed 1의 split을 쓴다. 분자 구성은 seed마다 섞이는 위치만 다르고
    전체 집합은 같으므로, 전체 비율을 세는 데는 어느 seed든 같다.
    """
    split = load_task(group, task_name, seed=SEEDS[0])
    parts = {}
    for name in ("train", "valid", "test"):
        smiles, ys = xy(split[name])
        parts[name] = (list(smiles), list(ys.to_numpy()))

    all_smiles = parts["train"][0] + parts["valid"][0] + parts["test"][0]
    groups = stereo_groups(all_smiles)
    # 입체이성질체가 둘 이상 등장한 골격 = 카이랄 비트가 구분력을 얻는 곳
    multi = {k for k, v in groups.items() if len(v) > 1}

    n_stereo = sum(1 for s in all_smiles if has_stereo(s))
    test_smiles = parts["test"][0]
    test_in_multi = sum(1 for s in test_smiles if skeleton_key(s) in multi)

    # **학습 경로가 실제로 열려 있는가.**
    #
    # TDC admet_group은 scaffold split이고, 거울상은 Bemis-Murcko 골격이
    # 같다. 그러면 입체이성질체 짝이 같은 split에 떨어지기 쉽다. test
    # 분자의 짝이 test 안에만 있으면 모델은 그 짝을 학습에서 본 적이
    # 없고, 카이랄 비트를 켜도 그 분자에 대해 배운 것이 없다.
    #
    # 그래서 "test 분자 중, 같은 골격의 **다른** 입체이성질체가
    # train/valid에 있는" 개수를 따로 센다. 이것이 카이랄 비트가 그
    # 분자에 대해 쓸 수 있는 정보를 가진 경우의 수다.
    train_canon: dict[str, set[str]] = {}
    for part in ("train", "valid"):
        for s in parts[part][0]:
            k, c = skeleton_key(s), canonical(s)
            if k is not None and c is not None:
                train_canon.setdefault(k, set()).add(c)

    test_with_partner_in_train = 0
    for s in test_smiles:
        k, c = skeleton_key(s), canonical(s)
        if k is None or c is None:
            continue
        others = train_canon.get(k, set()) - {c}
        if others:
            test_with_partner_in_train += 1

    # 그 골격들 안에서 Y가 실제로 얼마나 벌어지는가. 벌어지지 않으면
    # 구분할 수 있게 되어도 구분할 가치가 없다.
    by_key: dict[str, list[float]] = {}
    for smiles, ys in parts.values():
        for s, y in zip(smiles, ys):
            k = skeleton_key(s)
            if k in multi:
                by_key.setdefault(k, []).append(float(y))
    spreads = [max(v) - min(v) for v in by_key.values() if len(v) > 1]

    return {
        "task": task_name,
        "n_total": len(all_smiles),
        "n_with_stereo": n_stereo,
        "pct_with_stereo": round(100 * n_stereo / len(all_smiles), 1),
        "n_skeletons": len(groups),
        "n_skeletons_multi_isomer": len(multi),
        "pct_skeletons_multi_isomer": round(100 * len(multi) / len(groups), 1),
        "n_test": len(test_smiles),
        "n_test_in_multi_isomer_skeleton": test_in_multi,
        "pct_test_in_multi_isomer_skeleton": round(
            100 * test_in_multi / len(test_smiles), 1
        ),
        "n_test_with_stereo_partner_in_train": test_with_partner_in_train,
        "pct_test_with_stereo_partner_in_train": round(
            100 * test_with_partner_in_train / len(test_smiles), 2
        ),
        "median_Y_spread_within_skeleton": (
            round(float(np.median(spreads)), 3) if spreads else float("nan")
        ),
        "max_Y_spread_within_skeleton": (
            round(float(np.max(spreads)), 3) if spreads else float("nan")
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--models", nargs="+", default=["rf"])
    ap.add_argument("--features", nargs="+", default=["fp", "desc+fp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--out", default=str(RESULTS_DIR / "chirality.csv"))
    ap.add_argument("--audit-only", action="store_true",
                    help="데이터 점검만 하고 재학습은 건너뛴다")
    args = ap.parse_args()

    group = get_group()

    # --- 1. 데이터 점검 -----------------------------------------------------
    print("=" * 78)
    print("1. 입체화학 함량 점검 — 카이랄 비트가 쓸 수 있는 데이터가 있는가")
    print("=" * 78)
    audit = pd.DataFrame([audit_task(group, t) for t in args.tasks])
    for _, r in audit.iterrows():
        print(f"\n  {r['task']}  (n={r['n_total']})")
        print(f"    입체정보 적힌 분자          {r['n_with_stereo']:5d}"
              f" ({r['pct_with_stereo']}%)")
        print(f"    골격 그룹                   {r['n_skeletons']:5d}")
        print(f"    그중 입체이성질체 2개 이상  "
              f"{r['n_skeletons_multi_isomer']:5d}"
              f" ({r['pct_skeletons_multi_isomer']}%)   <- 여기서만 효과가 난다")
        print(f"    test 분자 중 해당 골격      "
              f"{r['n_test_in_multi_isomer_skeleton']:5d} / {r['n_test']}"
              f" ({r['pct_test_in_multi_isomer_skeleton']}%)")
        print(f"    그중 짝이 train/valid에 있음 "
              f"{r['n_test_with_stereo_partner_in_train']:5d} / {r['n_test']}"
              f" ({r['pct_test_with_stereo_partner_in_train']}%)"
              f"   <- 실제로 배울 수 있는 경우")
        print(f"    그 골격 내 Y 격차 중앙값/최대 "
              f"{r['median_Y_spread_within_skeleton']}"
              f" / {r['max_Y_spread_within_skeleton']}")

    audit_path = Path(args.out).with_name("chirality_audit.csv")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    if args.audit_only:
        print(f"\n저장: {audit_path}  (--audit-only, 재학습 건너뜀)")
        return

    # --- 2. 재학습 ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. includeChirality False vs True  (seed "
          f"{len(args.seeds)}개, TDC 공식 평가)")
    print("=" * 78)
    rows = []
    for task in args.tasks:
        metric = TASKS[task]["metric"]
        for model in args.models:
            for feat in args.features:
                for chir in (False, True):
                    started = time.time()
                    mean, std = run_task(
                        group, task, model, feat, args.seeds,
                        include_chirality=chir,
                    )
                    elapsed = time.time() - started
                    print(f"  {task:24} {model:5} {feat:8} "
                          f"chirality={str(chir):5} "
                          f"{metric}={mean:.4f} +/- {std:.4f}  ({elapsed:.0f}s)")
                    rows.append({
                        "task": task, "model": model, "features": feat,
                        "include_chirality": chir, "metric": metric,
                        "mean": round(mean, 4), "std": round(std, 4),
                        "n_seeds": len(args.seeds), "seconds": round(elapsed, 1),
                    })

    df = pd.DataFrame(rows)

    # --- 3. 나란히 비교 -----------------------------------------------------
    print("\n" + "=" * 78)
    print("3. 비교 (delta는 True - False, improved는 과제 지표 방향을 반영)")
    print("=" * 78)
    wide_rows = []
    for (task, model, feat), g in df.groupby(["task", "model", "features"],
                                             sort=False):
        off = g[~g["include_chirality"]].iloc[0]
        on = g[g["include_chirality"]].iloc[0]
        higher_better = TASKS[task]["higher_is_better"]
        delta = on["mean"] - off["mean"]
        improved = delta > 0 if higher_better else delta < 0
        # 개선폭을 seed 변동 대비로 본다. 1 SD 안쪽이면 구분이 안 되는 것이다.
        pooled_sd = float(np.sqrt((off["std"] ** 2 + on["std"] ** 2) / 2))
        wide_rows.append({
            "task": task, "model": model, "features": feat,
            "metric": off["metric"],
            "chirality_off": off["mean"], "off_std": off["std"],
            "chirality_on": on["mean"], "on_std": on["std"],
            "delta": round(delta, 4),
            "delta_in_sd": (round(abs(delta) / pooled_sd, 2)
                            if pooled_sd > 0 else float("nan")),
            "improved": bool(improved),
        })
    wide = pd.DataFrame(wide_rows)
    print(wide.to_string(index=False))

    wide_path = Path(args.out).with_name("chirality_comparison.csv")
    wide.to_csv(wide_path, index=False)
    out = Path(args.out)
    df.to_csv(out, index=False)

    print(f"\n저장: {out}")
    print(f"저장: {wide_path}")
    print(f"저장: {audit_path}")
    print("\n주의: results/baseline.csv 와 results/models.csv 는 건드리지 않았다"
          " (리더보드 비교 기준선).")


if __name__ == "__main__":
    main()
