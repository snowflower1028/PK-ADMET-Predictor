"""예시 분자를 넣어 2단 스태킹 예측이 합당한지 본다.

  python scripts/run_stack_demo.py
  python scripts/run_stack_demo.py --tasks PPBR_AZ --seeds 1
  python scripts/run_stack_demo.py --smiles "Ibuprofen=CC(C)Cc1ccc(cc1)C(C)C(=O)O"

**왜 필요한가.** MAE와 Spearman은 test set 전체의 평균이라, "이 모델에
약물 하나를 넣으면 말이 되는 값이 나오는가"를 말해 주지 않는다. 그리고
스태킹에서는 질문이 하나 더 붙는다 — **2단 GIN이 1단 예측을 실제로
고쳤는가, 아니면 그냥 베꼈는가.** 표의 평균 점수로는 둘을 구분할 수 없다.

그래서 분자마다 네 숫자를 나란히 찍는다.

    RF 예측 | LGBM 예측 | 둘의 평균(= 2단의 출발점) | 최종 예측

최종 예측과 평균의 차이가 **2단이 움직인 양**이다. 그 방향이 정답 쪽이면
2단이 보탬이 된 것이고, 반대면 망친 것이다. 분자별로 갈리는 게 정상이고,
그 분포를 보는 것이 이 스크립트의 목적이다.

**누출 점검은 run_sanity.py와 같은 방식이다.** 대상 분자가 train에 있으면
모델은 외운 값을 돌려주므로, 입체화학을 무시한 골격(InChIKey 첫 블록)으로
train/valid에서 전부 빼고 다시 학습한다. 정답은 데이터셋 자신의 라벨을
쓴다(문헌값을 손으로 적지 않는다 — 그 숫자가 맞는지부터 증명해야 한다).

분자 목록·구조·판정 기준은 scripts/run_sanity.py에서 가져와 쓴다. 같은
SMILES를 두 군데 적으면 한쪽만 고치는 사고가 난다.

seed 기본값이 3개인 것은 이 스크립트가 **빠른 점검**용이기 때문이다.
1단 OOF를 분자 제외된 train에 대해 새로 만들어야 해서 seed당 비용이 크다.
표에 싣는 숫자는 scripts/run_stack.py(5 seed)에서 나온 것을 쓴다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import Chem, RDLogger  # noqa: E402

from run_sanity import SANITY_SET, locate, verdict  # noqa: E402
from src.config import RESULTS_DIR, TASKS  # noqa: E402
from src.data import get_group, load_task, xy  # noqa: E402
from src.stack_predict import fit_predict_stack  # noqa: E402
from src.stereo import skeleton_key  # noqa: E402
from src.train import TrainConfig  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DEMO_SEEDS = [1, 2, 3]


def run_task(group, task_name, mols, keys, seeds, cfg, device, verbose):
    # cfg.residual_meta가 head 방식을 정한다(--head 옵션).
    """sanity set 전체를 train/valid에서 빼고 2단 파이프라인을 학습한다.

    반환: (분자 이름 -> {열 -> seed별 값 리스트}, 데이터셋 Y 통계, 제외 행 수)
    """
    target_keys = set(keys.values())
    target_smiles = [m[1] for m in mols]
    cols = ("rf", "lgbm", "base_mean", "gin+stack")

    preds: dict = {m[0]: {c: [] for c in cols} for m in mols}
    epochs: list = []
    excluded = 0
    y_all: list = []

    for seed in seeds:
        split = load_task(group, task_name, seed)

        kept: dict = {}
        for part in ("train", "valid"):
            smiles, ys = xy(split[part])
            smiles, ys = list(smiles), list(ys.to_numpy())
            if part == "train":
                y_all = list(ys)
            keep = [i for i, s in enumerate(smiles)
                    if skeleton_key(s) not in target_keys]
            excluded += len(smiles) - len(keep)
            kept[part] = ([smiles[i] for i in keep],
                          np.asarray([ys[i] for i in keep], dtype=np.float64))

        out = fit_predict_stack(
            task_name,
            kept["train"][0], kept["train"][1],
            kept["valid"][0], kept["valid"][1],
            target_smiles, seed, "gin+stack", cfg, device, verbose,
        )
        epochs.append(out["best_epoch"])
        for k, (name, *_rest) in enumerate(mols):
            for c in cols:
                preds[name][c].append(float(out[c][k]))
        if verbose:
            print(f"  seed {seed}: valid={out['valid_score']:+.4f}, "
                  f"best epoch {out['best_epoch']}", flush=True)

    arr = np.asarray(y_all, dtype=np.float64)
    stats = {"mean": float(arr.mean()), "std": float(arr.std()),
             "min": float(arr.min()), "max": float(arr.max()), "values": arr}
    return preds, stats, excluded // max(len(seeds), 1), epochs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=DEMO_SEEDS)
    ap.add_argument("--smiles", action="append", default=[],
                    help='추가 분자. "이름=SMILES" 형식. 여러 번 쓸 수 있다')
    ap.add_argument("--head", choices=["plain", "residual"], default="plain",
                    help="2단 head 방식. plain=concat 후 전부 학습, "
                         "residual=기반모델 평균에서 출발해 보정량만 학습. "
                         "기본값 plain은 results/stack.csv에서 더 나았던 쪽이다")
    ap.add_argument("--device", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_DIR / "stack_demo.csv"))
    args = ap.parse_args()

    mols = list(SANITY_SET)
    for item in args.smiles:
        if "=" not in item:
            raise SystemExit('--smiles 는 "이름=SMILES" 형식이어야 한다: ' + item)
        name, smi = item.split("=", 1)
        mols.append((name.strip(), smi.strip(), "사용자 입력", "명령행에서 추가"))

    print("=" * 94)
    print("0. 구조 확인")
    print("=" * 94)
    keys: dict = {}
    for name, smi, src, why in mols:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise SystemExit("SMILES를 파싱할 수 없다: " + name + " " + smi)
        keys[name] = skeleton_key(smi)
        print(f"  {name:14} {src:12} {Chem.MolToInchiKey(mol)}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(residual_meta=(args.head == "residual"))
    print(f"\ndevice: {device},  seeds: {args.seeds},  "
          f"모델: gin+stack (OOF 특징, {args.head} head)")

    group = get_group()
    rows = []

    for task in args.tasks:
        print("\n" + "=" * 94)
        print(f"{task}  —  {TASKS[task]['label']}")
        print("=" * 94)

        where = locate(group, task, keys)
        started = time.time()
        preds, stats, excluded, epochs = run_task(
            group, task, mols, keys, args.seeds, cfg, device, args.verbose)
        elapsed = time.time() - started

        print(f"  데이터셋 Y: 평균 {stats['mean']:.3f}, 표준편차 {stats['std']:.3f}, "
              f"범위 {stats['min']:.3f} ~ {stats['max']:.3f}")
        print(f"  train/valid에서 제외한 행: 약 {excluded}개 (sanity set 전체)")
        print(f"  seed {len(args.seeds)}개 학습 {elapsed:.0f}s, best epochs {epochs}\n")

        print(f"  {'분자':14}{'위치':14}{'정답':>9}{'RF':>9}{'LGBM':>9}"
              f"{'평균':>9}{'최종':>9}{'2단이동':>9}{'오차/SD':>9}  판정")
        print("  " + "-" * 90)

        for name, smi, src, _why in mols:
            p = {c: np.asarray(v) for c, v in preds[name].items()}
            final = float(p["gin+stack"].mean())
            base = float(p["base_mean"].mean())
            moved = final - base

            rec = where.get(name)
            if rec:
                ref = float(np.mean(rec["values"]))
                loc = "+".join(sorted(rec["where"]))
                if rec["rows"] > 1:
                    loc += f"({rec['rows']})"
                err = abs(final - ref)
                base_err = abs(base - ref)
                z = err / stats["std"] if stats["std"] > 0 else float("nan")
                ref_s, z_s = f"{ref:9.3f}", f"{z:9.2f}"
                # 2단이 정답 쪽으로 움직였는지. 이게 스태킹의 핵심 질문이다.
                helped = "개선" if err < base_err else "악화"
            else:
                ref = loc = err = base_err = z = None
                ref_s, z_s, helped = "        -", "        -", "-"
                loc = "없음"

            v = verdict(err, stats["std"])
            print(f"  {name:14}{loc:14}{ref_s}{p['rf'].mean():9.3f}"
                  f"{p['lgbm'].mean():9.3f}{base:9.3f}{final:9.3f}"
                  f"{moved:+9.3f}{z_s}  {v} / {helped}")

            rows.append({
                "task": task, "molecule": name, "source": src, "smiles": smi,
                "located_in": loc,
                "reference_from_dataset": None if ref is None else round(ref, 4),
                "n_dataset_rows": 0 if not rec else rec["rows"],
                "pred_rf": round(float(p["rf"].mean()), 4),
                "pred_lgbm": round(float(p["lgbm"].mean()), 4),
                "pred_base_mean": round(base, 4),
                "pred_stack": round(final, 4),
                "pred_stack_std": round(float(p["gin+stack"].std()), 4),
                "stage2_shift": round(moved, 4),
                "abs_error_base": None if base_err is None else round(base_err, 4),
                "abs_error_stack": None if err is None else round(err, 4),
                "stage2_effect": helped,
                "error_in_dataset_sd": None if z is None else round(z, 3),
                "dataset_mean": round(stats["mean"], 4),
                "dataset_std": round(stats["std"], 4),
                "head": args.head,
                "verdict": v, "n_seeds": len(args.seeds),
                "excluded_rows": excluded,
            })

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 94)
    print("요약")
    print("=" * 94)
    scored = df[df["abs_error_stack"].notna()]
    print(f"  정답이 있는 경우 {len(scored)} / {len(df)}")
    if len(scored):
        counts = scored["verdict"].value_counts()
        for k in ("합당", "경계", "확인 필요"):
            print(f"    {k:8} {int(counts.get(k, 0))}")
        # 과제마다 단위와 분포가 달라 절대 오차를 그냥 모을 수 없다. 행마다
        # 그 과제의 표준편차로 나눈 뒤 중앙값을 낸다.
        # (중앙값끼리 나누면 안 된다 — 분자와 분모가 서로 다른 행에서 온다.)
        z_base = (scored["abs_error_base"] / scored["dataset_std"])
        z_stack = (scored["abs_error_stack"] / scored["dataset_std"])
        print(f"  오차 중앙값: 1단 평균 {z_base.median():.2f} SD"
              f"  ->  2단 {z_stack.median():.2f} SD")
        per_task = scored.assign(z_base=z_base, z_stack=z_stack).groupby("task")
        for task, g in per_task:
            print(f"    {task:24} {g['z_base'].median():.2f} -> "
                  f"{g['z_stack'].median():.2f} SD  (n={len(g)})")
        eff = scored["stage2_effect"].value_counts()
        print(f"  2단이 정답 쪽으로 움직인 분자: {int(eff.get('개선', 0))}"
              f" / 악화 {int(eff.get('악화', 0))}")
        print("\n  '2단이동'이 0에 가까우면 GIN이 1단 예측을 거의 그대로 통과시킨")
        print("  것이고, 크면 그래프·descriptor로 고쳤다는 뜻이다. 개선/악화가")
        print("  반반이면 2단은 이 분자군에서 평균적으로 보탬이 없다는 뜻이다.")
    print(f"\n저장: {out}")
    print("\n주의: 모든 예측은 대상 분자를 train/valid에서 빼고 다시 학습해 얻은")
    print("      값이다. results/stack.csv(리더보드 숫자)와는 다른 학습이며,")
    print("      그 CSV는 건드리지 않는다.")


if __name__ == "__main__":
    main()
