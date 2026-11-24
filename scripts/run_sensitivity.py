"""민감도 분석: MPPGL과 Caco-2 -> Peff 기울기가 프로프라놀롤 PK를 얼마나 흔드는가.

  python scripts/run_sensitivity.py

**왜 이 두 개인가.** 3주차 결과에서 a priori 예측은 Cmax를 5배 가까이
틀렸고(GMFE 4.97, 2배 안쪽 0%), IV 데이터로 소실 쪽은 대체로 맞다는 것이
확인됐다. 그러면 남는 질문은 "흡수 쪽 가정 중 **어느 것**이 그 오차를
만들고 있는가"다. 모델에서 사람이 문헌값으로 정한 수(= 예측값이 아닌 수)
중 결과를 크게 움직일 수 있는 후보가 둘이다.

  MPPGL (간 1g당 미세소체 단백질)   기준 45 mg/g, 문헌 범위 30~60
  Caco-2 -> Peff 상관식 기울기       기준 0.6

두 파라미터는 모델에 들어가는 **방식**이 다르다. 그 차이가 결론이다.

  MPPGL  : CLint에 선형으로 곱해진다.
  기울기 : log Papp(약 -4.5)에 곱해져 **10의 지수**로 들어간다.

한 번에 하나씩(one-at-a-time) 흔든다. 두 파라미터는 서로 다른 경로
(소실 / 흡수)에 들어가서 상호작용이 작고, 무엇이 지배하는지를 보는
목적에는 OAT가 읽기 쉽다. 전역 민감도(Sobol 등)는 여기서는 과하다.

산출물
  results/sensitivity.csv            격자별 Cmax/AUC와 중간 파라미터
  results/sensitivity_summary.csv    파라미터별 변동폭과 탄성도
  results/sensitivity.png            그림 3패널
  results/admet_propranolol.json     ADMET 예측 캐시 (재실행을 빠르게)
                                     .gitignore 대상이다. 저장소에는 요약만
                                     올린다는 규칙을 따르고, 없으면 다시
                                     학습해서 만든다(몇 분).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

from pbpk.model import (  # noqa: E402
    PEFF_SLOPE, PHYS, params_from_admet, simulate,
)
from src.config import RESULTS_DIR  # noqa: E402

PROPRANOLOL = "CC(C)NCC(O)COc1cccc2ccccc12"

MPPGL_BASE = PHYS.mppgl_mg_g          # 45 mg/g
MPPGL_GRID = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]

SLOPE_BASE = PEFF_SLOPE               # 0.6
# 문헌의 Caco-2 -> Peff 로그-로그 상관식 기울기는 연구마다 0.4~0.8 범위에
# 흩어져 있다. 기준값 0.6의 ±0.2를 범위로 잡는다.
SLOPE_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


# ---------------------------------------------------------------------------
# ADMET 예측 (캐시)
# ---------------------------------------------------------------------------


def admet_inputs(cache_path: Path, refresh: bool = False) -> dict:
    """세 ADMET 예측값을 얻는다. 한 번 구하면 JSON에 남겨 재사용한다.

    예측 자체가 RF 5 seed x 3과제라 몇 분 걸린다. 민감도 격자를 돌릴 때
    매번 다시 학습할 이유가 없다. **단, 캐시는 예측 결과일 뿐이고
    민감도 분석이 흔드는 것은 예측값이 아니라 생리·상관식 가정이다.**
    """
    if cache_path.exists() and not refresh:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  ADMET 예측 캐시 사용: {cache_path}")
        return data

    from pbpk.predict_admet import predict_all
    from src.data import get_group

    print("  ADMET 예측 중 (RF, seed 5개, 3과제) …")
    group = get_group()
    preds = predict_all(group, PROPRANOLOL)
    data = {
        "smiles": PROPRANOLOL,
        "logp_rdkit": float(Descriptors.MolLogP(Chem.MolFromSmiles(PROPRANOLOL))),
        "tasks": {
            task: {
                "value": p.value, "std": p.std, "leaked": p.leaked,
                "excluded_n": p.excluded_n, "n_seeds": p.n_seeds,
            }
            for task, p in preds.items()
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  캐시 저장: {cache_path}")
    return data


def build_params(admet: dict, mppgl: float, slope: float):
    """한 (MPPGL, 기울기) 조합의 모델 파라미터를 만든다."""
    phys = replace(PHYS, mppgl_mg_g=float(mppgl))
    par = params_from_admet(
        caco2_log_papp=admet["tasks"]["Caco2_Wang"]["value"],
        ppbr_percent=admet["tasks"]["PPBR_AZ"]["value"],
        clearance_microsome=admet["tasks"]["Clearance_Microsome_AZ"]["value"],
        logp=admet["logp_rdkit"],
        phys=phys,
        peff_slope=float(slope),
    )
    return par, phys


def evaluate(admet: dict, mppgl: float, slope: float,
             dose_mg: float, t_end_h: float) -> dict:
    """한 조합의 Cmax/AUC와 중간 파라미터."""
    par, phys = build_params(admet, mppgl, slope)
    sim = simulate(par, dose_mg=dose_mg, route="PO",
                   t_end_h=t_end_h, n_points=1200, phys=phys)

    # 창에 의존하지 않는 해석해. 이 모델 구조에서는
    #   AUC(0-inf) = Dose x F / CL,  F = Fa(1-E_h),  CL = Qh x E_h
    # 를 정리하면 Qh와 E_h가 약분되어
    #   AUC(0-inf) = Dose x Fa / (fu x CLint)
    # 가 된다. 고추출 약물의 경구 AUC가 간혈류가 아니라 고유청소율로만
    # 정해진다는 교과서 결과다. MPPGL은 CLint에 선형으로 들어가므로
    # 이 기준에서 탄성도가 정확히 -1이어야 하고, 아래 표의 0-24h 값이
    # 그보다 둔한 것은 **창을 끊어서 생긴 차이**다. 둘을 같이 보고한다.
    auc_inf = 1000.0 * dose_mg * par.fa / (par.fu * par.clint_L_h)

    return {
        "mppgl_mg_g": mppgl, "peff_slope": slope,
        "cmax_ug_L": sim["cmax_ug_L"], "tmax_h": sim["tmax_h"],
        "auc_ug_h_L": sim["auc_ug_h_L"],
        "auc_inf_ug_h_L": auc_inf,
        "ka_h": par.ka_h, "fa": par.fa,
        "clint_L_h": par.clint_L_h,
        "E_h": par.hepatic_extraction(phys),
        "F": par.bioavailability(phys),
        "CL_L_h": par.clearance_L_h(phys),
    }


def elasticity(admet: dict, which: str, dose_mg: float, t_end_h: float,
               rel: float = 0.05) -> dict:
    """기준점에서의 국소 탄성도 d ln Y / d ln X (중심차분).

    "그 파라미터를 1% 올리면 결과가 몇 % 움직이는가"다. 단위가 다른
    파라미터를 같은 눈으로 비교할 수 있게 해 준다. MPPGL처럼 선형으로
    들어가는 값은 탄성도가 상수에 가깝고, 기울기처럼 지수로 들어가는
    값은 기준점에 따라 달라진다.
    """
    if which == "mppgl":
        lo = evaluate(admet, MPPGL_BASE / (1 + rel), SLOPE_BASE, dose_mg, t_end_h)
        hi = evaluate(admet, MPPGL_BASE * (1 + rel), SLOPE_BASE, dose_mg, t_end_h)
    else:
        lo = evaluate(admet, MPPGL_BASE, SLOPE_BASE / (1 + rel), dose_mg, t_end_h)
        hi = evaluate(admet, MPPGL_BASE, SLOPE_BASE * (1 + rel), dose_mg, t_end_h)
    denom = 2.0 * np.log(1 + rel)
    return {
        "cmax": float((np.log(hi["cmax_ug_L"]) - np.log(lo["cmax_ug_L"])) / denom),
        "auc": float((np.log(hi["auc_ug_h_L"]) - np.log(lo["auc_ug_h_L"])) / denom),
        "auc_inf": float(
            (np.log(hi["auc_inf_ug_h_L"]) - np.log(lo["auc_inf_ug_h_L"])) / denom
        ),
    }


# 관측과 대조할 때 쓸 투여 조건. 여기 시뮬레이션은 단회 80 mg 경구다.
OBSERVED_DATASET = "po_single_80mg"


def observed_cmax(path: Path, dataset: str = OBSERVED_DATASET) -> dict:
    """3주차 비교 결과에서 관측 Cmax 분포를 읽는다.

    AUC는 연구마다 마지막 채혈 시각이 8~32시간으로 달라 고정 창
    시뮬레이션과 바로 비교할 수 없다. Cmax는 모든 연구가 Tmax를 지나
    관측했으므로 창에 거의 의존하지 않는다. 그래서 관측 띠는 Cmax만
    그린다.
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    d = df[df["mode"] == "a_priori"]
    if "dataset" in d.columns:
        # **투여 조건이 같은 곡선만 골라야 한다.** 2026-11-24 관측 세트
        # 정정으로 dataset 열이 생겼다(단회 80 mg / 다른 용량 / 정상상태 /
        # IV). 전부 섞으면 용량이 다른 곡선과 정상상태 축적분이 함께 들어가
        # 기준선이 틀어진다. 섞으면 25곡선 65.9, 단회 80 mg만 보면
        # 7곡선 71.1 ug/L로 갈린다.
        d = d[d["dataset"] == dataset]
    d = d["cmax_obs"].dropna()
    if not len(d):
        return {}
    return {
        "median": float(d.median()), "min": float(d.min()),
        "max": float(d.max()), "n": int(len(d)),
    }


def plot(df_m: pd.DataFrame, df_s: pd.DataFrame, base: dict,
         summary: pd.DataFrame, obs: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    cmax_b, auc_b = base["cmax_ug_L"], base["auc_ug_h_L"]

    def panel(ax, df, xcol, xbase, xlabel, title):
        ax.plot(df[xcol], df["cmax_ug_L"] / cmax_b, "o-", lw=2,
                label="Cmax (ratio to base)")
        ax.plot(df[xcol], df["auc_ug_h_L"] / auc_b, "s--", lw=2,
                label="AUC 0-24h (ratio to base)")
        ax.axhline(1.0, color="0.6", lw=1)
        ax.axvline(xbase, color="0.6", lw=1, ls=":")
        if obs:
            # a priori Cmax를 관측 중앙값에 맞추려면 몇 배가 필요한가
            need = obs["median"] / cmax_b
            ax.axhline(need, color="crimson", lw=1.4, ls="-.",
                       label=f"need x{need:.1f} for observed median Cmax")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("fold change vs baseline")
        ax.set_title(title)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="best")

    panel(axes[0], df_m, "mppgl_mg_g", MPPGL_BASE,
          "MPPGL (mg microsomal protein / g liver)",
          f"MPPGL {MPPGL_GRID[0]:.0f}-{MPPGL_GRID[-1]:.0f}  (base {MPPGL_BASE:.0f})")
    panel(axes[1], df_s, "peff_slope", SLOPE_BASE,
          "Caco-2 -> Peff correlation slope",
          f"slope {SLOPE_GRID[0]:.2f}-{SLOPE_GRID[-1]:.2f}  (base {SLOPE_BASE:.2f})")

    # 토네이도: 각 파라미터의 문헌 범위가 만드는 변동폭
    ax = axes[2]
    labels, cmax_folds, auc_folds = [], [], []
    for _, r in summary.iterrows():
        labels.append(r["parameter"])
        cmax_folds.append(r["cmax_fold_range"])
        auc_folds.append(r["auc_fold_range"])
    y = np.arange(len(labels))
    h = 0.36
    # 변동폭은 정의상 1배 이상이므로 막대를 1에서 시작시키고 x축을 로그로
    # 둔다. 선형축이면 1.5배 막대가 563배 옆에서 보이지 않는다.
    ax.barh(y + h / 2, np.array(cmax_folds) - 1, left=1, height=h, label="Cmax")
    ax.barh(y - h / 2, np.array(auc_folds) - 1, left=1, height=h, label="AUC 0-24h")
    for i, (c, a) in enumerate(zip(cmax_folds, auc_folds)):
        ax.text(c, i + h / 2, f"  {c:.1f}x", va="center", fontsize=9)
        ax.text(a, i - h / 2, f"  {a:.1f}x", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(1, max(max(cmax_folds), max(auc_folds)) * 3)
    ax.set_xlabel("fold range across the parameter's literature range (max/min)")
    ax.set_title("Which assumption dominates")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "Propranolol 80 mg PO — one-at-a-time sensitivity of the reduced PBPK",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose-mg", type=float, default=80.0)
    ap.add_argument("--t-end-h", type=float, default=24.0,
                    help="AUC 적분 창. 격자 전체에 같은 창을 쓴다")
    ap.add_argument("--refresh-admet", action="store_true",
                    help="ADMET 예측 캐시를 무시하고 다시 학습한다")
    ap.add_argument("--out", default=str(RESULTS_DIR / "sensitivity.csv"))
    args = ap.parse_args()

    out = Path(args.out)

    print("=" * 78)
    print("1. ADMET 예측값 (민감도 분석이 흔드는 것은 이 값이 아니라 가정이다)")
    print("=" * 78)
    admet = admet_inputs(out.with_name("admet_propranolol.json"),
                         refresh=args.refresh_admet)
    for task, d in admet["tasks"].items():
        tag = f", train에서 {d['excluded_n']}건 제외" if d["leaked"] else ""
        print(f"  {task:24} {d['value']:9.3f} +/- {d['std']:.3f}"
              f" (seed {d['n_seeds']}개{tag})")
    print(f"  {'MolLogP (RDKit)':24} {admet['logp_rdkit']:9.3f}")

    # --- 기준점 ------------------------------------------------------------
    base = evaluate(admet, MPPGL_BASE, SLOPE_BASE, args.dose_mg, args.t_end_h)
    print("\n" + "=" * 78)
    print(f"2. 기준점  MPPGL={MPPGL_BASE:.0f} mg/g, 기울기={SLOPE_BASE:.2f}")
    print("=" * 78)
    print(f"  Cmax {base['cmax_ug_L']:8.2f} ug/L   Tmax {base['tmax_h']:5.2f} h"
          f"   AUC(0-{args.t_end_h:.0f}h) {base['auc_ug_h_L']:9.1f} ug*h/L")
    print(f"  ka {base['ka_h']:.4f} 1/h   Fa {base['fa']:.4f}"
          f"   CLint {base['clint_L_h']:7.1f} L/h"
          f"   E_h {base['E_h']:.3f}   F {base['F']:.4f}"
          f"   CL {base['CL_L_h']:5.1f} L/h")

    # 기준점이 3주차 저장값과 같은지 확인한다. 리팩터링으로 파라미터
    # 경로가 틀어졌다면 여기서 드러난다.
    ref = out.with_name("pbpk_parameters.csv")
    if ref.exists():
        prev = pd.read_csv(ref)
        prev = prev[prev["mode"] == "a_priori"]
        if len(prev):
            p = prev.iloc[0]
            ok = all(
                abs(base[k] - float(p[c])) < tol
                for k, c, tol in [("ka_h", "ka_h", 1e-6),
                                  ("fa", "fa", 1e-6),
                                  ("clint_L_h", "clint_L_h", 1e-4),
                                  ("E_h", "E_h", 1e-6)]
            )
            print(f"  3주차 저장 파라미터와 일치: {'예' if ok else '아니오 (확인 필요)'}")

    # --- OAT 격자 ----------------------------------------------------------
    rows_m = [evaluate(admet, m, SLOPE_BASE, args.dose_mg, args.t_end_h)
              for m in MPPGL_GRID]
    rows_s = [evaluate(admet, MPPGL_BASE, s, args.dose_mg, args.t_end_h)
              for s in SLOPE_GRID]
    df_m = pd.DataFrame(rows_m).assign(swept="mppgl_mg_g")
    df_s = pd.DataFrame(rows_s).assign(swept="peff_slope")

    def show(df, xcol, label):
        print("\n" + "=" * 78)
        print(f"3. {label}")
        print("=" * 78)
        t = df[[xcol, "ka_h", "fa", "clint_L_h", "E_h", "F",
                "cmax_ug_L", "tmax_h", "auc_ug_h_L"]].copy()
        t["cmax_fold"] = (df["cmax_ug_L"] / base["cmax_ug_L"]).round(3)
        t["auc_fold"] = (df["auc_ug_h_L"] / base["auc_ug_h_L"]).round(3)
        for c in ("ka_h", "fa", "E_h", "F"):
            t[c] = t[c].round(4)
        for c in ("clint_L_h", "cmax_ug_L", "auc_ug_h_L"):
            t[c] = t[c].round(2)
        t["tmax_h"] = t["tmax_h"].round(2)
        print(t.to_string(index=False))

    show(df_m, "mppgl_mg_g", f"MPPGL 흔들기 (기울기 {SLOPE_BASE:.2f} 고정)")
    show(df_s, "peff_slope", f"Caco-2 -> Peff 기울기 흔들기 (MPPGL {MPPGL_BASE:.0f} 고정)")

    # --- 요약 --------------------------------------------------------------
    el_m = elasticity(admet, "mppgl", args.dose_mg, args.t_end_h)
    el_s = elasticity(admet, "slope", args.dose_mg, args.t_end_h)
    summary = pd.DataFrame([
        {
            "parameter": "MPPGL (mg/g)",
            "base": MPPGL_BASE,
            "range": f"{MPPGL_GRID[0]:.0f}-{MPPGL_GRID[-1]:.0f}",
            "range_fold": round(MPPGL_GRID[-1] / MPPGL_GRID[0], 2),
            "enters_as": "linear multiplier on CLint",
            "cmax_min": round(df_m["cmax_ug_L"].min(), 2),
            "cmax_max": round(df_m["cmax_ug_L"].max(), 2),
            "cmax_fold_range": round(df_m["cmax_ug_L"].max() / df_m["cmax_ug_L"].min(), 2),
            "auc_min": round(df_m["auc_ug_h_L"].min(), 1),
            "auc_max": round(df_m["auc_ug_h_L"].max(), 1),
            "auc_fold_range": round(df_m["auc_ug_h_L"].max() / df_m["auc_ug_h_L"].min(), 2),
            "auc_inf_fold_range": round(
                df_m["auc_inf_ug_h_L"].max() / df_m["auc_inf_ug_h_L"].min(), 2),
            "elasticity_cmax": round(el_m["cmax"], 3),
            "elasticity_auc": round(el_m["auc"], 3),
            "elasticity_auc_inf": round(el_m["auc_inf"], 3),
        },
        {
            "parameter": "Caco-2->Peff slope",
            "base": SLOPE_BASE,
            "range": f"{SLOPE_GRID[0]:.2f}-{SLOPE_GRID[-1]:.2f}",
            "range_fold": round(SLOPE_GRID[-1] / SLOPE_GRID[0], 2),
            "enters_as": "exponent on 10^(slope x logPapp)",
            "cmax_min": round(df_s["cmax_ug_L"].min(), 2),
            "cmax_max": round(df_s["cmax_ug_L"].max(), 2),
            "cmax_fold_range": round(df_s["cmax_ug_L"].max() / df_s["cmax_ug_L"].min(), 2),
            "auc_min": round(df_s["auc_ug_h_L"].min(), 1),
            "auc_max": round(df_s["auc_ug_h_L"].max(), 1),
            "auc_fold_range": round(df_s["auc_ug_h_L"].max() / df_s["auc_ug_h_L"].min(), 2),
            "auc_inf_fold_range": round(
                df_s["auc_inf_ug_h_L"].max() / df_s["auc_inf_ug_h_L"].min(), 2),
            "elasticity_cmax": round(el_s["cmax"], 3),
            "elasticity_auc": round(el_s["auc"], 3),
            "elasticity_auc_inf": round(el_s["auc_inf"], 3),
        },
    ])

    print("\n" + "=" * 78)
    print("4. 요약 — 문헌 범위가 만드는 변동폭과 국소 탄성도")
    print("=" * 78)
    print(summary[["parameter", "base", "range", "enters_as",
                   "cmax_fold_range", "auc_fold_range", "auc_inf_fold_range",
                   "elasticity_cmax", "elasticity_auc",
                   "elasticity_auc_inf"]].to_string(index=False))
    print("\n  auc_*  = 0-24h 창, auc_inf_* = 해석해 Dose x Fa / (fu x CLint) (창 무관).")
    print("  창을 끊으면 느린 흡수 쪽이 과장되고 소실 쪽이 축소된다."
          " 두 기준 모두에서 지배 관계가 같은지 확인하는 것이 요점이다.")

    obs = observed_cmax(out.with_name("pk_comparison.csv"))
    if obs:
        need = obs["median"] / base["cmax_ug_L"]
        print(f"\n  관측 Cmax ({OBSERVED_DATASET}, 곡선 {obs['n']}개): 중앙값 {obs['median']:.1f},"
              f" 범위 {obs['min']:.1f}-{obs['max']:.1f} ug/L")
        print(f"  a priori Cmax {base['cmax_ug_L']:.1f} -> 중앙값까지 {need:.1f}배 필요")
        in_slope = df_s["cmax_ug_L"].max() / base["cmax_ug_L"] >= need
        in_mppgl = df_m["cmax_ug_L"].max() / base["cmax_ug_L"] >= need
        print(f"    기울기 범위 안에서 도달 가능: {'예' if in_slope else '아니오'}")
        print(f"    MPPGL  범위 안에서 도달 가능: {'예' if in_mppgl else '아니오'}")

    dom = summary.loc[summary["auc_fold_range"].idxmax(), "parameter"]
    print(f"\n  지배 가정: {dom}")

    # --- 저장 --------------------------------------------------------------
    pd.concat([df_m, df_s], ignore_index=True).to_csv(out, index=False)
    summary_path = out.with_name("sensitivity_summary.csv")
    summary.to_csv(summary_path, index=False)
    fig_path = out.with_name("sensitivity.png")
    plot(df_m, df_s, base, summary, obs, fig_path)

    print(f"\n저장: {out}")
    print(f"저장: {summary_path}")
    print(f"저장: {fig_path}")


if __name__ == "__main__":
    main()
