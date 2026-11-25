"""3주차: ADMET 예측값을 축소 PBPK에 넣고 관측 human PK와 비교한다.

  python scripts/run_pbpk.py

두 가지를 나눠서 돌린다.

1. **a priori** — 세 ADMET 예측값을 그대로 파라미터로 바꿔 넣는다.
   관측 데이터를 전혀 보지 않는다. 이게 "예측"이다.

2. **보정(calibration)** — Caco-2 -> Peff 상관식의 절편 하나만 **곡선 하나**에
   맞추고, 나머지로 검증한다. 상관식 상수가 문헌 확인이 필요한 가정이기
   때문에, 그 하나를 데이터가 정하게 하면 얼마나 나아지는지 본다. 맞춘
   곡선은 검증에서 뺀다. 전부에 맞추면 그건 검증이 아니다.

관측 곡선은 조건이 같은 것끼리만 묶어 비교한다(`pbpk/observed.py`).

  주 세트   건강 성인 경구 단회 80 mg, 총 농도
  보조 세트 다른 용량의 경구 단회 / 80 mg 반복투여 정상상태 / 정맥

단회와 정상상태를 섞으면 안 된다. 정상상태 Cmax는 앞선 투여분이 쌓여
단회보다 높고, 그걸 단회 시뮬레이션과 비교하면 모델이 실제보다 더
못 맞히는 것처럼 보인다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

from pbpk.model import (  # noqa: E402
    PEFF_INTERCEPT, params_from_admet, simulate, simulate_steady_state,
)
from pbpk.observed import EXCLUDED, load_curves, select  # noqa: E402
from pbpk.predict_admet import predict_all  # noqa: E402
from src.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data import get_group  # noqa: E402

PROPRANOLOL = "CC(C)NCC(O)COc1cccc2ccccc12"

# 곡선의 조건. 한 곡선 안에서는 모든 행이 같은 값을 가진다.
META = ["curve_id", "study", "sheet", "arm", "population", "route",
        "dose_mg", "regimen", "tau_h", "infusion_h"]


def curve_meta(d: pd.DataFrame) -> dict:
    return {k: d[k].iloc[0] for k in META}


def quantifiable(d: pd.DataFrame) -> pd.DataFrame:
    """정량된 점만 남긴다 (농도 > 0).

    뒤쪽의 0은 정량한계 미만이라는 뜻이지 농도가 0이라는 뜻이 아니다.
    AUC에 넣으면 꼬리를 실제보다 깎는다.

    앞쪽의 0은 투여 전 채혈이다. 이것도 뺀다. 정맥 순간 주입에서 투여 전
    0을 창의 시작으로 삼으면, 관측은 0에서 첫 채혈까지 삼각형인데
    시뮬레이션은 같은 구간에 주입 직후의 첨두가 통째로 들어간다.
    """
    d = d.sort_values("time_h")
    return d[d["conc_ug_L"] > 0]


def observed_metrics(d: pd.DataFrame) -> dict:
    """한 곡선의 관측 Cmax/Tmax/AUC. AUC는 첫 정량 시점부터 마지막 정량 시점까지."""
    d = quantifiable(d)
    i = d["conc_ug_L"].idxmax()
    return {
        "cmax_obs": float(d.loc[i, "conc_ug_L"]),
        "tmax_obs": float(d.loc[i, "time_h"]),
        "auc_obs": float(np.trapz(d["conc_ug_L"], d["time_h"])),
        "t_first": float(d["time_h"].min()),
        "t_last": float(d["time_h"].max()),
        "n_points": int(len(d)),
    }


def simulate_curve(par, meta: dict, t_end_h: float, n_points: int = 1200) -> dict:
    """곡선의 투여 조건(경로·투여법)에 맞는 시뮬레이션을 고른다."""
    dose = float(meta["dose_mg"])
    if meta["route"] == "IV":
        return simulate(par, dose, "IV", t_end_h, n_points,
                        infusion_h=float(meta["infusion_h"]))
    if meta["regimen"] == "steady_state":
        return simulate_steady_state(par, dose, float(meta["tau_h"]),
                                     t_end_h, n_points)
    return simulate(par, dose, "PO", t_end_h, n_points)


def window_auc(t: np.ndarray, c: np.ndarray, t0: float, t1: float) -> float:
    """시뮬레이션 곡선을 [t0, t1]에서만 적분한다."""
    inside = (t > t0) & (t < t1)
    tt = np.concatenate([[t0], t[inside], [t1]])
    cc = np.concatenate([[np.interp(t0, t, c)], c[inside], [np.interp(t1, t, c)]])
    return float(np.trapz(cc, tt))


def simulated_metrics(par, meta: dict, obs: dict) -> dict:
    """관측과 같은 시간 창에서 시뮬레이션 Cmax/Tmax/AUC.

    AUC는 **양쪽 끝을 모두** 관측에 맞춘다. 시뮬레이션은 0시간부터
    무한대까지 적분할 수 있지만 관측은 첫 채혈부터 마지막 채혈까지만
    있다. 뒤쪽만 맞추고 앞쪽을 0부터 적분하면 첫 채혈 이전의 면적이
    시뮬레이션에만 들어간다. 정맥 순간 주입에서는 그 면적이 크다.
    """
    sim = simulate_curve(par, meta, obs["t_last"])
    return {
        "cmax_sim": sim["cmax_ug_L"],
        "tmax_sim": sim["tmax_h"],
        "auc_sim": window_auc(sim["time_h"], sim["conc_ug_L"],
                              obs["t_first"], obs["t_last"]),
    }


def fold_error(sim: float, obs: float) -> float:
    """배수 오차. 1.0이 완벽, 2.0은 2배 틀림, 0.5도 2배 틀림이다.

    PK에서는 보통 2배 안쪽이면 받아들일 만하다고 본다.
    """
    if obs <= 0:
        return float("nan")
    return float(sim / obs)


def compare(par, df: pd.DataFrame, mode: str, dataset: str) -> pd.DataFrame:
    """곡선마다 관측과 시뮬레이션을 비교한다."""
    rows = []
    for _, d in df.groupby("curve_id", sort=False):
        meta = curve_meta(d)
        obs = observed_metrics(d)
        sim = simulated_metrics(par, meta, obs)
        rows.append({
            "mode": mode, "dataset": dataset,
            "curve_id": meta["curve_id"], "study": meta["study"],
            "arm": meta["arm"], "dose_mg": meta["dose_mg"],
            "regimen": meta["regimen"],
            "n_points": obs["n_points"],
            "t_first_h": round(obs["t_first"], 2),
            "t_last_h": round(obs["t_last"], 1),
            "cmax_obs": round(obs["cmax_obs"], 1),
            "cmax_sim": round(sim["cmax_sim"], 1),
            "cmax_fold": round(fold_error(sim["cmax_sim"], obs["cmax_obs"]), 3),
            "tmax_obs": round(obs["tmax_obs"], 2),
            "tmax_sim": round(sim["tmax_sim"], 2),
            "auc_obs": round(obs["auc_obs"], 1),
            "auc_sim": round(sim["auc_sim"], 1),
            "auc_fold": round(fold_error(sim["auc_sim"], obs["auc_obs"]), 3),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """기하평균 배수오차와 2배 안쪽 비율.

    배수오차는 기하평균으로 모아야 한다. 2배 큰 것과 2배 작은 것이
    산술평균에서는 상쇄되지 않고 한쪽으로 치우친다.

    bias는 부호를 살린 기하평균이다. 1보다 작으면 전체적으로 과소예측,
    크면 과대예측이다. GMFE만 보면 방향을 알 수 없다.
    """
    def gmfe(series):
        v = series.dropna()
        v = v[v > 0]
        return float(np.exp(np.mean(np.abs(np.log(v))))) if len(v) else float("nan")

    def bias(series):
        v = series.dropna()
        v = v[v > 0]
        return float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")

    def within2(series):
        v = series.dropna()
        return float(np.mean((v >= 0.5) & (v <= 2.0)) * 100) if len(v) else float("nan")

    out = {
        "mode": df["mode"].iloc[0],
        "dataset": df["dataset"].iloc[0],
        "n_curves": int(len(df)),
        "n_studies": int(df["study"].nunique()),
        "auc_gmfe": round(gmfe(df["auc_fold"]), 3),
        "auc_bias": round(bias(df["auc_fold"]), 3),
        "auc_within_2fold_pct": round(within2(df["auc_fold"]), 1),
    }
    if df["cmax_fold"].notna().any():
        out.update({
            "cmax_gmfe": round(gmfe(df["cmax_fold"]), 3),
            "cmax_bias": round(bias(df["cmax_fold"]), 3),
            "cmax_within_2fold_pct": round(within2(df["cmax_fold"]), 1),
            "tmax_median_obs": round(float(df["tmax_obs"].median()), 2),
            "tmax_sim": round(float(df["tmax_sim"].median()), 2),
        })
    return out


def print_summary(s: dict) -> None:
    line = (f"  [{s['mode']} / {s['dataset']}] 곡선 {s['n_curves']}개"
            f" (연구 {s['n_studies']}개)")
    if "cmax_gmfe" in s:
        line += (f"\n    Cmax 배수오차 {s['cmax_gmfe']} (치우침 {s['cmax_bias']}),"
                 f" 2배 안쪽 {s['cmax_within_2fold_pct']}%")
    line += (f"\n    AUC  배수오차 {s['auc_gmfe']} (치우침 {s['auc_bias']}),"
             f" 2배 안쪽 {s['auc_within_2fold_pct']}%")
    print(line)


def terminal_half_life(t: np.ndarray, c: np.ndarray) -> float:
    """로그-선형 회귀로 구한 반감기."""
    slope = np.polyfit(t, np.log(c), 1)[0]
    return float(np.log(2) / -slope) if slope < 0 else float("nan")


def iv_check(par, df_iv: pd.DataFrame, n_tail: int = 4) -> pd.DataFrame:
    """IV 데이터로 소실·분포를 따로 확인한다.

    IV는 흡수를 거치지 않으므로 Caco-2 쪽 가정이 들어가지 않는다. 경구에서
    어긋난 것이 흡수 때문인지 소실 때문인지 가르는 데 쓴다.

    세 가지를 본다.

    - 관측 창의 부분 AUC. **이것만으로는 소실을 판단할 수 없다.** 초기에
      과대, 중간에 과소, 말단에 과대인 곡선도 면적은 비슷하게 나온다.
    - 말단 반감기. 관측의 마지막 n_tail개 시점에서 구하고, 시뮬레이션도
      **같은 시점의 값으로** 구한다.
    - 청소율. 관측은 비구획 분석으로 구한다: CL = Dose / AUC(0-inf).
      꼬리는 C_last/lambda_z로 잇고, 첫 채혈 이전은 첫 농도로 채운다
      (주입 중이었으면 0에서 올라가는 삼각형). 모델 쪽은 파라미터에서
      바로 나오는 값이다.

    Cmax는 비교하지 않는다. 주입 직후의 값은 채혈 시점에 따라 크게
    달라진다.
    """
    rows = []
    for _, d in df_iv.groupby("curve_id", sort=False):
        meta = curve_meta(d)
        obs = observed_metrics(d)
        q = quantifiable(d)
        sim = simulate_curve(par, meta, obs["t_last"], n_points=2400)
        auc_sim = window_auc(sim["time_h"], sim["conc_ug_L"],
                             obs["t_first"], obs["t_last"])
        tail_t = q["time_h"].to_numpy()[-n_tail:]
        tail_c = q["conc_ug_L"].to_numpy()[-n_tail:]
        thalf_obs = terminal_half_life(tail_t, tail_c)
        c_first = float(q["conc_ug_L"].iloc[0])
        head = (0.5 if float(meta["infusion_h"]) > 0 else 1.0) * c_first * obs["t_first"]
        auc_inf = obs["auc_obs"] + head + tail_c[-1] * thalf_obs / np.log(2)
        rows.append({
            "mode": "a_priori", "dataset": "iv",
            "curve_id": meta["curve_id"], "study": meta["study"],
            "arm": meta["arm"], "dose_mg": round(float(meta["dose_mg"]), 2),
            "regimen": meta["regimen"],
            "n_points": obs["n_points"],
            "t_first_h": round(obs["t_first"], 2),
            "t_last_h": round(obs["t_last"], 1),
            "auc_obs": round(obs["auc_obs"], 1),
            "auc_sim": round(auc_sim, 1),
            "auc_fold": round(fold_error(auc_sim, obs["auc_obs"]), 3),
            "thalf_obs_h": round(thalf_obs, 1),
            "thalf_sim_h": round(terminal_half_life(
                tail_t, np.interp(tail_t, sim["time_h"], sim["conc_ug_L"])), 1),
            # mg -> ug 이므로 1000을 곱한다: ug / (ug·h/L) = L/h
            "cl_obs_L_h": round(float(meta["dose_mg"]) * 1000.0 / auc_inf, 1),
            "cl_sim_L_h": round(par.clearance_L_h(), 1),
        })
    return pd.DataFrame(rows)


def plot_curves(par, par_cal, obs: pd.DataFrame, calib_id: str,
                dose_mg: float, path: Path,
                df_iv: pd.DataFrame | None = None) -> None:
    """관측 점 위에 a priori와 보정 곡선을 겹쳐 그린다.

    경구 두 패널만으로는 2026-11-24 정정의 결론이 보이지 않는다. 단회
    80 mg에서 보정 곡선은 점 위를 지나가지만, 그 보정은 흡수 쪽 절편
    하나만 올린 것이라 소실이 맞다는 뜻이 아니다. 그래서 흡수 가정이
    들어가지 않는 IV 패널을 같이 그린다. 세 번째 패널에서 **관측 점의
    기울기가 모델 곡선보다 가파른 것**이 소실이 느리다는 증거다.
    """
    import matplotlib
    matplotlib.use("Agg")  # 화면 없는 환경에서 그린다
    import matplotlib.pyplot as plt

    obs = obs[obs["conc_ug_L"] > 0]
    t_end = float(min(obs["time_h"].max(), 24.0))
    sim_a = simulate(par, dose_mg, "PO", t_end, n_points=800)
    sim_c = simulate(par_cal, dose_mg, "PO", t_end, n_points=800)

    has_iv = df_iv is not None and len(df_iv) > 0
    n_panel = 3 if has_iv else 2
    fig, axes = plt.subplots(1, n_panel, figsize=(5.6 * n_panel, 4.8))
    for ax, logy in zip(axes[:2], (False, True)):
        rest = obs[obs["curve_id"] != calib_id]
        cal = obs[obs["curve_id"] == calib_id]
        ax.scatter(rest["time_h"], rest["conc_ug_L"], s=14, alpha=0.45,
                   color="0.45", label="observed (validation)")
        ax.scatter(cal["time_h"], cal["conc_ug_L"], s=22, marker="^",
                   color="tab:red", alpha=0.8,
                   label=f"observed (calibration: {cal['study'].iloc[0]})")
        ax.plot(sim_a["time_h"], sim_a["conc_ug_L"], lw=2.2,
                label="a priori (ADMET only)")
        ax.plot(sim_c["time_h"], sim_c["conc_ug_L"], lw=2.2, ls="--",
                label="1-parameter calibrated")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Plasma concentration (ug/L)")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1, None)
            ax.set_title("same data, log scale\n"
                         "the calibrated tail runs above the points")
        else:
            ax.set_title(f"{dose_mg:.0f} mg PO single dose, healthy adults\n"
                         f"({obs['curve_id'].nunique()} curves, "
                         f"{obs['study'].nunique()} studies)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

    if has_iv:
        # IV는 곡선마다 용량(1~10 mg)과 주입 시간이 다르다. 모델이 선형이라
        # 농도를 용량으로 나누면 전부 한 축에 겹쳐 놓고 볼 수 있다. 기울기만
        # 보는 패널이므로 이 정규화가 비교를 왜곡하지 않는다.
        ax = axes[2]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        lo = np.inf
        for i, (_, d) in enumerate(df_iv.groupby("curve_id", sort=False)):
            meta = curve_meta(d)
            q = quantifiable(d)
            if not len(q):
                continue
            dose = float(meta["dose_mg"])
            c_obs = np.asarray(q["conc_ug_L"], dtype=float) / dose
            lo = min(lo, c_obs.min())
            c = colors[i % len(colors)]
            ax.scatter(q["time_h"], c_obs, s=22, color=c, alpha=0.85, zorder=3,
                       label=f"{meta['study']} ({dose:.0f} mg)")
            sim = simulate_curve(par, meta, float(q["time_h"].max()),
                                 n_points=1200)
            ax.plot(sim["time_h"], np.asarray(sim["conc_ug_L"]) / dose,
                    lw=1.4, color="0.35", alpha=0.7, zorder=2,
                    label="model (a priori)" if i == 0 else None)
        ax.set_yscale("log")
        # 주입 직후의 뾰족한 봉우리는 채혈 시점에 따라 들쭉날쭉해서 비교
        # 대상이 아니다. 축을 소실상에 맞춰 잘라야 **관측이 모델보다
        # 가파르게 떨어지는 것**이 보인다.
        ax.set_ylim(lo / 2, 30)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Plasma concentration / dose (ug/L per mg)")
        ax.set_title("IV, dose-normalised (a priori)\n"
                     "observed falls faster than the model")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.25, which="both")

    fig.suptitle("Propranolol — reduced liver+gut PBPK vs observed", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=None, help="관측 데이터 엑셀 경로")
    ap.add_argument("--dose-mg", type=float, default=80.0,
                    help="주 검증 세트로 쓸 경구 단회 용량")
    ap.add_argument("--calib-curve", default=None,
                    help="보정에 쓸 곡선 id. 기본은 주 세트에서 관측점이 가장 많은 곡선")
    ap.add_argument("--out", default=str(RESULTS_DIR / "pk_comparison.csv"))
    args = ap.parse_args()
    primary = f"po_single_{args.dose_mg:g}mg"

    # --- 1. ADMET 예측 -----------------------------------------------------
    print("=" * 70)
    print("1. ADMET 예측 (프로프라놀롤)")
    print("=" * 70)
    group = get_group()
    preds = predict_all(group, PROPRANOLOL)
    for task, p in preds.items():
        print(f"  {p}")
        if p.leaked:
            print(f"     ! train에 있던 값 {p.train_values} -> 제외하고 재학습했다")

    logp = float(Descriptors.MolLogP(Chem.MolFromSmiles(PROPRANOLOL)))

    def make_params(intercept: float | None = None):
        return params_from_admet(
            caco2_log_papp=preds["Caco2_Wang"].value,
            ppbr_percent=preds["PPBR_AZ"].value,
            clearance_microsome=preds["Clearance_Microsome_AZ"].value,
            logp=logp,
            peff_intercept=intercept,
        )

    # --- 2. a priori 파라미터 ---------------------------------------------
    print("\n" + "=" * 70)
    print("2. PBPK 파라미터 (a priori — 관측을 보지 않았다)")
    print("=" * 70)
    par = make_params()
    for k, v in par.source.items():
        print(f"  {k:12} {v}")
    print(f"\n  {'Vss':14} {par.vss_L():8.0f} L     문헌 ~280")
    print(f"  {'간 추출률 E_h':14} {par.hepatic_extraction():8.3f}       문헌 ~0.6-0.75")
    print(f"  {'생체이용률 F':14} {par.bioavailability():8.3f}       문헌 ~0.26")
    print(f"  {'혈액 청소율':14} {par.clearance_L_h():8.1f} L/h   문헌 ~60-70")

    # --- 3. 관측 데이터 ----------------------------------------------------
    print("\n" + "=" * 70)
    print("3. 관측 데이터")
    print("=" * 70)
    df_all = load_curves(args.xlsx)
    po_single = select(df_all, "po_single")
    obs = select(df_all, "po_single", args.dose_mg)
    other = po_single[po_single["dose_mg"] != args.dose_mg]
    ss = select(df_all, "po_ss")
    iv = select(df_all, "iv")
    if obs.empty:
        raise SystemExit(f"경구 단회 {args.dose_mg:g} mg 곡선이 없다")

    def count(d):
        return (f"곡선 {d['curve_id'].nunique():2d}개 / 연구 {d['study'].nunique():2d}개"
                f" / {len(d):3d}점")

    print(f"  주 세트  경구 단회 {args.dose_mg:g} mg : {count(obs)}")
    print(f"  보조     경구 단회 다른 용량 : {count(other)}")
    print(f"  보조     경구 정상상태       : {count(ss)}")
    print(f"  보조     정맥                : {count(iv)}")
    print(f"  검증에 쓰지 않음 (이성질체별·특수군): {count(select(df_all, 'aux'))}")
    print("\n  싣지 않은 것:")
    for what, why in EXCLUDED.items():
        print(f"    시트 {what}: {why}")

    # 워크북에서 읽은 결과를 정리본으로 남긴다. 원본은 14 MB에 시트가
    # 39개이고 한 시트에 여러 군이 섞여 있어 사람이 읽기 어렵다. 이 CSV가
    # 코드가 실제로 쓰는 형태다.
    tidy_path = DATA_DIR / "propranolol_human_tidy.csv"
    df_all.to_csv(tidy_path, index=False, encoding="utf-8-sig")
    print(f"\n  정리본 저장: {tidy_path}")

    show = ["curve_id", "study", "arm", "cmax_obs", "cmax_sim", "cmax_fold",
            "tmax_obs", "tmax_sim", "auc_obs", "auc_sim", "auc_fold"]

    # --- 4. a priori 비교 --------------------------------------------------
    cmp_apriori = compare(par, obs, "a_priori", primary)
    print("\n" + "=" * 70)
    print(f"4. a priori 비교 — 주 세트 (경구 단회 {args.dose_mg:g} mg)")
    print("=" * 70)
    print(cmp_apriori[show].to_string(index=False))
    s_apriori = summarize(cmp_apriori)
    print()
    print_summary(s_apriori)

    # --- 5. 절편 하나만 보정 ------------------------------------------------
    print("\n" + "=" * 70)
    print("5. Caco-2 -> Peff 절편 하나만 보정 (곡선 하나로 맞추고 나머지로 검증)")
    print("=" * 70)
    counts = obs.groupby("curve_id").size().sort_values(ascending=False)
    calib_id = args.calib_curve or counts.index[0]
    calib = obs[obs["curve_id"] == calib_id]
    if calib.empty:
        raise SystemExit(f"주 세트에 곡선 {calib_id}가 없다: {list(counts.index)}")
    valid = obs[obs["curve_id"] != calib_id]
    print(f"  보정용 곡선: {calib_id} "
          f"({calib['study'].iloc[0]}, {calib['arm'].iloc[0]}, {len(calib)}점)")
    print(f"  검증용: {count(valid)}")

    # 절편을 바꾸면 ka가 바뀐다. 보정용 곡선의 Cmax를 맞추는 절편을 찾는다.
    target = observed_metrics(calib)
    calib_meta = curve_meta(calib)
    grid = np.linspace(PEFF_INTERCEPT, 1.5, 120)
    best, best_err = None, np.inf
    for inter in grid:
        m = simulated_metrics(make_params(float(inter)), calib_meta, target)
        err = abs(np.log(max(m["cmax_sim"], 1e-9) / target["cmax_obs"]))
        if err < best_err:
            best, best_err = float(inter), err
    par_cal = make_params(best)
    print(f"  절편 {PEFF_INTERCEPT} -> {best:.3f}  "
          f"(ka {par.ka_h:.3f} -> {par_cal.ka_h:.3f} 1/h, "
          f"Fa {par.fa:.3f} -> {par_cal.fa:.3f})")
    print(f"  생체이용률 F {par.bioavailability():.3f} -> {par_cal.bioavailability():.3f}")

    cmp_cal = compare(par_cal, valid, "calibrated_holdout", primary)
    print("\n  검증용 곡선(보정에 쓰지 않은 것들):")
    print(cmp_cal[show].to_string(index=False))
    s_cal = summarize(cmp_cal)
    print()
    print_summary(s_cal)

    # --- 6. 보조 세트 ------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. 보조 세트 — 다른 용량, 정상상태")
    print("=" * 70)
    extra, summaries = [], [s_apriori, s_cal]
    for name, d in [("po_single_other", other), ("po_ss", ss)]:
        if d.empty:
            continue
        for mode, p in [("a_priori", par), ("calibrated", par_cal)]:
            c = compare(p, d, mode, name)
            extra.append(c)
            summaries.append(summarize(c))
            print(f"\n  {name} / {mode}")
            print(c[["curve_id", "study", "dose_mg", "cmax_obs", "cmax_sim",
                     "cmax_fold", "auc_obs", "auc_sim", "auc_fold"]].to_string(index=False))
            print_summary(summaries[-1])

    # --- 7. IV 교차 확인 ----------------------------------------------------
    print("\n" + "=" * 70)
    print("7. IV 데이터로 소실만 따로 확인 (흡수 가정이 들어가지 않는다)")
    print("=" * 70)
    cmp_iv = iv_check(par, iv)
    if len(cmp_iv):
        print(cmp_iv[["curve_id", "study", "dose_mg", "t_first_h", "t_last_h",
                      "auc_obs", "auc_sim", "auc_fold", "thalf_obs_h",
                      "thalf_sim_h", "cl_obs_L_h", "cl_sim_L_h"]].to_string(index=False))
        s_iv = summarize(cmp_iv.assign(cmax_fold=np.nan))
        summaries.append(s_iv)
        print()
        print_summary(s_iv)
        print(f"    말단 반감기: 관측 {cmp_iv['thalf_obs_h'].min()}~"
              f"{cmp_iv['thalf_obs_h'].max()} h,"
              f" 같은 시점의 시뮬레이션 {cmp_iv['thalf_sim_h'].min()}~"
              f"{cmp_iv['thalf_sim_h'].max()} h")
        print(f"    청소율: 관측 {cmp_iv['cl_obs_L_h'].min()}~"
              f"{cmp_iv['cl_obs_L_h'].max()} L/h"
              f" (중앙값 {cmp_iv['cl_obs_L_h'].median()}),"
              f" 모델 {par.clearance_L_h():.1f} L/h")
    else:
        print("  IV 곡선이 없다")

    # --- 8. 그림 -----------------------------------------------------------
    fig_path = Path(args.out).with_name("pk_propranolol_po.png")
    plot_curves(par, par_cal, obs, calib_id, args.dose_mg, fig_path, df_iv=iv)
    print(f"\n그림 저장: {fig_path}")

    # --- 9. 저장 -----------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([cmp_apriori, cmp_cal, *extra, cmp_iv], ignore_index=True).to_csv(
        out, index=False)

    summary_path = out.with_name("pk_summary.csv")
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    params_path = out.with_name("pbpk_parameters.csv")
    pd.DataFrame([
        {"mode": "a_priori", "fu": par.fu, "clint_L_h": par.clint_L_h,
         "ka_h": par.ka_h, "fa": par.fa, "kp": par.kp_liver,
         "vss_L": par.vss_L(), "E_h": par.hepatic_extraction(),
         "F": par.bioavailability(), "CL_L_h": par.clearance_L_h()},
        {"mode": "calibrated", "fu": par_cal.fu, "clint_L_h": par_cal.clint_L_h,
         "ka_h": par_cal.ka_h, "fa": par_cal.fa, "kp": par_cal.kp_liver,
         "vss_L": par_cal.vss_L(), "E_h": par_cal.hepatic_extraction(),
         "F": par_cal.bioavailability(), "CL_L_h": par_cal.clearance_L_h()},
    ]).to_csv(params_path, index=False)

    print(f"\n저장: {out}")
    print(f"저장: {summary_path}")
    print(f"저장: {params_path}")


if __name__ == "__main__":
    main()
