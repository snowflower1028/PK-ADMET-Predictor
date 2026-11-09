"""3주차: ADMET 예측값을 축소 PBPK에 넣고 관측 human PK와 비교한다.

  python scripts/run_pbpk.py

두 가지를 나눠서 돌린다.

1. **a priori** — 세 ADMET 예측값을 그대로 파라미터로 바꿔 넣는다.
   관측 데이터를 전혀 보지 않는다. 이게 "예측"이다.

2. **보정(calibration)** — Caco-2 -> Peff 상관식의 절편 하나만 **한 연구**에
   맞추고, 나머지 연구로 검증한다. 상관식 상수가 문헌 확인이 필요한
   가정이기 때문에, 그 하나를 데이터가 정하게 하면 얼마나 나아지는지
   본다. 맞춘 연구는 검증에서 뺀다. 전부에 맞추면 그건 검증이 아니다.

관측 데이터 원본 엑셀은 data/propranolol_human.xlsx에 있다. 코드로
다시 만들 수 없는 데이터라 저장소에 함께 넣어 두었다.
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

from pbpk.model import PHYS, params_from_admet, simulate  # noqa: E402
from pbpk.observed import healthy_oral, load_studies, tidy  # noqa: E402
from pbpk.predict_admet import predict_all  # noqa: E402
from src.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data import get_group  # noqa: E402

PROPRANOLOL = "CC(C)NCC(O)COc1cccc2ccccc12"


def observed_metrics(d: pd.DataFrame) -> dict:
    """한 연구의 관측 Cmax/Tmax/AUC(마지막 관측시점까지)."""
    d = d.sort_values("time_h")
    i = d["conc_ug_L"].idxmax()
    return {
        "cmax_obs": float(d.loc[i, "conc_ug_L"]),
        "tmax_obs": float(d.loc[i, "time_h"]),
        "auc_obs": float(np.trapz(d["conc_ug_L"], d["time_h"])),
        "t_last": float(d["time_h"].max()),
        "n_points": int(len(d)),
    }


def simulated_metrics(par, dose_mg: float, t_last: float) -> dict:
    """같은 시간 창에서 시뮬레이션 Cmax/Tmax/AUC.

    AUC를 관측과 같은 창에서 끊어야 비교가 성립한다. 시뮬레이션은
    무한대까지 적분할 수 있지만 관측은 마지막 채혈까지만 있다.
    """
    sim = simulate(par, dose_mg=dose_mg, route="PO", t_end_h=t_last, n_points=1200)
    return {
        "cmax_sim": sim["cmax_ug_L"],
        "tmax_sim": sim["tmax_h"],
        "auc_sim": sim["auc_ug_h_L"],
    }


def fold_error(sim: float, obs: float) -> float:
    """배수 오차. 1.0이 완벽, 2.0은 2배 틀림, 0.5도 2배 틀림이다.

    PK에서는 보통 2배 안쪽이면 받아들일 만하다고 본다.
    """
    if obs <= 0:
        return float("nan")
    return float(sim / obs)


def compare(par, studies_df: pd.DataFrame, dose_mg: float, label: str) -> pd.DataFrame:
    rows = []
    for (sheet, author, year), d in studies_df.groupby(["sheet", "author", "year"]):
        obs = observed_metrics(d)
        sim = simulated_metrics(par, dose_mg, obs["t_last"])
        rows.append({
            "mode": label,
            "sheet": sheet, "author": author, "year": year,
            "n_points": obs["n_points"], "t_last_h": round(obs["t_last"], 1),
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


def summarize(df: pd.DataFrame, label: str) -> dict:
    """기하평균 배수오차와 2배 안쪽 비율.

    배수오차는 기하평균으로 모아야 한다. 2배 큰 것과 2배 작은 것이
    산술평균에서는 상쇄되지 않고 한쪽으로 치우친다.
    """
    def gmfe(series):
        v = series.dropna()
        v = v[v > 0]
        return float(np.exp(np.mean(np.abs(np.log(v))))) if len(v) else float("nan")

    within2 = lambda s: float(np.mean((s >= 0.5) & (s <= 2.0)) * 100)
    return {
        "mode": label,
        "n_studies": int(len(df)),
        "cmax_gmfe": round(gmfe(df["cmax_fold"]), 3),
        "cmax_within_2fold_pct": round(within2(df["cmax_fold"]), 1),
        "auc_gmfe": round(gmfe(df["auc_fold"]), 3),
        "auc_within_2fold_pct": round(within2(df["auc_fold"]), 1),
        "tmax_median_obs": round(float(df["tmax_obs"].median()), 2),
        "tmax_sim": round(float(df["tmax_sim"].median()), 2),
    }


def plot_po_curves(par, par_cal, obs: pd.DataFrame, dose_mg: float,
                   path: Path) -> None:
    """경구. 관측 점 위에 a priori와 보정 곡선을 겹쳐 그린다."""
    import matplotlib
    matplotlib.use("Agg")  # 화면 없는 환경에서 그린다
    import matplotlib.pyplot as plt

    t_end = float(min(obs["time_h"].max(), 24.0))
    sim_a = simulate(par, dose_mg, "PO", t_end, n_points=800)
    sim_c = simulate(par_cal, dose_mg, "PO", t_end, n_points=800)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, logy in zip(axes, (False, True)):
        for sheet, d in obs.groupby("sheet"):
            ax.scatter(d["time_h"], d["conc_ug_L"], s=14, alpha=0.45,
                       color="0.45", label="_nolegend_")
        ax.plot(sim_a["time_h"], sim_a["conc_ug_L"], lw=2.2,
                label="a priori (ADMET only)")
        ax.plot(sim_c["time_h"], sim_c["conc_ug_L"], lw=2.2, ls="--",
                label="1-parameter calibrated")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Plasma concentration (ug/L)")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1, None)
            ax.set_title("log scale")
        else:
            ax.set_title(f"Propranolol {dose_mg:.0f} mg PO, healthy adults "
                         f"({obs['sheet'].nunique()} studies)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_iv_curves(par, df_all: pd.DataFrame, path: Path) -> None:
    """IV. 관측 점 위에 a priori 곡선만 겹쳐 그린다.

    보정 곡선을 그리지 않는 것은 게으름이 아니다. 보정한 파라미터는 ka와
    Fa뿐이고 둘 다 흡수 쪽이라, IV에서는 a priori와 보정 곡선이 완전히
    겹친다. 그래서 이 그림은 흡수 가정이 전혀 들어가지 않은 상태의
    소실(fu, CLint)만 보여 준다. 경구에서 어긋난 것이 흡수 때문이라는
    주장의 근거가 이 그림이다.
    """
    import matplotlib
    matplotlib.use("Agg")  # 화면 없는 환경에서 그린다
    import matplotlib.pyplot as plt

    iv = df_all[(df_all["route"] == "IV") & (df_all["conc_ug_L"] > 0)].copy()
    iv = iv[iv["dose_mg"].notna() & np.isfinite(iv["dose_mg"])]
    if iv.empty:
        print("  IV 관측이 없어 그림을 건너뛴다")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, logy in zip(axes, (False, True)):
        labels = []
        for (sheet, author, year), d in iv.groupby(["sheet", "author", "year"]):
            dose = float(d["dose_mg"].iloc[0])
            d = d.sort_values("time_h")
            ax.scatter(d["time_h"], d["conc_ug_L"], s=22, alpha=0.75,
                       color="0.25", zorder=3,
                       label=f"observed {author} {year} ({dose:.0f} mg IV)")
            sim = simulate(par, dose_mg=dose, route="IV",
                           t_end_h=float(d["time_h"].max()), n_points=1200)
            ax.plot(sim["time_h"], sim["conc_ug_L"], lw=2.2,
                    label="a priori (ADMET only)")
            labels.append((author, year, dose))
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Plasma concentration (ug/L)")
        if logy:
            ax.set_yscale("log")
            ax.set_title("log scale")
        else:
            # 순간 주입이라 t=0의 곡선 값이 관측 최고점의 20배쯤 된다.
            # 그대로 두면 선형 축이 그 한 점에 눌려 나머지가 안 보인다.
            ax.set_ylim(0, float(iv["conc_ug_L"].max()) * 1.3)
            doses = ", ".join(f"{a} {y} {dm:.0f} mg" for a, y, dm in labels)
            ax.set_title(f"Propranolol IV, healthy adults ({doses})")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def iv_check(par, df_all: pd.DataFrame) -> pd.DataFrame:
    """IV 데이터로 청소율·분포를 따로 확인한다.

    IV는 흡수를 거치지 않으므로 Caco-2 쪽 가정이 들어가지 않는다. 그래서
    청소율과 분포 용적만 떼어 평가할 수 있다. 경구에서 어긋난 것이
    흡수 때문인지 소실 때문인지 가르는 데 쓴다.
    """
    rows = []
    iv = df_all[(df_all["route"] == "IV") & (df_all["conc_ug_L"] > 0)]
    for (sheet, author, year), d in iv.groupby(["sheet", "author", "year"]):
        dose = d["dose_mg"].iloc[0]
        if not dose or not np.isfinite(dose):
            continue  # mg/kg 표기는 총 용량을 모른다
        d = d.sort_values("time_h")
        auc_obs = float(np.trapz(d["conc_ug_L"], d["time_h"]))
        sim = simulate(par, dose_mg=float(dose), route="IV",
                       t_end_h=float(d["time_h"].max()), n_points=1200)
        rows.append({
            "mode": "iv_check", "sheet": sheet, "author": author, "year": year,
            "dose_mg": dose, "n_points": len(d),
            "auc_obs": round(auc_obs, 1), "auc_sim": round(sim["auc_ug_h_L"], 1),
            "auc_fold": round(fold_error(sim["auc_ug_h_L"], auc_obs), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=None, help="관측 데이터 엑셀 경로")
    ap.add_argument("--dose-mg", type=float, default=80.0)
    ap.add_argument("--calib-sheet", default=None,
                    help="보정에 쓸 연구 시트. 기본은 관측점이 가장 많은 연구")
    ap.add_argument("--out", default=str(RESULTS_DIR / "pk_comparison.csv"))
    args = ap.parse_args()

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

    # --- 2. a priori 파라미터 ---------------------------------------------
    print("\n" + "=" * 70)
    print("2. PBPK 파라미터 (a priori — 관측을 보지 않았다)")
    print("=" * 70)
    par = params_from_admet(
        caco2_log_papp=preds["Caco2_Wang"].value,
        ppbr_percent=preds["PPBR_AZ"].value,
        clearance_microsome=preds["Clearance_Microsome_AZ"].value,
        logp=logp,
    )
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
    df_all = tidy(load_studies(args.xlsx))
    obs = healthy_oral(df_all, args.dose_mg)
    print(f"  전체 {len(df_all)}점 / 건강 성인 경구 {args.dose_mg:.0f}mg "
          f"{len(obs)}점, 연구 {obs['sheet'].nunique()}개")

    # 엑셀을 파싱한 결과를 정리본으로 남긴다. 원본 엑셀은 14MB에 시트가
    # 39개라 사람이 읽기 어렵고, 시간 단위가 섞여 있어 그대로 쓰면 틀린다.
    # 이 CSV가 코드가 실제로 쓰는 형태다.
    tidy_path = DATA_DIR / "propranolol_human_tidy.csv"
    df_all.to_csv(tidy_path, index=False, encoding="utf-8-sig")
    print(f"  정리본 저장: {tidy_path}")

    # --- 4. a priori 비교 --------------------------------------------------
    cmp_apriori = compare(par, obs, args.dose_mg, "a_priori")
    print("\n" + "=" * 70)
    print("4. a priori 비교")
    print("=" * 70)
    print(cmp_apriori[["sheet", "author", "year", "cmax_obs", "cmax_sim",
                       "cmax_fold", "tmax_obs", "tmax_sim",
                       "auc_obs", "auc_sim", "auc_fold"]].to_string(index=False))
    s_apriori = summarize(cmp_apriori, "a_priori")
    print(f"\n  Cmax 기하평균 배수오차 {s_apriori['cmax_gmfe']}, "
          f"2배 안쪽 {s_apriori['cmax_within_2fold_pct']}%")
    print(f"  AUC  기하평균 배수오차 {s_apriori['auc_gmfe']}, "
          f"2배 안쪽 {s_apriori['auc_within_2fold_pct']}%")

    # --- 5. 절편 하나만 보정 ------------------------------------------------
    print("\n" + "=" * 70)
    print("5. Caco-2 -> Peff 절편 하나만 보정 (한 연구로 맞추고 나머지로 검증)")
    print("=" * 70)
    counts = obs.groupby("sheet").size().sort_values(ascending=False)
    calib_sheet = args.calib_sheet or counts.index[0]
    calib = obs[obs["sheet"] == calib_sheet]
    valid = obs[obs["sheet"] != calib_sheet]
    print(f"  보정용 연구: sheet {calib_sheet} "
          f"({calib['author'].iloc[0]} {calib['year'].iloc[0]}, {len(calib)}점)")
    print(f"  검증용: 연구 {valid['sheet'].nunique()}개")

    # 절편을 바꾸면 ka가 바뀐다. 보정용 연구의 Cmax를 맞추는 절편을 찾는다.
    target = observed_metrics(calib)
    grid = np.linspace(-1.9, 1.5, 120)
    best, best_err = None, np.inf
    for inter in grid:
        p2 = params_from_admet(
            preds["Caco2_Wang"].value, preds["PPBR_AZ"].value,
            preds["Clearance_Microsome_AZ"].value, logp=logp,
            peff_intercept=float(inter),
        )
        m = simulated_metrics(p2, args.dose_mg, target["t_last"])
        err = abs(np.log(max(m["cmax_sim"], 1e-9) / target["cmax_obs"]))
        if err < best_err:
            best, best_err = float(inter), err
    par_cal = params_from_admet(
        preds["Caco2_Wang"].value, preds["PPBR_AZ"].value,
        preds["Clearance_Microsome_AZ"].value, logp=logp,
        peff_intercept=best,
    )
    print(f"  절편 {-1.9} -> {best:.3f}  (ka {par.ka_h:.3f} -> {par_cal.ka_h:.3f} 1/h, "
          f"Fa {par.fa:.3f} -> {par_cal.fa:.3f})")
    print(f"  생체이용률 F {par.bioavailability():.3f} -> {par_cal.bioavailability():.3f}")

    cmp_cal = compare(par_cal, valid, args.dose_mg, "calibrated_holdout")
    print("\n  검증용 연구(보정에 쓰지 않은 것들):")
    print(cmp_cal[["sheet", "author", "year", "cmax_obs", "cmax_sim", "cmax_fold",
                   "tmax_obs", "tmax_sim", "auc_fold"]].to_string(index=False))
    s_cal = summarize(cmp_cal, "calibrated_holdout")
    print(f"\n  Cmax 기하평균 배수오차 {s_cal['cmax_gmfe']}, "
          f"2배 안쪽 {s_cal['cmax_within_2fold_pct']}%")
    print(f"  AUC  기하평균 배수오차 {s_cal['auc_gmfe']}, "
          f"2배 안쪽 {s_cal['auc_within_2fold_pct']}%")

    # --- 6. IV 교차 확인 ----------------------------------------------------
    print("\n" + "=" * 70)
    print("6. IV 데이터로 소실만 따로 확인 (흡수 가정이 들어가지 않는다)")
    print("=" * 70)
    cmp_iv = iv_check(par, df_all)
    if len(cmp_iv):
        print(cmp_iv.to_string(index=False))
    else:
        print("  총 용량이 mg으로 적힌 IV 연구가 없다 (mg/kg 표기는 건너뛴다)")

    # --- 7. 그림 -----------------------------------------------------------
    po_fig = Path(args.out).with_name("pk_propranolol_po.png")
    plot_po_curves(par, par_cal, obs, args.dose_mg, po_fig)
    print(f"\n그림 저장: {po_fig}")

    # IV는 흡수 가정이 들어가지 않으므로 경구와 섞어 그리지 않고 따로 낸다.
    iv_fig = Path(args.out).with_name("pk_propranolol_iv.png")
    plot_iv_curves(par, df_all, iv_fig)
    if iv_fig.exists():
        print(f"그림 저장: {iv_fig}")

    # --- 8. 저장 -----------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([cmp_apriori, cmp_cal, cmp_iv], ignore_index=True).to_csv(
        out, index=False)

    summary = pd.DataFrame([s_apriori, s_cal])
    summary_path = out.with_name("pk_summary.csv")
    summary.to_csv(summary_path, index=False)

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
