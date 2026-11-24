"""간·장 포함 축소 PBPK 모델.

구조 (경구 투여)

    장 내용물 ──ka──▶ 간 ──Qh──▶ 중앙(혈액) ◀──Qp──▶ 말초
                      │                  ▲
                      └── CLint·fu ─▶ 소실   │
                                   투여(IV) ┘

왜 이 구조인가. 세 ADMET 예측값이 각각 들어갈 자리가 생긴다.

    Caco2_Wang             → 장벽 투과도 → 흡수속도 ka, 흡수분율 Fa
    PPBR_AZ                → 혈장 비결합분율 fu
    Clearance_Microsome_AZ → 간 고유청소율 CLint → 간 추출률

흡수된 약물이 중앙 구획이 아니라 **간으로 먼저 들어간다**는 점이 핵심이다.
문맥순환을 타기 때문이다. 그래서 초회통과(first-pass)가 구조에서 자연히
나온다. 프로프라놀롤은 초회통과가 커서 경구 생체이용률이 낮은 약물이고
(문헌상 ~26%), 이 모델이 그걸 재현하는지가 주된 시험이 된다.

전신 PBPK(장기별 구획 10개 이상)로 가지 않은 이유는, 그만큼의 파라미터를
세 개의 예측값으로 뒷받침할 수 없기 때문이다. 구획을 늘리면 모델이
좋아지는 게 아니라 임의로 정한 숫자가 늘어난다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# 사람 생리 파라미터 (70 kg 성인)
#
# 이 값들은 예측하는 것이 아니라 생리학에서 가져오는 상수다. ADMET 모델이
# 예측하는 것은 약물 고유 성질(투과도·결합·대사)뿐이고, 생리는 주어진다.
# 그 구분이 PBPK의 요점이다.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Physiology:
    bw_kg: float = 70.0

    # 간
    liver_weight_g: float = 1800.0      # 성인 간 무게
    q_hepatic_L_h: float = 90.0         # 간 전체 혈류 (문맥 + 간동맥)
    v_liver_L: float = 1.8              # 간 용적
    mppgl_mg_g: float = 45.0            # 간 1 g당 미세소체 단백질 (mg)

    # 중앙·말초 구획
    #
    # 용적은 생리값이다. 말초 60 L는 70 kg 성인에서 혈액과 간을 뺀
    # 분포 가능 조직의 용적 수준이다. 여기에 조직 분배계수 Kp를 곱한 것이
    # 분포 용적에 들어간다: Vss = V_central + V_liver·Kp_l + V_peri·Kp_p.
    #
    # Kp를 상수로 손으로 정하지 않는다. 그렇게 하면 분포 용적을 관측에
    # 맞춰 끼워 넣는 것이 되어, 모델이 맞는 이유가 예측이 아니라 조정이
    # 된다. 아래 `kp_from_logp()`로 지질친화성과 fu에서 계산한다.
    v_central_L: float = 5.0            # 혈액 용적 수준
    v_peripheral_L: float = 60.0
    q_peripheral_L_h: float = 60.0      # 말초 조직 관류

    # 장
    gut_radius_cm: float = 1.75         # 소장 반지름 (표면적/용적 = 2/R)
    blood_plasma_ratio: float = 1.0     # 혈액/혈장 농도비


PHYS = Physiology()


# ---------------------------------------------------------------------------
# ADMET 예측값 -> 모델 파라미터
# ---------------------------------------------------------------------------


def fu_from_ppbr(ppbr_percent: float) -> float:
    """혈장단백결합률(%) -> 비결합분율.

    가정이 없는 유일한 변환이다. 정의상 fu = 1 - PPB.
    """
    fu = 1.0 - float(ppbr_percent) / 100.0
    # 결합률이 100%에 붙으면 fu가 0이 되어 소실이 사라진다. 측정 한계를
    # 생각하면 0으로 두는 것보다 작은 하한을 두는 게 안전하다.
    return float(np.clip(fu, 1e-4, 1.0))


def clint_liver_L_h(clint_microsome_uL_min_mg: float, phys: Physiology = PHYS) -> float:
    """마이크로솜 고유청소율(uL/min/mg protein) -> 간 전체 고유청소율(L/h).

    교과서적인 IVIVE 스케일업이다. 곱하는 값은 두 개뿐이고 둘 다 생리값이다.

        uL/min/mg  ×  MPPGL(mg/g liver)  ×  liver weight(g)  =  uL/min
        그 뒤 단위만 L/h로 바꾼다.

    MPPGL(간 1g당 미세소체 단백질)은 45 mg/g을 쓴다. 문헌 범위는 30~60이고,
    여기에 비례해서 결과가 움직인다. 민감도 분석 대상이다.
    """
    uL_min = float(clint_microsome_uL_min_mg) * phys.mppgl_mg_g * phys.liver_weight_g
    return uL_min * 60.0 / 1e6  # uL/min -> L/h


# Caco-2 -> Peff 상관식 상수. **둘 다 검증이 필요한 가정이다.**
# 민감도 분석(`scripts/run_sensitivity.py`)이 흔드는 대상이므로 모듈 상수로
# 빼 두고, 기본값을 쓰는 경로가 한 군데만 있게 한다.
PEFF_SLOPE = 0.6
PEFF_INTERCEPT = -1.9


def peff_from_caco2(
    log_papp_cm_s: float,
    intercept: float | None = None,
    slope: float | None = None,
) -> float:
    """Caco-2 투과도(log Papp, cm/s) -> 사람 장벽 유효투과도 Peff(cm/s).

    **여기가 이 모델에서 가장 가정이 많은 지점이다.** Caco-2는 세포주
    단층이고 사람 소장은 아니다. 둘 사이에는 경험적 상관식을 쓴다.
    형태는 로그-로그 선형이다.

        log10 Peff = SLOPE * log10 Papp,Caco2 + INTERCEPT

    기울기 1보다 작은 값을 쓰는 이유: Caco-2는 능동수송과 점막 표면적을
    과소평가해서, 투과도가 낮은 약물일수록 사람 쪽이 상대적으로 높게 나온다.

    아래 두 상수는 **검증이 필요한 가정**이다. 출처 상관식을 확인해
    바꿀 수 있도록 모듈 상수(PEFF_SLOPE, PEFF_INTERCEPT)로 빼 두었고,
    `scripts/run_sensitivity.py`에서 민감도를 본다.

    기울기가 특히 민감한 이유는 구조적이다. log Papp가 -4.5 근처이므로
    기울기는 **큰 음수에 곱해지는 지수**로 들어간다. 기울기를 0.1만
    바꿔도 Peff가 10^0.45 ≈ 2.8배 움직인다. 선형으로 들어가는 파라미터와
    같은 눈으로 보면 안 된다.
    """
    s = PEFF_SLOPE if slope is None else float(slope)
    b = PEFF_INTERCEPT if intercept is None else float(intercept)
    return float(10.0 ** (s * float(log_papp_cm_s) + b))


def ka_from_peff(peff_cm_s: float, phys: Physiology = PHYS) -> float:
    """Peff(cm/s) -> 1차 흡수속도상수 ka(1/h).

    원통 모양 장을 가정하면 표면적/용적 = 2/R 이다. 흡수속도상수는
    투과도에 이 비를 곱한 것이다.

        ka [1/s] = Peff [cm/s] × (2 / R [cm])
    """
    return float(peff_cm_s) * (2.0 / phys.gut_radius_cm) * 3600.0


def fa_from_ka(ka_h: float, transit_time_h: float = 3.0) -> float:
    """흡수속도상수 -> 흡수분율.

    소장 통과시간 동안 1차 흡수가 일어난다고 보면

        Fa = 1 - exp(-ka × T_transit)

    통과시간 3시간은 사람 소장의 통상값이다.
    """
    return float(1.0 - np.exp(-float(ka_h) * transit_time_h))


# 조직·혈장 조성 (Poulin & Theil 계열 조직 조성법의 단순형)
_F_NEUTRAL_LIPID_TISSUE = 0.06
_F_WATER_TISSUE = 0.72
_F_NEUTRAL_LIPID_PLASMA = 0.004
_F_WATER_PLASMA = 0.945


def kp_from_logp(logp: float, fu_plasma: float) -> float:
    """지질친화성과 혈장 비결합분율에서 조직 분배계수 Kp를 계산한다.

    조직 조성법의 단순형이다. 조직과 혈장을 각각 '중성지질 + 물'로 보고,
    약물이 지질에 분배되는 정도를 옥탄올/물 분배계수 P로 대신한다.

        Kp = [(P·f_lipid,조직 + f_water,조직) / (P·f_lipid,혈장 + f_water,혈장)]
             × (fu,혈장 / fu,조직)

    조직 쪽 비결합분율은 혈장 결합에서 추정한다(통상적 단순화).

    **중요한 한계**: 세 ADMET 과제(투과도·단백결합·대사)는 조직 분포에
    대해 거의 아무 말도 하지 않는다. 그래서 분포 용적은 이 모델에서
    가장 ADMET 예측에 기대지 않는 부분이고, logP는 RDKit이 구조에서
    계산한 값(descriptor `MolLogP`)을 쓴다. TDC 모델의 예측이 아니다.
    """
    p_ow = 10.0 ** float(logp)
    tissue = p_ow * _F_NEUTRAL_LIPID_TISSUE + _F_WATER_TISSUE
    plasma = p_ow * _F_NEUTRAL_LIPID_PLASMA + _F_WATER_PLASMA
    fu_p = float(np.clip(fu_plasma, 1e-4, 1.0))
    # 조직에서는 결합 단백질이 혈장의 절반 수준이라고 본다
    fu_tissue = 1.0 / (1.0 + (1.0 - fu_p) / fu_p * 0.5)
    return float((tissue / plasma) * (fu_p / fu_tissue))


@dataclass
class DrugParams:
    """모델이 받는 약물 파라미터. 출처를 함께 들고 다닌다."""

    fu: float                    # 혈장 비결합분율
    clint_L_h: float             # 간 고유청소율
    ka_h: float                  # 흡수속도상수
    fa: float                    # 흡수분율
    kp_liver: float = 5.0        # 간 분배계수
    kp_peripheral: float = 5.0   # 말초 분배계수
    f_gut: float = 1.0           # 장벽 대사로 인한 손실. 모델링하지 않는다.
    source: dict = field(default_factory=dict)  # 각 값이 어디서 왔는지

    def vss_L(self, phys: Physiology = PHYS) -> float:
        """정상상태 분포 용적. 관측에 맞춘 값이 아니라 계산 결과다."""
        return float(
            phys.v_central_L
            + phys.v_liver_L * self.kp_liver
            + phys.v_peripheral_L * self.kp_peripheral
        )

    def hepatic_extraction(self, phys: Physiology = PHYS) -> float:
        """well-stirred 모델의 간 추출률.

            E_h = fu·CLint / (Q_h + fu·CLint)

        혈액 기준으로 계산해야 하므로 혈액/혈장비로 보정한다.
        """
        fu_blood = self.fu / phys.blood_plasma_ratio
        fu_clint = fu_blood * self.clint_L_h
        return float(fu_clint / (phys.q_hepatic_L_h + fu_clint))

    def bioavailability(self, phys: Physiology = PHYS) -> float:
        """경구 생체이용률 F = Fa × Fg × (1 - E_h)."""
        return float(self.fa * self.f_gut * (1.0 - self.hepatic_extraction(phys)))

    def clearance_L_h(self, phys: Physiology = PHYS) -> float:
        """전신 혈액 청소율 (간 소실만 고려)."""
        return float(phys.q_hepatic_L_h * self.hepatic_extraction(phys))


def params_from_admet(
    caco2_log_papp: float,
    ppbr_percent: float,
    clearance_microsome: float,
    logp: float,
    phys: Physiology = PHYS,
    peff_intercept: float | None = None,
    peff_slope: float | None = None,
) -> DrugParams:
    """세 ADMET 예측값을 모델 파라미터로 바꾼다.

    logp는 RDKit이 구조에서 계산한 MolLogP다. 조직 분배계수를 구하는 데만
    쓰인다(세 ADMET 과제는 조직 분포를 말해 주지 않는다).

    peff_intercept / peff_slope를 주면 Caco-2 -> Peff 상관식의 절편·기울기를
    그 값으로 바꾼다. 절편은 보정 실험(`run_pbpk.py`), 기울기는 민감도
    분석(`run_sensitivity.py`)에서 쓴다. 기본값(None)이면 a priori 상수다.

    MPPGL은 `phys`에 들어 있다. 흔들려면 `dataclasses.replace(PHYS,
    mppgl_mg_g=...)`로 만든 Physiology를 넘긴다.
    """
    fu = fu_from_ppbr(ppbr_percent)
    clint = clint_liver_L_h(clearance_microsome, phys)
    peff = peff_from_caco2(
        caco2_log_papp, intercept=peff_intercept, slope=peff_slope
    )
    ka = ka_from_peff(peff, phys)
    fa = fa_from_ka(ka)
    kp = kp_from_logp(logp, fu)
    return DrugParams(
        fu=fu, clint_L_h=clint, ka_h=ka, fa=fa,
        kp_liver=kp, kp_peripheral=kp,
        source={
            "kp": f"MolLogP {logp:.2f}, fu {fu:.4f} -> Kp={kp:.2f} (RDKit 계산 logP)",
            "fu": f"PPBR_AZ {ppbr_percent:.2f}% -> fu={fu:.4f}",
            "clint_L_h": (f"Clearance_Microsome_AZ {clearance_microsome:.2f} uL/min/mg"
                          f" x MPPGL {phys.mppgl_mg_g:.0f} mg/g"
                          f" -> {clint:.1f} L/h"),
            "ka_h": (f"Caco2_Wang logPapp {caco2_log_papp:.3f}"
                     f" (slope {PEFF_SLOPE if peff_slope is None else peff_slope:.2f},"
                     f" intercept {PEFF_INTERCEPT if peff_intercept is None else peff_intercept:.2f})"
                     f" -> Peff {peff:.2e} cm/s -> ka {ka:.3f} 1/h"),
            "fa": f"ka {ka:.3f} 1/h, 통과시간 3h -> Fa={fa:.3f}",
        },
    )


# ---------------------------------------------------------------------------
# ODE
# ---------------------------------------------------------------------------


def _rhs(t, y, p: DrugParams, phys: Physiology):
    """상태변수: [장 내용물 양(mg), 간 농도, 중앙 농도, 말초 농도] (mg/L)"""
    a_gut, c_liver, c_central, c_peri = y

    # 장 -> 간 (문맥순환). 흡수분율만큼만 들어간다.
    absorption = p.ka_h * a_gut * p.fa * p.f_gut

    # 간을 떠나는 혈액의 농도
    c_liver_out = c_liver / p.kp_liver
    c_peri_out = c_peri / p.kp_peripheral

    # well-stirred 소실: 간을 떠나는 혈액의 비결합 농도에 비례
    fu_blood = p.fu / phys.blood_plasma_ratio
    elimination = p.clint_L_h * fu_blood * c_liver_out

    d_gut = -p.ka_h * a_gut
    d_liver = (
        phys.q_hepatic_L_h * c_central
        - phys.q_hepatic_L_h * c_liver_out
        + absorption
        - elimination
    ) / phys.v_liver_L
    d_central = (
        phys.q_hepatic_L_h * c_liver_out
        + phys.q_peripheral_L_h * c_peri_out
        - (phys.q_hepatic_L_h + phys.q_peripheral_L_h) * c_central
    ) / phys.v_central_L
    d_peri = (
        phys.q_peripheral_L_h * c_central
        - phys.q_peripheral_L_h * c_peri_out
    ) / phys.v_peripheral_L

    return [d_gut, d_liver, d_central, d_peri]


def simulate(
    params: DrugParams,
    dose_mg: float,
    route: str = "PO",
    t_end_h: float = 24.0,
    n_points: int = 600,
    phys: Physiology = PHYS,
    infusion_h: float = 0.0,
) -> dict:
    """혈중(중앙 구획) 농도-시간 곡선을 계산한다.

    돌려주는 농도 단위는 ug/L(= ng/mL)로, 관측 데이터와 같다.
    """
    route = route.upper()
    t_eval = np.linspace(0.0, t_end_h, n_points)

    def solve(rhs, t0, t1, y0, at):
        sol = solve_ivp(
            rhs, (t0, t1), y0, t_eval=at,
            args=(params, phys),
            # 흡수는 빠르고 소실은 느려서 시간 척도가 섞인다(stiff).
            method="LSODA", rtol=1e-8, atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(f"ODE 적분 실패: {sol.message}")
        return sol

    if route == "PO":
        y = solve(_rhs, 0.0, t_end_h, [dose_mg, 0.0, 0.0, 0.0], t_eval).y
    elif route == "IV" and infusion_h > 0:
        # 정속 주입. 주입이 끝나는 순간 입력이 끊기므로 그 시점에서 구간을
        # 나눠 푼다. 한 번에 풀면 적분기가 불연속을 넘다가 오차를 낸다.
        #
        # 순간 주입으로 대신하면 안 되는 이유: 중앙 구획이 5 L라 10 mg를
        # 한 번에 넣으면 시작 농도가 2000 ug/L가 되고, 이 뾰족한 구간이
        # 관측에는 없는 면적을 AUC에 더한다.
        rate = dose_mg / infusion_h

        def rhs_infusion(t, state, p, ph):
            d = _rhs(t, state, p, ph)
            d[2] += rate / ph.v_central_L
            return d

        t_stop = min(infusion_h, t_end_h)
        during = t_eval <= t_stop
        at = t_eval[during]
        if at[-1] < t_stop:
            at = np.append(at, t_stop)  # 주입 종료 시점의 상태가 필요하다
        first = solve(rhs_infusion, 0.0, t_stop, [0.0, 0.0, 0.0, 0.0], at)
        y = first.y[:, : int(during.sum())]
        if not during.all():
            rest = solve(_rhs, t_stop, t_end_h, first.y[:, -1], t_eval[~during])
            y = np.hstack([y, rest.y])
    elif route == "IV":
        # 순간 주입. 중앙 구획에 바로 넣는다.
        y = solve(_rhs, 0.0, t_end_h,
                  [0.0, 0.0, dose_mg / phys.v_central_L, 0.0], t_eval).y
    else:
        raise ValueError(f"알 수 없는 투여 경로: {route}")

    # mg/L -> ug/L
    conc_ug_L = y[2] * 1000.0
    return {
        "time_h": t_eval,
        "conc_ug_L": conc_ug_L,
        "cmax_ug_L": float(np.max(conc_ug_L)),
        "tmax_h": float(t_eval[int(np.argmax(conc_ug_L))]),
        # numpy 1.26에는 trapezoid가 없다(2.0에서 추가). rdkit 2023.09 상한
        # 때문에 numpy는 1.x에 묶여 있으므로 trapz를 쓴다.
        "auc_ug_h_L": float(np.trapz(conc_ug_L, t_eval)),
        "bioavailability": params.bioavailability(phys),
        "hepatic_extraction": params.hepatic_extraction(phys),
        "clearance_L_h": params.clearance_L_h(phys),
    }


def simulate_steady_state(
    params: DrugParams,
    dose_mg: float,
    tau_h: float,
    t_end_h: float | None = None,
    n_points: int = 600,
    phys: Physiology = PHYS,
) -> dict:
    """경구 반복투여 정상상태에서 **마지막 투여 뒤**의 곡선.

    중첩 원리로 계산한다. 이 모델은 선형이라(포화되는 과정이 없다) 단회
    곡선을 투여 간격만큼 밀어 더한 것이 반복투여 곡선과 정확히 같다.

        C_ss(t) = sum_k C_single(t + k·tau),   k = 0, 1, 2, …

    t가 tau를 넘어가면 다음 투여 없이 계속 떨어지는 구간이 된다. 정상상태
    연구는 마지막 투여 뒤 12~24시간을 관측하는 경우가 있어서 그 구간이
    필요하다. t_end_h를 주지 않으면 한 투여 간격만 계산한다.

    **한계**: 실제 프로프라놀롤은 반복투여에서 초회통과가 줄어 선형
    예측보다 더 쌓인다(문헌의 축적비 1.4~1.7). 이 함수는 그 비선형을
    담지 못한다. 관측이 예측보다 높게 나오면 이쪽을 먼저 의심한다.
    """
    t_end = float(tau_h if t_end_h is None else t_end_h)
    step = 0.01  # 단회 곡선의 격자 간격(h). 봉우리를 보간으로 뭉개지 않을 만큼.
    horizon = max(240.0, 20.0 * tau_h)
    while True:
        total = t_end + horizon
        single = simulate(params, dose_mg, "PO", t_end_h=total,
                          n_points=int(round(total / step)) + 1, phys=phys)
        # 꼬리가 충분히 죽었는지 본다. 남아 있으면 더 길게 다시 푼다.
        if single["conc_ug_L"][-1] <= 1e-6 * single["cmax_ug_L"]:
            break
        if horizon > 5000.0:
            raise RuntimeError("정상상태에 도달하지 않는다 (소실이 거의 없다)")
        horizon *= 2.0

    t = np.linspace(0.0, t_end, n_points)
    conc = np.zeros_like(t)
    for k in range(int(horizon // tau_h) + 1):
        conc += np.interp(t + k * tau_h, single["time_h"], single["conc_ug_L"])
    return {
        "time_h": t,
        "conc_ug_L": conc,
        "cmax_ug_L": float(np.max(conc)),
        "tmax_h": float(t[int(np.argmax(conc))]),
        "auc_ug_h_L": float(np.trapz(conc, t)),
        "accumulation_cmax": float(np.max(conc) / single["cmax_ug_L"]),
    }
