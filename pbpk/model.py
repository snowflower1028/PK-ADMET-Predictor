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


def peff_from_caco2(log_papp_cm_s: float, intercept: float | None = None) -> float:
    """Caco-2 투과도(log Papp, cm/s) -> 사람 장벽 유효투과도 Peff(cm/s).

    **여기가 이 모델에서 가장 가정이 많은 지점이다.** Caco-2는 세포주
    단층이고 사람 소장은 아니다. 둘 사이에는 경험적 상관식을 쓴다.
    형태는 로그-로그 선형이다.

        log10 Peff = SLOPE * log10 Papp,Caco2 + INTERCEPT

    기울기 1보다 작은 값을 쓰는 이유: Caco-2는 능동수송과 점막 표면적을
    과소평가해서, 투과도가 낮은 약물일수록 사람 쪽이 상대적으로 높게 나온다.

    아래 두 상수는 **검증이 필요한 가정**이다. 출처 상관식을 확인해
    바꿀 수 있도록 상수로 빼 두었고, `run_pbpk.py`에서 민감도를 본다.
    """
    SLOPE = 0.6
    INTERCEPT = -1.9 if intercept is None else float(intercept)
    return float(10.0 ** (SLOPE * float(log_papp_cm_s) + INTERCEPT))


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
) -> DrugParams:
    """세 ADMET 예측값을 모델 파라미터로 바꾼다.

    logp는 RDKit이 구조에서 계산한 MolLogP다. 조직 분배계수를 구하는 데만
    쓰인다(세 ADMET 과제는 조직 분포를 말해 주지 않는다).

    peff_intercept를 주면 Caco-2 -> Peff 상관식의 절편을 그 값으로 바꾼다.
    보정 실험용이고, 기본값(None)이면 a priori 상수를 쓴다.
    """
    fu = fu_from_ppbr(ppbr_percent)
    clint = clint_liver_L_h(clearance_microsome, phys)
    peff = peff_from_caco2(caco2_log_papp, intercept=peff_intercept)
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
                          f" -> {clint:.1f} L/h"),
            "ka_h": (f"Caco2_Wang logPapp {caco2_log_papp:.3f}"
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
    if route == "PO":
        y0 = [dose_mg, 0.0, 0.0, 0.0]
    elif route == "IV":
        # 순간 주입으로 본다. 중앙 구획에 바로 넣는다.
        y0 = [0.0, 0.0, dose_mg / phys.v_central_L, 0.0]
    else:
        raise ValueError(f"알 수 없는 투여 경로: {route}")

    t_eval = np.linspace(0.0, t_end_h, n_points)
    sol = solve_ivp(
        _rhs, (0.0, t_end_h), y0, t_eval=t_eval,
        args=(params, phys),
        # 흡수는 빠르고 소실은 느려서 시간 척도가 섞인다(stiff).
        method="LSODA", rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"ODE 적분 실패: {sol.message}")

    # mg/L -> ug/L
    conc_ug_L = sol.y[2] * 1000.0
    return {
        "time_h": sol.t,
        "conc_ug_L": conc_ug_L,
        "cmax_ug_L": float(np.max(conc_ug_L)),
        "tmax_h": float(sol.t[int(np.argmax(conc_ug_L))]),
        # numpy 1.26에는 trapezoid가 없다(2.0에서 추가). rdkit 2023.09 상한
        # 때문에 numpy는 1.x에 묶여 있으므로 trapz를 쓴다.
        "auc_ug_h_L": float(np.trapz(conc_ug_L, sol.t)),
        "bioavailability": params.bioavailability(phys),
        "hepatic_extraction": params.hepatic_extraction(phys),
        "clearance_L_h": params.clearance_L_h(phys),
    }
