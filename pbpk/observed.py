"""관측 PK 데이터 읽기.

출판 논문에서 디지타이즈한 프로프라놀롤 human PK 워크북을 읽는다. 시트
하나가 논문 하나이고, 시트는 ARTICLE INFO / DATA / RAW DATA 블록으로 되어
있다.

**어떤 칸을 읽을지 추측하지 않고 아래 목록(`CURVES`)에 적는다.**

처음에는 'Time (' 헤더를 찾아 그 아래를 긁는 방식이었다. 그 방식은 이
워크북에서 네 군데가 조용히 틀렸다(2026-11-24 점검).

1. 한 열에 투여군이 세로로 쌓여 있으면 전부를 한 곡선으로 읽었다.
   남·녀(시트 63), 단회·정상상태(124), 대조·비만(131)이 합쳐졌다.
2. 시간 옆 **첫 번째** 농도 열만 읽었다. 이성질체별 열과 합(SUM) 열이
   나란히 있는 시트(64, 86, 87)에서는 한쪽 이성질체만 들어갔다.
3. DATA 블록과 RAW DATA 블록의 중복 제거가 마지막 시간이 반올림해서
   같을 때만 됐다. 디지타이즈 시간(23.97)과 명목 시간(24)이 다르면 같은
   곡선이 두 번 들어갔다.
4. 용량 칸이 'GROUP1'처럼 적힌 시트는 용량을 못 읽어 통째로 빠졌다.

그래서 곡선마다 시트·헤더 행·농도 열을 **엑셀 좌표 그대로** 적고, 투여
조건(경로·용량·단회/정상상태·대상군)도 같이 적는다. 워크북을 열어 한 줄씩
대조할 수 있다. 읽은 점 개수가 목록과 다르면 멈춘다.

읽는 쪽은 DATA 블록(E열부터)이다. RAW DATA 블록(T열부터)은 디지타이저
출력 그대로라 시간이 명목값과 어긋나고, 값이 틀린 시트도 있다(25, 83).

농도는 전부 ng/mL = ug/L로 적혀 있어 환산이 없다. 다른 단위로 적힌
유일한 시트(172, nmol/L)는 출처 미확인으로 빠져 있다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pbpk.model import PHYS
from src.config import DATA_DIR

DEFAULT_XLSX = DATA_DIR / "propranolol_human.xlsx"

TIME_COL = "E"  # DATA 블록의 시간 열. 모든 시트에서 같다.

# 체중당 용량(mg/kg)으로 투여한 연구를 총 용량으로 바꿀 때 쓰는 체중.
# 피험자 평균 체중이 아니라 **모델이 가정하는 사람**의 체중을 곱한다.
# 모델의 용적·혈류가 이 체중 기준이므로, 같은 mg/kg를 주면 농도가 같게
# 나온다. 피험자 체중으로 환산한 총 용량을 70 kg 모델에 넣으면 그만큼
# 농도가 어긋난다.
MODEL_BW_KG = PHYS.bw_kg


@dataclass(frozen=True)
class CurveSpec:
    """곡선 하나. 워크북의 어느 칸인지와 어떤 조건의 데이터인지."""

    curve_id: str
    sheet: str
    header_row: int          # 시간 헤더가 있는 엑셀 행 번호 (1부터)
    conc_col: str            # 농도 열 문자
    n_points: int            # 읽혀야 하는 점 개수
    dataset: str             # po_single / po_ss / iv / aux
    route: str               # PO / IV
    dose_mg: float
    arm: str                 # 시트 안에서 어느 군인지
    time_unit: str = "h"     # h / min
    analyte: str = "total"   # total 또는 시트 표기 그대로의 이성질체 이름
    population: str = "healthy"
    regimen: str = "single"  # single / steady_state
    tau_h: float | None = None   # 정상상태일 때 투여 간격
    infusion_h: float = 0.0      # IV 주입 시간. 0이면 순간 주입
    # 시트의 시간에 더하는 값. 시트의 0시간이 투여 시작이 아닐 때 쓴다.
    # 정리본의 time_h는 항상 **투여 시작부터** 잰 시간이다.
    time_offset_h: float = 0.0
    note: str = ""


def _mgkg(per_kg: float) -> float:
    return per_kg * MODEL_BW_KG


# 데이터셋
#   po_single  건강 성인, 속방형 경구 단회, 총 농도. 80 mg이 주 검증 세트다.
#   po_ss      경구 반복투여 정상상태, 총 농도.
#   iv         건강 성인 정맥 투여, 총 농도.
#   aux        검증에 쓰지 않지만 정리본에는 싣는 것 (이성질체별, 특수군).
CURVES: list[CurveSpec] = [
    # ---- 경구 단회 80 mg --------------------------------------------------
    CurveSpec("46", "46", 3, "F", 10, "po_single", "PO", 80, "healthy control"),
    CurveSpec("49", "49", 3, "F", 10, "po_single", "PO", 80, "healthy control"),
    CurveSpec("63-M", "63", 3, "F", 8, "po_single", "PO", 80, "male",
              note="중국인, 체중 52 kg"),
    CurveSpec("63-F", "63", 13, "F", 8, "po_single", "PO", 80, "female",
              note="중국인"),
    CurveSpec("64-PO80", "64", 4, "H", 16, "po_single", "PO", 80, "80 mg IR, S+R",
              note="시트 Amount 칸은 GROUP1. 초록: 12명 단회 4-way 교차"),
    CurveSpec("128", "128", 3, "F", 12, "po_single", "PO", 80, "mid-cycle (day 16)",
              note="여성 9명. 시트의 bid 표기와 달리 초록은 단회. 마지막 행 중복 제거"),
    CurveSpec("144", "144", 3, "F", 10, "po_single", "PO", 80,
              "without nicardipine", time_unit="min"),

    # ---- 경구 단회, 다른 용량 ---------------------------------------------
    CurveSpec("25-PO10", "25", 22, "F", 16, "po_single", "PO", 10, "PO",
              time_unit="min"),
    CurveSpec("76-PO10", "76", 20, "I", 12, "po_single", "PO", 10, "mean of 3",
              note="피험자 3명 평균"),
    CurveSpec("84-PO10", "84", 3, "F", 6, "po_single", "PO", 10, "10 mg"),
    CurveSpec("76-PO20", "76", 35, "I", 12, "po_single", "PO", 20, "mean of 3",
              note="피험자 3명 평균"),
    CurveSpec("84-PO20", "84", 12, "F", 6, "po_single", "PO", 20, "20 mg"),
    CurveSpec("105", "105", 3, "F", 9, "po_single", "PO", 20, "propranolol alone"),
    CurveSpec("23-PO40", "23", 24, "G", 13, "po_single", "PO", 40, "control",
              note="간경변 연구의 건강 대조군"),
    CurveSpec("66", "66", 3, "F", 12, "po_single", "PO", 40, "(군 표기 없음)",
              time_unit="min", note="대조·사우나 중 어느 군인지 시트에 없다"),
    CurveSpec("71", "71", 3, "F", 9, "po_single", "PO", 40, "propranolol"),
    CurveSpec("76-PO40", "76", 50, "I", 12, "po_single", "PO", 40, "mean of 3",
              note="피험자 3명 평균"),
    CurveSpec("64-PO160", "64", 23, "H", 16, "po_single", "PO", 160,
              "160 mg IR (2 x 80), S+R"),
    CurveSpec("124-SD", "124", 3, "F", 8, "po_single", "PO", 160, "acute dose",
              note="시트 Amount 칸은 80mg이지만 초록은 160 mg regular"),

    # ---- 경구 정상상태 80 mg ----------------------------------------------
    CurveSpec("86", "86", 34, "H", 11, "po_ss", "PO", 80, "control, d+l",
              time_unit="min", regimen="steady_state", tau_h=8.0),
    CurveSpec("87-B", "87", 4, "H", 10, "po_ss", "PO", 80, "Black, l+d",
              regimen="steady_state", tau_h=8.0, note="마지막 투여 후 12시간 관측"),
    CurveSpec("87-W", "87", 17, "H", 10, "po_ss", "PO", 80, "White, l+d",
              regimen="steady_state", tau_h=8.0, note="마지막 투여 후 12시간 관측"),
    CurveSpec("112", "112", 3, "F", 10, "po_ss", "PO", 80, "propranolol alone",
              population="angina patients", regimen="steady_state", tau_h=12.0,
              note="시트 Strain 칸은 Healthy지만 초록은 협심증 환자 5명, 4주 투여"),
    CurveSpec("126", "126", 3, "F", 11, "po_ss", "PO", 80, "placebo, day 8",
              time_unit="min", regimen="steady_state", tau_h=12.0),
    CurveSpec("151", "151", 3, "F", 8, "po_ss", "PO", 80, "control",
              regimen="steady_state", tau_h=12.0, note="마지막 투여 후 24시간 관측"),

    # ---- 정맥 -------------------------------------------------------------
    CurveSpec("23-IV1", "23", 4, "G", 15, "iv", "IV", 1, "control",
              note="주입 시간 미기재, 순간 주입으로 본다"),
    CurveSpec("64-IV10", "64", 42, "H", 19, "iv", "IV", 10, "10 mg bolus, S+R"),
    CurveSpec("76-IV", "76", 4, "I", 13, "iv", "IV", _mgkg(0.1), "mean of 3",
              infusion_h=20 / 60, note="0.1 mg/kg, 20분 주입. 피험자 3명 평균"),
    CurveSpec("131-IV", "131", 4, "F", 8, "iv", "IV", 10, "control",
              time_unit="min", infusion_h=10 / 60),
    # 시트 145의 시간은 주입 종료부터 잰 것으로 본다. 1 mg/min으로 7분쯤
    # 주입했는데 농도가 첫 채혈(1분)에서 가장 높고 그 뒤로 계속 떨어진다.
    # 주입 시작부터 잰 시간이라면 주입 중에 농도가 떨어질 수 없다.
    CurveSpec("145-IV", "145", 3, "H", 10, "iv", "IV", _mgkg(0.1), "(-) + (+)",
              time_unit="min", infusion_h=_mgkg(0.1) / 60,
              time_offset_h=_mgkg(0.1) / 60,
              note="0.1 mg/kg, 1 mg/min 주입. 시트 시간은 주입 종료 기준으로 보고"
                   " 주입 시간만큼 더했다"),

    # ---- 검증에 쓰지 않는 것 ----------------------------------------------
    # 이성질체별 농도. 분석물 이름은 시트 표기를 그대로 옮긴다.
    CurveSpec("64-PO80-S", "64", 4, "F", 16, "aux", "PO", 80, "80 mg IR", analyte="S"),
    CurveSpec("64-PO80-R", "64", 4, "G", 16, "aux", "PO", 80, "80 mg IR", analyte="R"),
    CurveSpec("64-PO160-S", "64", 23, "F", 16, "aux", "PO", 160, "160 mg IR", analyte="S"),
    CurveSpec("64-PO160-R", "64", 23, "G", 16, "aux", "PO", 160, "160 mg IR", analyte="R"),
    CurveSpec("64-IV10-S", "64", 42, "F", 19, "aux", "IV", 10, "10 mg bolus", analyte="S"),
    CurveSpec("64-IV10-R", "64", 42, "G", 19, "aux", "IV", 10, "10 mg bolus", analyte="R"),
    CurveSpec("86-d", "86", 34, "F", 11, "aux", "PO", 80, "control", time_unit="min",
              analyte="d", regimen="steady_state", tau_h=8.0),
    CurveSpec("86-l", "86", 34, "G", 11, "aux", "PO", 80, "control", time_unit="min",
              analyte="l", regimen="steady_state", tau_h=8.0),
    CurveSpec("87-B-l", "87", 4, "F", 10, "aux", "PO", 80, "Black", analyte="l",
              regimen="steady_state", tau_h=8.0),
    CurveSpec("87-B-d", "87", 4, "G", 10, "aux", "PO", 80, "Black", analyte="d",
              regimen="steady_state", tau_h=8.0),
    CurveSpec("87-W-l", "87", 17, "F", 10, "aux", "PO", 80, "White", analyte="l",
              regimen="steady_state", tau_h=8.0),
    CurveSpec("87-W-d", "87", 17, "G", 10, "aux", "PO", 80, "White", analyte="d",
              regimen="steady_state", tau_h=8.0),
    CurveSpec("145-IV-minus", "145", 3, "F", 10, "aux", "IV", _mgkg(0.1), "",
              time_unit="min", analyte="(-)", infusion_h=_mgkg(0.1) / 60,
              time_offset_h=_mgkg(0.1) / 60),
    CurveSpec("145-IV-plus", "145", 3, "G", 10, "aux", "IV", _mgkg(0.1), "",
              time_unit="min", analyte="(+)", infusion_h=_mgkg(0.1) / 60,
              time_offset_h=_mgkg(0.1) / 60),
    # 특수군과 다른 용량의 정상상태. 생리 파라미터나 투여 조건이 달라 같은
    # 세트로 묶지 않는다.
    CurveSpec("23-PO40-cirrhosis", "23", 24, "F", 13, "aux", "PO", 40, "patients",
              population="liver cirrhosis"),
    CurveSpec("23-IV1-cirrhosis", "23", 4, "F", 15, "aux", "IV", 1, "patients",
              population="liver cirrhosis"),
    CurveSpec("131-IV-obese", "131", 4, "G", 8, "aux", "IV", 10, "obese",
              time_unit="min", population="obese", infusion_h=10 / 60),
    CurveSpec("124-SS", "124", 16, "F", 9, "aux", "PO", 160, "steady state",
              regimen="steady_state", tau_h=24.0, note="160 mg 1일 1회 7일"),
]

# 워크북에 있지만 싣지 않는 것과 그 이유. 데이터가 왜 없는지를 코드에서
# 찾을 수 있게 남긴다.
EXCLUDED: dict[str, str] = {
    "129": "Williams 1986. 로그축 그림을 선형축으로 디지타이즈해 값이 틀리다"
           " (참값 = 10^(y/50)). 다시 디지타이즈할지 정해질 때까지 제외",
    "172": "출처 미확인. 제목·저자가 템플릿에서 남은 값이고, 그 논문에는"
           " 이 투여군(10 mg IV, 40/80/120 mg PO)이 없다",
    "118": "Straka 1987. 비결합 농도라 총 농도와 섞을 수 없다",
    "36": "Wojcicki 2003 대조군. 시트 46과 같은 곡선이다(시트 메모)",
    "83": "Bano 1991. 큐레이터 메모: 그래프가 신뢰하기 어려움",
    "25 (SL)": "설하 투여군. 모델에 없는 경로",
    "86 (diltiazem, verapamil)": "병용 약물군",
    "76 (GB, MG, RS)": "개별 피험자 열. 평균 열만 싣는다",
    "13, 34, 39, 74, 79, 109, 114, 116, 121, 147, 163":
        "그림만 있고 디지타이즈된 값이 없다 (특수군 또는 장기투여)",
}


# ---------------------------------------------------------------------------
# 읽기
# ---------------------------------------------------------------------------


def _col_index(letter: str) -> int:
    """엑셀 열 문자 -> 0부터 세는 열 번호."""
    n = 0
    for ch in letter.upper():
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def _label_value(raw: pd.DataFrame, name: str) -> str:
    """1열에서 라벨을 찾아 그 오른쪽 칸의 값을 돌려준다."""
    col0 = raw[0].astype(str).str.strip().str.lower()
    hit = col0[col0 == name.lower()]
    if hit.empty:
        return ""
    row = hit.index[0]
    for j in range(1, min(4, raw.shape[1])):
        v = raw.iat[row, j]
        if pd.notna(v) and str(v).strip():
            return str(v).strip()
    return ""


def _read_curve(raw: pd.DataFrame, spec: CurveSpec) -> pd.DataFrame:
    """목록에 적힌 칸에서 시간·농도를 읽는다.

    헤더 행 아래로 시간 칸이 숫자인 동안 내려간다. 다음 투여군의 제목이나
    빈 칸을 만나면 멈추므로 세로로 쌓인 군이 섞이지 않는다.
    """
    where = f"시트 {spec.sheet} {TIME_COL}{spec.header_row} (곡선 {spec.curve_id})"
    r0 = spec.header_row - 1
    tcol, ccol = _col_index(TIME_COL), _col_index(spec.conc_col)
    if r0 >= raw.shape[0] or ccol >= raw.shape[1]:
        raise ValueError(f"{where}: 시트 범위를 벗어난다")

    header = str(raw.iat[r0, tcol]).strip().lower()
    if not header.startswith("time"):
        raise ValueError(
            f"{where}: 시간 헤더가 있어야 하는데 '{raw.iat[r0, tcol]}'이다. "
            "워크북의 행이 밀렸는지 확인한다."
        )
    # 헤더에 단위가 적혀 있으면 목록과 맞는지 본다
    stated = "min" if "min" in header else ("h" if "(h" in header else None)
    if stated and stated != spec.time_unit:
        raise ValueError(f"{where}: 헤더 단위는 {stated}인데 목록은 {spec.time_unit}이다")

    times, concs = [], []
    for i in range(r0 + 1, raw.shape[0]):
        t = pd.to_numeric(raw.iat[i, tcol], errors="coerce")
        if pd.isna(t):
            break
        c = pd.to_numeric(raw.iat[i, ccol], errors="coerce")
        if pd.isna(c):
            continue  # 그 시점에 이 열만 비어 있는 경우 (평균 열 등)
        times.append(float(t) / (60.0 if spec.time_unit == "min" else 1.0)
                     + spec.time_offset_h)
        concs.append(float(c))

    d = pd.DataFrame({"time_h": times, "conc_ug_L": concs}).drop_duplicates()
    if len(d) != spec.n_points:
        raise ValueError(
            f"{where}: {spec.n_points}점이어야 하는데 {len(d)}점을 읽었다. "
            "워크북이나 목록 중 하나가 바뀌었다."
        )
    return d.reset_index(drop=True)


def load_curves(path=None) -> pd.DataFrame:
    """목록의 모든 곡선을 긴 표 한 장으로 돌려준다.

    한 행이 관측점 하나다. 곡선의 조건(경로·용량·투여법 등)이 행마다
    반복해서 붙는다. 저자·연도·저널은 시트의 ARTICLE INFO에서 읽는다.
    """
    path = Path(path or DEFAULT_XLSX)
    if not path.exists():
        raise FileNotFoundError(
            f"관측 데이터가 없다: {path}\n"
            "data/propranolol_human.xlsx 를 두어야 한다."
        )
    ids = [s.curve_id for s in CURVES]
    if len(ids) != len(set(ids)):
        raise ValueError("CURVES에 같은 curve_id가 두 번 있다")

    sheets = sorted({s.sheet for s in CURVES})
    raws = pd.read_excel(path, sheet_name=sheets, header=None)

    parts = []
    for spec in CURVES:
        raw = raws[spec.sheet]
        d = _read_curve(raw, spec)
        author = _label_value(raw, "Author")
        year = _label_value(raw, "Year")
        surname = author.rsplit(" ", 1)[0] if " " in author else author
        d.insert(0, "curve_id", spec.curve_id)
        d.insert(1, "dataset", spec.dataset)
        d.insert(2, "study", f"{surname} {year}".strip())
        d.insert(3, "sheet", spec.sheet)
        d.insert(4, "arm", spec.arm)
        d.insert(5, "analyte", spec.analyte)
        d.insert(6, "population", spec.population)
        d.insert(7, "route", spec.route)
        d.insert(8, "dose_mg", float(spec.dose_mg))
        d.insert(9, "regimen", spec.regimen)
        d.insert(10, "tau_h", float("nan") if spec.tau_h is None else float(spec.tau_h))
        d.insert(11, "infusion_h", float(spec.infusion_h))
        d["journal"] = _label_value(raw, "Journal")
        d["note"] = spec.note
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def select(df: pd.DataFrame, dataset: str, dose_mg: float | None = None) -> pd.DataFrame:
    """데이터셋 하나를 고른다. dose_mg를 주면 그 용량만."""
    out = df[df["dataset"] == dataset]
    if dose_mg is not None:
        out = out[out["dose_mg"] == float(dose_mg)]
    return out.copy()
