"""관측 PK 데이터 읽기.

출판 논문에서 디지타이즈한 프로프라놀롤 human PK 데이터를 받는다.
엑셀 한 파일에 연구마다 시트가 하나씩 들어 있고, 각 시트는
ARTICLE INFO / ANIMAL INFO / DOSING INFO / RAW DATA 블록으로 되어 있다.

이 엑셀은 출판 논문에서 디지타이즈한 것이라 코드로 다시 만들 수 없다.
그래서 TDC 캐시와 달리 원본과 정리본을 저장소에 함께 넣는다
(data/propranolol_human.xlsx, data/propranolol_human_tidy.csv).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR

DEFAULT_XLSX = DATA_DIR / "propranolol_human.xlsx"

# 시트의 1열에 있는 라벨들. 값은 2~3열에 들어 있다.
_LABELS = [
    "Title", "Author", "Journal", "Year",
    "Species", "Strain", "BW", "N", "Analysis",
    "Dosing", "Amount", "Isomer",
]


@dataclass
class Study:
    """한 연구(시트 하나)."""

    sheet: str
    info: dict = field(default_factory=dict)
    curves: list[pd.DataFrame] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return int(sum(len(c) for c in self.curves))

    def __repr__(self) -> str:
        i = self.info
        return (f"Study({self.sheet}, {i.get('Author','?')} {i.get('Year','?')}, "
                f"{i.get('Dosing','?')} {i.get('Amount','?')}, {self.n_points} pts)")


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


# 프로프라놀롤 단회 투여는 반감기가 3~6시간이고, 관측은 길어도 48시간을
# 넘지 않는다. 그보다 큰 시간 값이 나오면 단위가 분이다. 헤더를 믿지 않고
# 데이터로 판단하는 이유는, 실제로 'Time (h)'라고 적힌 블록에 분이 들어
# 있는 시트가 있기 때문이다(sheet 126: tmax 717.8 "h").
_MAX_PLAUSIBLE_HOURS = 100.0


def _extract_curves(raw: pd.DataFrame) -> list[pd.DataFrame]:
    """'Time (' 헤더를 찾아 그 아래 시간·농도 쌍을 긁어온다.

    이 엑셀은 같은 곡선을 두 번 담고 있다. DATA 블록은 논문 단위 그대로,
    RAW DATA 블록은 시간/ug-L로 정리한 것이다. 둘 다 긁은 뒤 중복을
    제거한다. 반면 'S-PPL', 'Control', 'Patients', '120mg po' 같은
    헤더는 **서로 다른 투여군**이므로 남겨야 한다.
    """
    curves = []
    for j in range(raw.shape[1]):
        col = raw[j].astype(str).str.strip()
        hits = col[col.str.startswith("Time (")]
        if hits.empty or j + 1 >= raw.shape[1]:
            continue
        r0 = hits.index[0]
        unit_time = col.iloc[r0]
        unit_conc = str(raw.iat[r0, j + 1]).strip()
        t = pd.to_numeric(raw.iloc[r0 + 1:, j], errors="coerce")
        c = pd.to_numeric(raw.iloc[r0 + 1:, j + 1], errors="coerce")
        ok = t.notna() & c.notna()
        if not ok.any():
            continue
        t = t[ok].to_numpy(dtype=float)
        c = c[ok].to_numpy(dtype=float)

        # 시간 단위를 시간으로 맞춘다. 헤더보다 데이터를 우선한다.
        said_min = "min" in unit_time.lower()
        looks_min = float(np.nanmax(t)) > _MAX_PLAUSIBLE_HOURS
        if said_min or looks_min:
            t = t / 60.0
        curves.append(pd.DataFrame({
            "time_h": t,
            "conc": c,
            "unit_time": unit_time,
            "unit_conc": unit_conc,
            "time_converted": said_min or looks_min,
        }))

    return _dedupe(curves)


def _dedupe(curves: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """같은 곡선의 두 사본 중 하나만 남긴다.

    같다고 보는 기준: 점 개수가 같고, 최대 농도와 시간 범위가 거의 같다.
    단위가 ug/L로 적힌 쪽을 남긴다(이미 정리된 블록).
    """
    kept: list[pd.DataFrame] = []
    for cur in curves:
        sig = (len(cur), round(float(cur["conc"].max()), 4),
               round(float(cur["time_h"].max()), 3))
        dup_at = None
        for i, k in enumerate(kept):
            if (len(k), round(float(k["conc"].max()), 4),
                    round(float(k["time_h"].max()), 3)) == sig:
                dup_at = i
                break
        if dup_at is None:
            kept.append(cur)
            continue
        # 중복이면 ug/L 표기를 가진 쪽을 남긴다
        new_is_ugl = "ug/l" in cur["unit_conc"].iloc[0].lower()
        old_is_ugl = "ug/l" in kept[dup_at]["unit_conc"].iloc[0].lower()
        if new_is_ugl and not old_is_ugl:
            kept[dup_at] = cur
    return kept


def load_studies(path=None) -> list[Study]:
    """엑셀을 읽어 Study 목록으로 돌려준다. 곡선이 없는 시트는 버린다."""
    path = Path(path or DEFAULT_XLSX)
    if not path.exists():
        raise FileNotFoundError(
            f"관측 데이터가 없다: {path}\n"
            "원본 엑셀은 저장소에 넣지 않으므로 data/ 에 직접 두어야 한다."
        )
    xl = pd.ExcelFile(path)
    studies = []
    for sheet in xl.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if raw.shape[1] < 2:
            continue
        curves = _extract_curves(raw)
        if not curves:
            continue  # 메타데이터만 있고 곡선이 없는 시트
        info = {k: _label_value(raw, k) for k in _LABELS}
        studies.append(Study(sheet=sheet, info=info, curves=curves))
    return studies


def _parse_mg(amount: str) -> float | None:
    """'80mg', '80 mg', '0.1mg/kg' 같은 문자열에서 mg 용량을 뽑는다.

    체중당 용량(mg/kg)은 총 용량을 모르니 None으로 둔다. 애매한 값을
    70kg으로 임의 환산하면 조용히 틀린 숫자가 된다.
    """
    s = str(amount).lower().replace(" ", "")
    if not s or "/kg" in s:
        return None
    import re

    m = re.search(r"(\d+(?:\.\d+)?)mg", s)
    return float(m.group(1)) if m else None


def tidy(studies: list[Study]) -> pd.DataFrame:
    """연구 목록을 한 장의 긴 표로 만든다.

    농도 단위는 ug/L(= ng/mL)로 맞춘다. 프로프라놀롤 경구 80mg의
    Cmax는 수십 ng/mL 수준이라 ug/L가 다루기 편하다.
    """
    rows = []
    for st in studies:
        dose_mg = _parse_mg(st.info.get("Amount", ""))
        route = "IV" if "iv" in st.info.get("Dosing", "").lower() else "PO"
        for k, curve in enumerate(st.curves):
            unit = curve["unit_conc"].iloc[0].lower()
            scale = 1.0
            if "ng/ml" in unit or "ug/l" in unit or "µg/l" in unit:
                scale = 1.0
            elif "ug/ml" in unit or "µg/ml" in unit or "mg/l" in unit:
                scale = 1000.0
            elif "nmol/l" in unit or "nm" == unit.strip():
                scale = 0.2593  # 프로프라놀롤 MW 259.3 g/mol -> nmol/L to ug/L
            for t, c in zip(curve["time_h"], curve["conc"]):
                rows.append({
                    "sheet": st.sheet,
                    "author": st.info.get("Author", ""),
                    "year": st.info.get("Year", ""),
                    "journal": st.info.get("Journal", ""),
                    "strain": st.info.get("Strain", ""),
                    "n_subjects": st.info.get("N", ""),
                    "route": route,
                    "dosing_raw": st.info.get("Dosing", ""),
                    "amount_raw": st.info.get("Amount", ""),
                    "dose_mg": dose_mg,
                    "group": k,
                    "time_h": float(t),
                    "conc_ug_L": float(c) * scale,
                    "conc_unit_raw": curve["unit_conc"].iloc[0],
                })
    return pd.DataFrame(rows)


def healthy_oral(df: pd.DataFrame, dose_mg: float = 80.0) -> pd.DataFrame:
    """건강 성인 경구 특정 용량만 고른다. 주 비교군이다.

    간경변·영아·임신 등 특수군은 생리 파라미터가 달라 같은 모델로
    비교할 수 없으므로 제외한다.
    """
    strain = df["strain"].astype(str).str.lower()
    excluded = (
        strain.str.contains("cirrho|infant|pregna|obes|hemangiom", regex=True)
    )
    return df[
        (df["route"] == "PO")
        & (df["dose_mg"] == dose_mg)
        & (~excluded)
        & (df["time_h"] >= 0)
        & (df["conc_ug_L"] > 0)
    ].copy()
