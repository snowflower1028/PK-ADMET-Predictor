"""프로젝트 전역 설정: 대상 과제, 경로, 재현성 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# TDC ADMET Benchmark Group 중 내가 직접 실험해 본 세 물성.
# metric은 TDC가 과제마다 지정한 공식 지표다. 내가 고른 것이 아니다.
TASKS = {
    "Caco2_Wang": {
        "label": "Caco-2 투과도 (log Papp, cm/s)",
        "metric": "mae",
        "higher_is_better": False,
        "wet_lab_link": "백색지방 과제에서 Caco-2 투과도 평가·모델링 수행",
    },
    "PPBR_AZ": {
        "label": "혈장단백결합률 (%)",
        "metric": "mae",
        "higher_is_better": False,
        "wet_lab_link": "랫드 혈장단백결합·입체선택적 대사 연구 수행",
    },
    "Clearance_Microsome_AZ": {
        "label": "간 마이크로솜 고유 청소율 (uL/min/mg)",
        "metric": "spearman",
        "higher_is_better": True,
        "wet_lab_link": "프로테아좀 과제에서 간 마이크로솜 대사안정성 LC-MS/MS(QQQ) 측정",
    },
}

# TDC 리더보드 관행: 서로 다른 seed 5회의 평균±표준편차를 보고한다.
SEEDS = [1, 2, 3, 4, 5]

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
