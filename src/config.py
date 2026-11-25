"""프로젝트 전역 설정: 대상 과제, 경로, 재현성 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# TDC ADMET Benchmark Group 중 직접 실험해 본 세 물성.
# metric은 TDC가 각 과제에 지정한 공식 지표이며, 우리가 고른 것이 아니다.
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

# Morgan fingerprint에 입체화학을 넣을지. **기본값은 False로 둔다.**
#
# results/baseline.csv와 results/models.csv는 이 값이 False인 상태에서 나온
# 공식 split 결과이고, 리더보드 비교 표의 기준선이다. 기본값을 True로
# 바꾸면 그 숫자들이 조용히 달라진다. 그래서 전역 기본은 건드리지 않고,
# 카이랄 재실험은 scripts/run_chirality.py에서 이 값을 덮어써서 돌린다.
MORGAN_CHIRALITY = False
