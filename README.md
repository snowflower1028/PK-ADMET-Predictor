# PK-ADMET-Predictor

SMILES 구조에서 ADMET 물성을 예측하고, 그 예측값을 PBPK 모델에 넣어
혈중 농도까지 이어 보는 프로젝트다.

아론티어 AI Researcher Scientist(전문연구요원) 지원용 포트폴리오로 만들지만,
내용 자체는 내 석사 연구(PBPK·ODE 모델링, 마이크로솜 대사안정성, Caco-2
투과도)의 연장선에 있다.

## 왜 이 세 과제인가

TDC ADMET Benchmark Group의 22개 과제 중, **내가 벤치에서 직접 측정해 본
물성** 세 가지를 골랐다. 데이터가 어떻게 만들어지는지 아는 상태에서
예측 모델을 만든다는 점이 이 포트폴리오의 차별점이다.

| 과제 | 물성 | 공식 지표 | 내 실험 경험 |
| --- | --- | --- | --- |
| `Clearance_Microsome_AZ` | 간 마이크로솜 고유 청소율 | Spearman | 프로테아좀 과제에서 대사안정성 LC-MS/MS(QQQ) 측정 |
| `PPBR_AZ` | 혈장단백결합률 | MAE | 랫드 혈장단백결합·입체선택적 대사 연구 |
| `Caco2_Wang` | Caco-2 투과도 | MAE | 백색지방 과제에서 Caco-2 투과도 평가·모델링 |

세 과제 모두 TDC가 **scaffold split**을 쓴다. train과 test에 구조적으로 다른
분자가 들어가므로 무작위 분할보다 어렵고, 처음 보는 골격을 예측해야 하는
실제 신약개발 상황에 가깝다.

## 지금 상태 (2026-11-06)

1주차 완료. 세 과제 베이스라인 숫자가 나왔다.

| 항목 | 상태 |
| --- | --- |
| 저장소 구조, 환경 파일, 문서 | 준비 완료 |
| RDKit 특징 추출 (`src/features.py`) | 작성·동작 확인 |
| 베이스라인 (RandomForest, LightGBM) | **실행 완료** → `results/baseline.csv` |
| TDC 데이터 로드 (`src/data.py`) | **실행 확인** |
| RDKit 입문 노트북 | **전체 실행 확인** (에러 셀 0개) |
| 2주차 PyTorch 모델 | 작성 중 |
| 3주차 PBPK 연계 (`pbpk/`) | 미작성 |

환경은 **WSL2 Ubuntu-24.04(권장)와 Windows 네이티브 두 가지**를 다
만들어 돌렸다. 둘 다 `cuda available : True`. 막혔던 지점과 해결은
[`docs/02_week1_issues.md`](docs/02_week1_issues.md)에 남겼다.

### 1주차 베이스라인 결과

세 과제 모두 RandomForest에 descriptor+fingerprint를 넣은 것이 가장 좋았다.
2주차 모델은 이 숫자를 기준으로 판단한다.

| 과제 | 지표 | 최고 | 2위 |
| --- | --- | --- | --- |
| `Caco2_Wang` | MAE ↓ | **0.279 ± 0.003** | lgbm desc+fp 0.281 ± 0.002 |
| `PPBR_AZ` | MAE ↓ | **8.657 ± 0.020** | lgbm desc+fp 8.747 ± 0.051 |
| `Clearance_Microsome_AZ` | Spearman ↑ | **0.506 ± 0.002** | lgbm fp 0.492 ± 0.014 |

세 환경(클라우드 Linux, WSL2, Windows)에서 독립적으로 돌려 위 세 숫자가
소수점 3자리까지 같았다. 전체 18조합 중 Windows만 LightGBM 2개가
표준편차 안쪽에서 달랐다(`docs/02_week1_issues.md` 7번).

특징 하나. fingerprint만 쓴 쪽이 descriptor만 쓴 쪽보다 좋은 과제는
Clearance 뿐이다(0.445 vs 0.425). 투과도·단백결합은 분자 전체의
물리화학적 성질이 지배하는데, 대사 청소율은 특정 대사 취약부위가
결정하기 때문으로 읽는다. 2주차 GNN이 구조를 직접 보는 모델이므로
Clearance에서 가장 많이 오를 것이라는 가설을 여기서 세웠다.

## 빠른 시작

환경 구성은 [`docs/00_setup_wsl2_cuda.md`](docs/00_setup_wsl2_cuda.md)를 따른다.

```bash
conda env create -f environment.yml && conda activate pk-admet
pip install torch --index-url https://download.pytorch.org/whl/cu124

python scripts/check_env.py     # 패키지·CUDA 확인
python scripts/smoke_test.py    # 네트워크 없이 RDKit·모델 확인
python scripts/run_baseline.py --tasks Caco2_Wang --models rf --features desc --seeds 1
python scripts/run_baseline.py  # 전체: 3과제 x 2모델 x 3특징 x 5seed
```

## 저장소 구조

```
src/config.py      대상 과제, 지표, seed 정의
src/data.py        TDC 로드와 train/valid/test 분할
src/features.py    RDKit descriptor(27개) + Morgan fingerprint(2048비트)
src/baseline.py    RandomForest / LightGBM, TDC 공식 평가 호출
src/utils.py       난수 고정, 결과 저장
scripts/           check_env, smoke_test, run_baseline
notebooks/         01_rdkit_intro — RDKit 기초 + 데이터 탐색
docs/              환경 구성, 주차별 체크리스트
pbpk/              3주차 PBPK 연계 (예정)
data/ results/     TDC 캐시와 결과 (저장소에는 요약만 올린다)
```

## 일정

| 주차 | 내용 | 산출물 |
| --- | --- | --- |
| 1주차 11/5~11/11 | 환경, RDKit 학습, 베이스라인 | `results/baseline.csv` |
| 2주차 11/12~11/18 | PyTorch Dataset/DataLoader, MLP·GNN, DDP | `results/models.csv` |
| 3주차 11/19~11/23 | 예측값을 PBPK에 입력, 관측 PK와 비교 | `results/pk_comparison.csv` |
| 11/24~11/25 | README 정리, 이력서·자기소개서 | 지원 서류 |
| 11/26 | 지원 (마감 11/27) | — |

## 공고 요건과의 대응

| 공고 필수요건 | 이 프로젝트에서 증명하는 것 |
| --- | --- |
| Linux·Python 능숙 | WSL2 환경 구성, 패키지 구조화 |
| 딥러닝 프레임워크 능숙 | 2주차 PyTorch MLP·GNN 학습 |
| 최신 모델 분석·적용·개선 | 공개 GNN 구현을 가져와 이 데이터에 맞춤 |
| Data loader 구현 | 분자 그래프용 `Dataset`·`collate_fn` 직접 작성 |
| multi-GPU 활용 | DDP 학습 코드 (검증은 Kaggle T4 x2, 자세한 사정은 계획 문서 참고) |
| ADMET 모델링 경험(우대) | 세 과제 자체가 ADMET 예측 |
| RDKit·descriptor(우대) | `src/features.py` |

## 규칙

- **test set으로 모델을 고르지 않는다.** TDC가 준 train/valid만 쓰고,
  평가는 `group.evaluate()` / `group.evaluate_many()`로만 한다.
  test 점수를 보고 하이퍼파라미터를 바꾸면 리더보드와 비교할 수 없는 숫자가 된다.
- seed 5개의 평균±표준편차로 보고한다. 한 번 잘 나온 숫자는 쓰지 않는다.
- 연구실 미공개 데이터·코드는 넣지 않는다. 공개 데이터와 이미 발표된 논문만 쓴다.
- 면접에서 코드를 설명해야 한다. 이해되지 않는 줄은 남겨 두지 않는다.

## 출처

- [TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/)
- [아론티어 채용 공고 (사람인)](https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=55427749)
- [지원 준비 계획 문서](https://claude.ai/artifact/P5FXPZJBfHG6UP1UKVHcwc)
