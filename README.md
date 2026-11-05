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

## 지금 상태 (2026-11-05)

1주차 코드까지 준비된 상태다.

| 항목 | 상태 |
| --- | --- |
| 저장소 구조, 환경 파일, 문서 | 준비 완료 |
| RDKit 특징 추출 (`src/features.py`) | 작성·동작 확인 |
| 베이스라인 (RandomForest, LightGBM) | 작성·동작 확인 |
| TDC 데이터 로드 (`src/data.py`) | 작성, **아직 실행 미확인** |
| RDKit 입문 노트북 | 작성, **아직 실행 미확인** |
| 2주차 PyTorch 모델 | 미작성 |
| 3주차 PBPK 연계 (`pbpk/`) | 미작성 |

`src/features.py`와 베이스라인은 `scripts/smoke_test.py`로 실제 분자에
돌려서 확인했다. TDC 데이터 로드는 데이터 서버 접근이 막힌 환경에서
작성했기 때문에 실행으로 확인하지 못했다. **WSL에서 처음 돌릴 때 여기서
막힐 수 있다.**

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
