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
| 2주차 PyTorch 모델 (MLP, GIN, GIN+desc) | **실행 완료** → `results/models.csv` |
| 2주차 DDP (`scripts/train_ddp.py`) | 작성 완료, **GPU 1장이라 미검증** |
| 3주차 PBPK 연계 (`pbpk/`) | **실행 완료** → `results/pk_comparison.csv` |

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

### 2주차 모델 결과

위 가설대로 됐다. 네 모델 모두 **입력이 과제의 물리에 맞을 때만 이긴다.**

| 과제 | 지표 | 1주차 rf desc+fp | MLP | GIN | GIN+desc |
| --- | --- | --- | --- | --- | --- |
| `Caco2_Wang` | MAE ↓ | **0.279 ± 0.003** | 0.288 ± 0.012 | 0.434 ± 0.033 | 0.343 ± 0.026 |
| `PPBR_AZ` | MAE ↓ | 8.657 ± 0.020 | 8.283 ± 0.245 | 8.240 ± 0.527 | **7.982 ± 0.350** |
| `Clearance_Microsome_AZ` | Spearman ↑ | 0.506 ± 0.002 | 0.508 ± 0.025 | 0.524 ± 0.040 | **0.562 ± 0.027** |

- **PPBR**: `gin+desc`가 7.982로 RandomForest(8.657)보다 0.675 낫다.
  표준편차 0.350 대비 약 1.9배.
- **Clearance**: `gin+desc`가 0.562로 RandomForest(0.506)보다 0.056 높다.
  표준편차 0.027 대비 약 2배.
- **Caco2**: 여전히 RandomForest가 가장 좋다. descriptor를 넣어 0.434에서
  0.343으로 크게 좁혔지만 못 넘었다.

**descriptor를 넣으면 세 과제 모두 GIN이 좋아진다**(0.434→0.343,
8.240→7.982, 0.524→0.562). 예외가 없고, **개선 폭이 가장 큰 것이
Caco2**다. 그 과제가 물리화학적 성질을 가장 많이 필요로 하기 때문이다.
1주차에서 fingerprint가 descriptor를 이긴 과제가 Clearance 뿐이었던 것과
같은 이야기의 뒷면이다.

같이 적어 둘 것 두 가지.

**(1) 표준편차가 트리 모델보다 한 자릿수 크다**(0.026~0.527 대
0.002~0.020). GNN은 seed에 민감하고 PPBR의 한 seed는 epoch 8에서 멈췄다.
seed 5개의 표준편차 대비 2배는 일관된 방향이라고는 말할 수 있어도
엄밀한 유의성 주장은 아니다.

**(2) seed 고정만으로는 재현되지 않았다.** `index_add_`의 병렬 덧셈
순서 때문에 같은 seed로도 GIN 결과가 달라졌다. 결정적 커널을 강제해서
고쳤고, 전체 격자를 다시 돌린 뒤 두 번 실행해 평균·표준편차·멈춘
epoch이 모두 같은 것을 확인했다. 위 표는 그 결정적 실행의 숫자다.
자세한 추적 과정은 [`docs/03_week2_notes.md`](docs/03_week2_notes.md).

하이퍼파라미터는 test를 보고 고르지 않았다. 데이터 크기만 보고 미리
정했고(batch 64, AdamW lr 1e-3, Huber loss, patience 40), epoch 선택은
전부 valid로 했다. `results/models.csv`에 seed별 valid 점수와 멈춘
epoch을 남겨 그 근거를 볼 수 있게 했다.

### 3주차: 예측값을 PBPK에 넣어 혈중 농도까지

세 ADMET 예측값을 간·장 포함 축소 PBPK에 넣고, 출판된 프로프라놀롤
human PK(건강 성인 경구 80 mg, 12개 연구 288점)와 비교했다.

| | Cmax 기하평균 배수오차 | 2배 안쪽 | AUC 배수오차 | 2배 안쪽 |
| --- | --- | --- | --- | --- |
| a priori (ADMET 예측값만) | 4.97 | **0%** | 2.26 | 42% |
| 파라미터 1개 보정 (검증 11개 연구) | 1.48 | 82% | 1.51 | 91% |

**a priori로는 Cmax를 하나도 맞히지 못했다.** 보정은 Caco-2 → Peff
상관식의 절편 하나만 연구 1개에 맞추고 나머지 11개로 검증한 것이다.

관측 데이터에 IV가 있어 **흡수와 소실을 분리**할 수 있었다. IV는
a priori 그대로 배수오차 1.21로 들어온다. 즉 소실 쪽(fu, CLint)은
대체로 맞고 **틀린 것은 흡수 쪽**이다. 보정할 파라미터로 Caco-2 절편을
고른 것은 이 진단의 결과다.

아직 못 맞추는 것은 Tmax다(관측 1.5~3.0 h, a priori 4.0, 보정 후 0.75).
Cmax를 맞추면 Tmax가 어긋난다. 흡수를 1차 속도상수 하나로만 써서
봉우리 모양을 만들 자유도가 없는 구조적 한계다.

누출 검사도 했다. 프로프라놀롤은 `PPBR_AZ` train에만 4건 있었고(R체·S체),
Morgan fingerprint가 `includeChirality=False`라 **모델에게 거울상은 같은
입력**이다. 그래서 PPBR만 해당 분자를 빼고 재학습해 예측했다.

자세한 내용은 [`docs/04_week3_pbpk.md`](docs/04_week3_pbpk.md).

![프로프라놀롤 경구 80mg](results/pk_propranolol_po.png)

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
src/graph.py       SMILES -> 원자/결합 그래프 (GNN 입력)
src/torch_data.py  Dataset 2종 + collate_fn 직접 작성, 타깃 정규화
src/models.py      MLP, GIN (torch_geometric 없이 직접 구현)
src/train.py       학습 루프, valid 기반 early stopping
src/utils.py       난수 고정, 결정적 실행, 결과 저장
pbpk/model.py      간·장 축소 PBPK (ODE 4개), ADMET -> 파라미터 변환
pbpk/predict_admet.py  대상 분자 물성 예측 (누출 분자는 제외 재학습)
pbpk/observed.py   출판된 human PK 엑셀 파싱
scripts/           check_env, smoke_test, run_baseline, run_models,
                   train_ddp, run_pbpk
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
