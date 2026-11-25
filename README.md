# PK-ADMET-Predictor

SMILES 구조에서 ADMET 물성을 예측하고, 그 예측값을 PBPK 모델에 넣어
혈중 농도까지 이어 보는 프로젝트다.

내용은 내 석사 연구(PBPK·ODE 모델링, 마이크로솜 대사안정성, Caco-2
투과도)의 연장선에 있다.

## 왜 이 세 과제인가

TDC ADMET Benchmark Group의 22개 과제 중, **내가 벤치에서 직접 측정해 본
물성** 세 가지를 골랐다. 데이터가 어떻게 만들어지는지 아는 상태에서
예측 모델을 만든다는 점이 이 프로젝트의 출발점이다.

| 과제 | 물성 | 공식 지표 | 내 실험 경험 |
| --- | --- | --- | --- |
| `Clearance_Microsome_AZ` | 간 마이크로솜 고유 청소율 | Spearman | 프로테아좀 과제에서 대사안정성 LC-MS/MS(QQQ) 측정 |
| `PPBR_AZ` | 혈장단백결합률 | MAE | 랫드 혈장단백결합·입체선택적 대사 연구 |
| `Caco2_Wang` | Caco-2 투과도 | MAE | 백색지방 과제에서 Caco-2 투과도 평가·모델링 |

세 과제 모두 TDC가 **scaffold split**을 쓴다. train과 test에 구조적으로 다른
분자가 들어가므로 무작위 분할보다 어렵고, 처음 보는 골격을 예측해야 하는
실제 신약개발 상황에 가깝다.

## 지금 상태 (2026-11-24)

4주차까지 실행 완료. 3주차 관측 세트는 11/24에 다시 만들어 전부 다시 돌렸다.

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
| 4주차 카이랄 재실험 | **실행 완료** → `results/chirality*.csv` |
| 4주차 민감도 분석 | **실행 완료** → `results/sensitivity*` |
| 알려진 약물 sanity check | **실행 완료** → `results/sanity_check.csv` |

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
human PK와 비교했다. 주 세트는 건강 성인 경구 **단회** 80 mg의 총 농도
곡선 7개(연구 6개, 74점)다. 관측 세트는 2026-11-24에 다시 만들었다.

| 주 세트 | Cmax 기하평균 배수오차 | 2배 안쪽 | AUC 배수오차 | 2배 안쪽 |
| --- | --- | --- | --- | --- |
| a priori (ADMET 예측값만) | 4.51 | **0%** | 1.97 | 57% |
| 파라미터 1개 보정 (검증 곡선 6개) | 1.51 | 83% | 1.41 | 100% |

**a priori로는 Cmax를 하나도 맞히지 못했다.** 보정은 Caco-2 → Peff
상관식의 절편 하나만 곡선 1개에 맞추고 나머지 6개로 검증한 것이다.

관측 데이터에 IV가 있어 **흡수와 소실을 분리**해 볼 수 있다. 정맥 곡선
5개에서 부분 AUC는 배수오차 1.22로 맞지만, **말단 반감기가 3~4배 길고
(관측 3.3~5.3 h, 모델 13.9 h) 청소율이 2.5배 낮다**(관측 중앙값 61 L/h,
모델 24.6 L/h). 즉 틀린 것은 흡수만이 아니다. 흡수 쪽 절편만 보정한
모델은 단회 80 mg에서는 맞지만, 약물이 쌓이는 조건(정상상태, 다른 용량)에서
AUC를 약 2배 과대예측한다.

아직 못 맞추는 것은 Tmax다(관측 1.5~3.0 h, a priori 4.0, 보정 후 0.78).
Cmax를 맞추면 Tmax가 어긋난다. 흡수를 1차 속도상수 하나로만 써서
봉우리 모양을 만들 자유도가 없는 구조적 한계다.

누출 검사도 했다. 프로프라놀롤은 `PPBR_AZ` train에만 4건 있었고(R체·S체),
Morgan fingerprint가 `includeChirality=False`라 **모델에게 거울상은 같은
입력**이다. 그래서 PPBR만 해당 분자를 빼고 재학습해 예측했다.

자세한 내용은 [`docs/04_week3_pbpk.md`](docs/04_week3_pbpk.md).

![프로프라놀롤 경구 80mg와 IV](results/pk_propranolol_po.png)

왼쪽 두 패널은 경구 80 mg 단회다. 보정 곡선이 점 위를 지나가지만 로그
축에서는 꼬리가 점들 위로 떠 있다. 오른쪽은 흡수 가정이 들어가지 않는
IV를 용량으로 나눠 겹친 것이다(모델이 선형이라 정규화가 정확하다).
**4시간 이후로 관측 점이 모델 선 아래로 내려가고 14시간에는 5배쯤
벌어진다.** 보정이 고친 것은 흡수뿐이라는 근거다.

## 4주차: 두 가지 확인 실험

3주차가 남긴 두 질문을 각각 실험 하나로 닫았다. 둘 다 기능 추가가 아니라
**이미 내린 결론의 근거를 확인하는** 작업이다.
자세한 내용은 [`docs/05_week4_experiments.md`](docs/05_week4_experiments.md).

**1. 카이랄 fingerprint** (`scripts/run_chirality.py`) — `includeChirality`를
켜면 6개 조합 중 5개가 나빠졌다. 유일한 개선(PPBR_AZ `desc+fp`,
8.657 → 8.578 MAE)도 입체 학습의 증거로 보지 않는다. TDC는 scaffold
split이고 거울상은 골격이 같아, **입체이성질체 짝이 train과 test로 갈라지는
경우가 세 과제 모두 0건**이기 때문이다. 배울 경로 자체가 없다.
→ 결론은 "입체화학이 중요하지 않다"가 아니라 **"이 벤치마크로는 그 질문에
답할 수 없다"**이다.

**2. 민감도 분석** (`scripts/run_sensitivity.py`) — a priori Cmax가 4.5배
틀린 원인이 어느 가정인지 찾았다. **Caco-2 → Peff 상관식의 기울기가
지배한다.** 문헌 범위에서 MPPGL은 AUC를 2.0배 움직이는 데 그치지만
기울기는 25.3배 움직인다(창 무관 기준). 기울기는 `10^(기울기 × logPapp)`의
**지수**로 들어가고 MPPGL은 CLint에 **선형**으로 들어가기 때문이다.
관측 Cmax 중앙값에 닿는 데 필요한 4.0배는 기울기 범위 안에서는 도달
가능하고 MPPGL 범위 안에서는 불가능하다.

![민감도 분석](results/sensitivity.png)

오른쪽 토네이도의 초록 막대(2.0배 / 25.3배)가 본문에서 인용한 창 무관
기준이다. 주황(0-24h 창)이 기울기 쪽에서 130배까지 늘어나는 것은 창을
끊으면 느린 흡수가 과장되기 때문이고, 어느 기준으로 봐도 지배 관계는
같다.

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
src/stereo.py      입체 무시 골격 식별, 입체이성질체 그룹 세기
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
                   train_ddp, run_pbpk, run_chirality, run_sensitivity,
                   run_sanity
notebooks/         01_rdkit_intro — RDKit 기초 + 데이터 탐색
docs/              환경 구성, 막혔던 것들, 주차별 기록
data/ results/     TDC 캐시와 결과 (저장소에는 요약만 올린다)
```

## 규칙

- **test set으로 모델을 고르지 않는다.** TDC가 준 train/valid만 쓰고,
  평가는 `group.evaluate()` / `group.evaluate_many()`로만 한다.
  test 점수를 보고 하이퍼파라미터를 바꾸면 리더보드와 비교할 수 없는 숫자가 된다.
- seed 5개의 평균±표준편차로 보고한다. 한 번 잘 나온 숫자는 쓰지 않는다.
- 미공개 실험 데이터는 넣지 않는다. 공개 데이터와 이미 발표된 논문만 쓴다.
- 남이 읽고 한 줄씩 설명할 수 있는 코드로 쓴다. 이해되지 않는 줄은
  남겨 두지 않는다.

## 출처

- [TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/)
