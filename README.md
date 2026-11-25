# PK-ADMET-Predictor

**화학 구조 기반의 ADMET 예측값만으로 사람의 혈중 약물 농도-시간 곡선(PK profile)을 예측할 수 있는지 검증하고, 예측 오차가 발생하는 단계를 분리해 진단한 프로젝트임.**

약물의 SMILES 구조로부터 세 가지 ADMET 물성(`Caco2_Wang`, `PPBR_AZ`, `Clearance_Microsome_AZ`)을 머신러닝으로 예측한 뒤, 이를 4-compartment **minimal PBPK (mPBPK)** 모델에 입력해 Propranolol의 실제 human clinical PK 데이터와 비교했다.

---

## 전체 분석 흐름 (Pipeline)

```text
[1. Molecular Input]     [2. ADMET Prediction]        [3. mPBPK Simulation]          [4. Clinical PK Validation]
  SMILES string    ──▶    Caco-2 permeability   ──▶   Gut → Liver → Central ↔ Peri   ──▶  PO (80 mg) · IV (5 curves)
  Descriptors / Graph     PPBR / Hepatic CLint        4-compartment ODE                   Fold error · Sensitivity · Chirality
```

1. **Featurization (`src/features.py`, `src/graph.py`, `src/stereo.py`)**: SMILES 문자열을 27개 physicochemical **descriptor**, 2048-bit **Morgan fingerprint**, 또는 atom-bond **molecular graph**로 변환한다.
2. **ADMET Model Training (`src/baseline.py`, `src/models.py`, `src/train.py`)**: TDC benchmark 데이터셋으로 tree-based model과 neural network를 동일한 split 기준으로 비교한다.
3. **mPBPK Simulation (`pbpk/predict_admet.py`, `pbpk/model.py`)**: 예측된 ADMET 값을 70 kg 성인 기준 physiological constants와 결합해 시간별 plasma concentration을 계산한다.
4. **Clinical Validation & Verification (`pbpk/observed.py`, `scripts/run_pbpk.py`, `scripts/run_chirality.py`, `scripts/run_sensitivity.py`, `scripts/run_sanity.py`)**: 문헌에서 수집한 healthy adult clinical PK 데이터와 비교하고, chiral fingerprint 재실험 및 parameter sensitivity analysis로 결론의 근거를 검증한다.

---

## 핵심 결과 요약

Propranolol **single-dose PO 80 mg** 투여 시(healthy adults, total concentration 7 curves from 6 studies, 73 data points) 실제 관측값과 시뮬레이션 결과를 비교했다.

* **a priori prediction**: clinical PK 데이터를 전혀 보지 않고 ADMET 예측값만으로 계산한 결과임.
* **1-parameter calibration**: `Caco-2 → Peff` 경험식의 intercept(`-1.9`) 하나만 1개 curve에 맞춰 보정하고, 나머지 6개 holdout curves로 검증한 결과임.
* **GMFE (Geometric Mean Fold Error)**: 예측값과 관측값의 비율 차이를 나타내는 지표로, `1.0`에 가까울수록 정확하며 PK에서는 보통 2-fold 이내(0.5–2.0)를 허용 범위로 본다.

| Mode (PO 80 mg single-dose) | Cmax GMFE | Within 2-fold (Cmax) | AUC GMFE | Within 2-fold (AUC) |
| --- | --- | --- | --- | --- |
| **a priori** (ADMET 예측값만 사용) | 4.51 | 0.0% | 1.97 | 57% |
| **1-parameter calibrated** (6 holdout curves) | 1.51 | 83% | 1.41 | 100% |

### 주요 진단 결과

1. **Absorption과 Elimination의 분리 진단 (IV Cross-Check)**
   IV 투여(5 curves)는 gut absorption을 거치지 않고 central compartment로 직접 들어간다. IV 데이터에서 부분 AUC는 **1.22-fold**로 맞지만, **terminal half-life는 약 3.8배 길고**(관측 3.3–5.3 h vs. 모델 13.9 h, 곡선별 2.6–4.2배), **clearance는 2.5배 낮게**(관측 중앙값 61 L/h vs. 모델 24.6 L/h) 예측되었다. 즉 틀린 것은 absorption만이 아니며, absorption intercept만 보정한 모델은 single-dose 80 mg에서는 맞지만 steady-state나 다회 투여처럼 약물이 축적되는 조건에서는 AUC를 약 2배 과대예측한다.
2. **겉보기 Bioavailability (`F`) 일치 이면의 Error Cancellation**
   Oral bioavailability는 `F = Fa × (1 − E_h)`로 계산된다. a priori 예측에서 `F = 0.185`로 문헌값(`~0.26`)과 비슷해 보였으나, 실제로는 `Fa`(예측 0.26 vs. 거의 완전 흡수)와 hepatic extraction ratio `E_h`(예측 0.27 vs. 문헌 `~0.6–0.75`)를 모두 과소평가해 두 오차가 반대 방향으로 상쇄된 결과였음. 숫자는 맞지만 기전은 틀린 경우다.
3. **Achiral Features의 구조적 한계와 Data Leakage 방지**
   Propranolol은 R/S **enantiomer**에 따라 실제 protein binding과 metabolism 속도가 다르다. 하지만 기본 입력(`Morgan fingerprint` with `includeChirality=False`, 27 achiral descriptors)은 **chirality**를 구분하지 못해 R체·S체·racemate를 동일한 입력으로 인식한다. 또한 `PPBR_AZ` train set에 Propranolol 4건(R·S체 각 2건)이 포함되어 있음을 확인해, mPBPK 입력 예측 시 해당 분자를 제외하고 재학습했다.

### PO vs. IV 비교 그래프

![Propranolol PO 80 mg single-dose와 dose-normalized IV](results/pk_propranolol_po.png)

* **왼쪽 두 패널 (PO 80 mg single-dose)**: 보정 곡선이 관측 점 위를 지나가지만, log scale에서는 말단 꼬리(tail)가 관측 점들 위로 떠 있음.
* **오른쪽 패널 (Dose-normalized IV, 5 curves)**: Absorption 가정이 들어가지 않는 IV 곡선으로, 4시간 이후부터 관측 점이 모델 선 아래로 내려가고 14시간에는 약 5배 벌어진다. 이는 보정이 교정한 대상이 absorption뿐임을 뒷받침한다.

자세한 시뮬레이션 기록은 [docs/04_week3_pbpk.md](docs/04_week3_pbpk.md)에서 확인할 수 있다.

---

## 1단계: ADMET 예측 모델

[TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/)에서 mPBPK 입력에 직접 필요한 3가지 task를 선정했다. 모든 데이터는 구조적으로 다른 분자를 예측하는 실제 상황에 가깝도록 **scaffold split**을 적용해 평가했다.

| Task | Property (Unit) | Official Metric | Role in mPBPK |
| --- | --- | --- | --- |
| `Caco2_Wang` | Intestinal permeability (`log Papp`, cm/s) | MAE ↓ | `Peff`, `ka`, `Fa` 추정 |
| `PPBR_AZ` | Plasma protein binding rate (`%`) | MAE ↓ | Plasma unbound fraction (`fu`) 계산 |
| `Clearance_Microsome_AZ` | Hepatic microsome intrinsic clearance (`uL/min/mg`) | Spearman ↑ | Hepatic `CLint` 및 `E_h` 계산 |

### 모델 비교 결과 (5 seeds, mean ± std)

Tree-based baseline (`RandomForest`, `LightGBM`)과 neural network (`MLP`, `GIN`, `GIN+desc`)를 동일한 split 기준으로 비교했다.

| Task | Metric | RF (`desc+fp`) | LightGBM (`desc+fp`) | MLP | GIN (graph only) | GIN+desc (graph+desc) |
| --- | --- | --- | --- | --- | --- | --- |
| `Caco2_Wang` | MAE ↓ | **0.279 ± 0.003** | 0.281 ± 0.002 | 0.288 ± 0.012 | 0.434 ± 0.033 | 0.343 ± 0.026 |
| `PPBR_AZ` | MAE ↓ | 8.657 ± 0.020 | 8.747 ± 0.051 | 8.283 ± 0.245 | 8.240 ± 0.527 | **7.982 ± 0.350** |
| `Clearance_Microsome_AZ` | Spearman ↑ | 0.506 ± 0.002 | 0.486 ± 0.008 | 0.508 ± 0.025 | 0.524 ± 0.040 | **0.562 ± 0.027** |

* **Task 특성에 따른 모델 적합성**: 분자 전체의 physicochemical properties가 지배적인 `Caco2_Wang`에서는 descriptor를 활용하는 tree model이 가장 뛰어났다. 반면 특정 metabolic soft spot이 중요한 `Clearance_Microsome_AZ`에서는 원자 연결 구조를 직접 보는 `GIN+desc`가 가장 높은 성능을 보였다.
* **Descriptor 결합 효과**: Graph만 입력받는 `GIN`에 27개 descriptor를 결합(`GIN+desc`)하면 세 task 모두 평균값이 향상되었다(단 `PPBR_AZ`는 seed 간 std가 커 구간이 겹친다).
* **엄격한 평가 및 Reproducibility**: **test set**은 최종 평가에만 사용하며, **feature/target normalization** 통계량과 **early stopping** 기준은 `train`/`valid`에서만 계산한다. 또한 PyTorch GPU 연산(`index_add_`)의 비결정성을 해결하기 위해 **deterministic kernel**을 강제해 재현성을 확보했다([docs/03_week2_notes.md](docs/03_week2_notes.md)).

---

## 2단계: mPBPK (Minimal PBPK) 모델 구조

세 가지 ADMET 예측값만으로 뒷받침할 수 있는 최소한의 생리학 구조를 설계하기 위해, 전신 장기 구획을 모두 나누는 대신 **4-compartment mPBPK** 구조를 사용했다. 경구 흡수된 약물이 central compartment가 아니라 **portal circulation**을 통해 **Liver**로 먼저 유입되므로 **first-pass metabolism**이 구조적으로 자연스럽게 반영된다.

```text
PO dose ──ka──▶ Gut Lumen ──ka·Fa──▶ Liver ──Qh──▶ Central (Blood) ◀──Qp──▶ Peripheral
                                       │                  ▲
                                   CLint·fu               │
                                       ▼               IV dose
                                   Elimination
```

* **Physiological Constants (70 kg adult)**: `liver_weight = 1800 g`, `V_liver = 1.8 L`, `Q_h = 90.0 L/h`, `MPPGL = 45.0 mg/g`, `V_central = 5.0 L`, `V_peripheral = 60.0 L`, `Q_p = 60.0 L/h`, `gut_radius = 1.75 cm`, `blood/plasma ratio = 1.0`.
* **Well-stirred Liver Model**: `E_h = (fu · CLint) / (Q_h + fu · CLint)`

| Input Source | mPBPK Parameter | Conversion | Assumption Level |
| --- | --- | --- | --- |
| `PPBR_AZ` | `fu` | `fu = 1 − PPB/100` | None (definition) |
| `Clearance_Microsome_AZ` | `CLint` (L/h) | `CLint (uL/min/mg) × MPPGL (45 mg/g) × liver_weight (1800 g)` | Low (standard IVIVE) |
| `Caco2_Wang` | `ka`, `Fa` | `log10 Peff = 0.6 · log10 Papp − 1.9` → `ka = Peff · (2/R)` | **High (empirical correlation)** |
| RDKit `MolLogP` | `Kp` (partition coeff.) | Tissue composition model (`MolLogP`, `fu`) | Calculated from structure |

---

## 3단계: 검증 실험 (Chirality · Sensitivity · Sanity Check)

결론의 근거를 확인하기 위해 추가 검증 실험을 수행했다([docs/05_week4_experiments.md](docs/05_week4_experiments.md)).

### 1. Chiral Fingerprint 실험 (`scripts/run_chirality.py`)
* `includeChirality=True` 옵션을 켜면 6개 조합 중 5개에서 성능이 하락했다(`results/chirality_comparison.csv`).
* 유일하게 개선된 `PPBR_AZ desc+fp` (`8.657 → 8.578 MAE`) 역시 입체 학습의 증거로 볼 수 없다. TDC가 **scaffold split**을 사용하고 enantiomer끼리는 scaffold가 같아, **stereoisomer pair가 train과 test로 갈라지는 경우가 세 task 모두 0건**이기 때문임.
* 결론은 "chirality가 중요하지 않다"가 아니라 **"현재 benchmark 분할 구조로는 해당 질문에 답할 수 없다"**임.

### 2. Parameter Sensitivity Analysis (`scripts/run_sensitivity.py`)
* `a priori` Cmax가 4.5배 틀린 주원인을 정량적으로 확인하기 위해 `MPPGL (30–60 mg/g)`과 `Caco-2 → Peff slope (0.40–0.80)`의 민감도를 비교했다(`results/sensitivity_summary.csv`).
* 문헌 범위 내에서 `MPPGL`은 `AUC_inf`를 **2.0-fold** 움직이는 데 그치지만, `Caco-2 → Peff slope`는 **25.3-fold** (`0–24h` 창 기준 **130.3-fold**, Cmax **563.3-fold**) 움직인다.
* `slope`는 `10^(slope × logPapp)`의 **exponent**로 작용하는 반면 `MPPGL`은 `CLint`에 **linear**하게 들어가기 때문이며, 관측 Cmax 중앙값 도달에 필요한 4.0-fold 변화는 `slope` 범위에서만 도달 가능하다.

![파라미터 민감도 분석 (MPPGL vs. Caco-2 → Peff slope)](results/sensitivity.png)

### 3. Known Drugs Sanity Check (`scripts/run_sanity.py`)
* 8개 알려진 약물(Atenolol, Metoprolol, Propranolol, Theophylline, Caffeine, Diazepam, Warfarin, Verapamil)에 대해 세 ADMET 모델의 예측 합당성을 점검했다(`results/sanity_check.csv`).

---

## 한계 및 향후 과제

* **단일 약물·단일 용량 검증**: 주 검증 세트가 Propranolol PO 80 mg single-dose 조건이므로, 보정된 `Caco-2 → Peff` intercept의 다른 약물/용량에 대한 external validity는 검증되지 않았다.
* **`Tmax` 예측 오차**: 흡수를 단일 1차 속도상수 `ka` 하나로만 모델링해 관측 `Tmax` (1.5–3.0 h) 대비 시뮬레이션 `Tmax` (a priori 4.0 h, calibrated 0.78 h)가 어긋난다. **Transit compartment**를 추가하면 해결할 수 있는 구조적 한계임.
* **Clearance 과소평가 및 다회 투여 한계**: IV 검증에서 모델의 clearance(24.6 L/h)가 관측 중앙값(61 L/h)보다 약 2.5배 낮게 나타나, steady-state 등 약물 축적 조건에서는 추가 보정 없이는 AUC가 과대예측된다.
* **Neural Network 분산 및 환경 제약**: Neural network는 tree model 대비 seed 간 std가 크며(`0.012–0.527` vs `0.002–0.051`), multi-GPU DDP(`scripts/train_ddp.py`)는 단일 GPU 환경 제약으로 미검증 상태임. 플랫폼 간 차이를 막기 위해 `lightgbm=4.7.0`, `scikit-learn=1.9.1`로 고정했다.

**향후 개선 방향**

1. Gut **transit compartment** 도입을 통한 `Tmax` 예측 개선
2. Stereoisomer pair가 train/test에 분리되지 않는 scaffold split 한계를 고려한 **chiral benchmark** 재설계
3. 보정된 상관식의 타 약물군 적용 및 특수군(cirrhosis, infant, pregnancy) physiological parameter 확장

---

## 재현 방법 및 저장소 구조

환경은 WSL2 Ubuntu-24.04를 기준으로 한다(`cuda available : True`, RTX 3070). Windows 네이티브는 `conda env create`가 `tiledbsoma`에서 실패하고 `Clearance_Microsome_AZ` task에서 MKL 관련 crash가 있어 권장하지 않는다. 플랫폼별 이슈와 우회 방법은 [docs/00_setup_wsl2_cuda.md](docs/00_setup_wsl2_cuda.md)와 [docs/02_week1_issues.md](docs/02_week1_issues.md)에 정리되어 있다.

```bash
conda env create -f environment.yml && conda activate pk-admet
pip install torch --index-url https://download.pytorch.org/whl/cu124

python scripts/check_env.py        # 패키지 및 CUDA 확인
python scripts/smoke_test.py       # 오프라인 환경에서 RDKit 및 모델 점검
python scripts/run_baseline.py     # Tree baseline 학습 (3 tasks × 2 models × 3 features × 5 seeds)
python scripts/run_models.py       # Neural network 학습 (MLP, GIN, GIN+desc)
python scripts/run_pbpk.py         # ADMET prediction → mPBPK simulation → clinical PK 비교
python scripts/run_chirality.py    # Chiral fingerprint 재실험
python scripts/run_sensitivity.py  # Parameter sensitivity analysis
python scripts/run_sanity.py       # Known drugs sanity check
```

### Repository Structure

```text
src/
  config.py          Target tasks, metrics, seeds
  data.py            TDC data loading & train/valid/test split
  features.py        RDKit descriptors (27) + Morgan fingerprint (2048 bits)
  stereo.py          Achiral skeleton identification & stereoisomer group counting
  baseline.py        RandomForest / LightGBM training & TDC evaluation
  graph.py           SMILES → atom/bond graph conversion
  torch_data.py      Dataset, graph batch collate_fn, train-only normalization
  models.py          MLP & GIN implementation (without torch_geometric)
  train.py           Training loop & valid-based early stopping
  utils.py           Seed fixing, deterministic execution, result saving
pbpk/
  model.py           4-compartment mPBPK ODE & ADMET parameter conversion
  predict_admet.py   Target ADMET prediction (retraining excluding leaked molecules)
  observed.py        Published human PK Excel parsing (explicit curve manifest)
scripts/             check_env, smoke_test, run_baseline, run_models, train_ddp,
                     run_pbpk, run_chirality, run_sensitivity, run_sanity
notebooks/           01_rdkit_intro.ipynb
docs/                Setup guides & weekly experiment notes (00~05)
data/ results/       Observed PK data, benchmark tables & simulation plots
```

> **참고 (`pbpk/observed.py`)**: 원본 엑셀(`data/propranolol_human.xlsx`)은 헤더를 찾아 자동으로 긁지 않는다. 처음에는 `Time (` 헤더 아래를 스캔했는데 이 워크북에서 네 가지 방식으로 조용히 틀렸다 — 한 열에 세로로 쌓인 투여군을 한 곡선으로 병합(남·녀, 단회·정상상태, 대조·비만), 이성질체별 열과 `SUM` 열이 나란한 시트에서 첫 열만 읽음, `DATA`/`RAW DATA` 블록의 중복 제거가 디지타이즈 시간(23.97)과 명목 시간(24)이 다를 때 실패, 용량 칸이 `GROUP1`처럼 문자열인 시트는 통째로 누락. 그래서 곡선마다 **시트·헤더 행·농도 열을 엑셀 좌표 그대로** 적은 `CURVES` 목록과 제외 사유를 적은 `EXCLUDED` 목록을 두고, 시간 단위(`h`/`min`)·투여 경로·용량·단회/정상상태·대상군도 함께 선언한다. 읽은 점 개수가 선언과 다르면 중단한다.

## 출처

* [TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/)
