# 2주차 기록: 재현되지 않는 GNN과 descriptor 추가 (2026-11-06)

## 1. 같은 seed인데 결과가 달라졌다

`gin`을 두 번 돌렸더니 숫자가 달랐다. seed도, 코드도, 환경도 같았다.

| 조합 | 1차 | 2차 |
| --- | --- | --- |
| Clearance gin | 0.567 ± 0.026 | 0.530 ± 0.041 |
| PPBR gin | 8.157 ± 0.422 | 8.236 ± 0.346 |
| Caco2 gin | 0.422 ± 0.068 | 0.431 ± 0.061 |

**MLP 세 줄은 두 실행에서 완전히 같았다.** 이 대비가 범인을 가리켰다.
MLP에 없고 GIN에만 있는 것은 메시지 집계, 즉 `index_add_`다.

### 원인: 난수가 아니라 덧셈 순서

`set_seed()`로 난수는 이미 고정돼 있었다. 문제는 다른 곳이었다.

`index_add_`는 흩어진 값을 같은 자리에 더한다. GPU에서는 여러 스레드가
atomic 연산으로 동시에 더하므로 **더해지는 순서가 실행마다 바뀐다.**
부동소수점 덧셈은 교환법칙은 성립하지만 **결합법칙은 성립하지 않는다**
(`(a+b)+c != a+(b+c)`). 그래서 순서가 바뀌면 마지막 비트가 달라지고,
그 차이가 수십 epoch의 학습을 거치며 증폭된다.

CPU에서도 재현되지 않았다. OpenBLAS가 리덕션을 여러 스레드로 쪼개면
같은 일이 일어난다. 같은 seed로 예측값 최대 차이가 36.8까지 벌어졌다.

```
cuda: identical=False  max abs diff: 36.8
cpu : identical=False
```

### 해결

세 가지를 같이 해야 했다.

```python
torch.use_deterministic_algorithms(True)   # 결정적 커널만 쓴다
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # GEMM 리덕션 고정
DataLoader(..., generator=seeded_generator(seed))   # 배치 순서 고정
```

`CUBLAS_WORKSPACE_CONFIG`는 cuBLAS가 핸들을 만들 때 읽으므로 **torch를
import하기 전에** 세워야 한다. 그래서 `scripts/run_models.py` 맨 위에 둔다.

DataLoader에 생성기를 따로 주는 이유는, 안 주면 전역 RNG를 쓰기 때문이다.
그러면 앞에서 난수를 몇 번 뽑았는지에 따라 배치 순서가 바뀐다.

확인:

```
cuda: identical=True  maxdiff=0.000e+00
cpu : identical=True  maxdiff=0.000e+00
```

전체 격자를 결정적 모드로 돌린 뒤 Clearance만 한 번 더 돌려
평균·표준편차·멈춘 epoch이 모두 같은 것을 확인했다.

```
1회차: gin 0.524+/-0.040 (epochs 69|61|41|16|103)
2회차: gin 0.524+/-0.040 (epochs 69|61|41|16|103)
```

결정적 모드는 느리다(Clearance gin 44초 → 62초). 재현되지 않는 숫자는
리더보드에 쓸 수 없으니 받아들일 비용이다.

### 1주차 결과는 영향 없다

트리 모델은 이 경로를 타지 않는다. 1주차 베이스라인은 이미 세 환경
(클라우드 Linux, WSL2, Windows)에서 일치를 확인했다.

### 교훈

seed 고정은 재현성의 **필요조건일 뿐**이다. 병렬 리덕션이 들어가는 연산이
있으면 seed를 고정해도 재현되지 않는다. "두 번 돌려 같은지" 확인하지
않으면 알 수 없고, 한 번 돌린 숫자로 "유의한 개선"을 주장하면 틀릴 수 있다.
실제로 그렇게 틀렸다 — 처음에 Clearance GIN을 "2 SD 이상 개선"이라고
적었는데, 그 ±0.026은 한 실행 안의 seed 분산이었고 실행을 다시 하면
0.530~0.567로 움직였다.

## 2. GIN에 descriptor를 넣었다

2주차 첫 결과에서 그래프만 보는 GIN은 구조가 지배하는 Clearance에서
가장 셌고, 분자 전체의 물리화학적 성질이 지배하는 Caco2에서 가장 약했다.
그 성질을 직접 넣어 준다.

descriptor는 **분자 하나당 한 벡터**라서 원자 단위가 아니라 **readout
뒤에** 붙인다. 그래프 임베딩 128차원 + descriptor 27차원을 이어서 head에
넣는다.

`FeatureScaler`를 새로 만들어 열마다 정규화한다. descriptor는 열끼리
스케일이 전혀 다르다(MolWt 수백, qed 0~1, BertzCT 수천). 정규화 없이
신경망에 넣으면 큰 열이 gradient를 독차지해서 작은 열은 사실상 무시된다.
**1주차 트리 모델에는 없던 문제다** — 트리는 분할 기준만 보기 때문이다.
통계는 train에서만 구해 세 split에 같이 적용한다.

### 결과 (결정적 모드, 5 seeds)

| 과제 | 지표 | 1주차 rf desc+fp | MLP | GIN | GIN+desc |
| --- | --- | --- | --- | --- | --- |
| Caco2_Wang | MAE ↓ | **0.279 ± 0.003** | 0.288 ± 0.012 | 0.434 ± 0.033 | 0.343 ± 0.026 |
| PPBR_AZ | MAE ↓ | 8.657 ± 0.020 | 8.283 ± 0.245 | 8.240 ± 0.527 | **7.982 ± 0.350** |
| Clearance | Spearman ↑ | 0.506 ± 0.002 | 0.508 ± 0.025 | 0.524 ± 0.040 | **0.562 ± 0.027** |

**descriptor를 넣으면 세 과제 모두 GIN이 좋아진다.** 예외가 없다.

| 과제 | gin → gin+desc |
| --- | --- |
| Caco2 MAE | 0.434 → 0.343 |
| PPBR MAE | 8.240 → 7.982 |
| Clearance Spearman | 0.524 → 0.562 |

과제별 최고는 이렇게 갈린다.

- **PPBR**: gin+desc 7.982가 RandomForest 8.657보다 0.675 낫다. 표준편차
  0.350 대비 약 1.9배.
- **Clearance**: gin+desc 0.562가 RandomForest 0.506보다 0.056 높다.
  표준편차 0.027 대비 약 2배.
- **Caco2**: 여전히 RandomForest(0.279)가 가장 좋다. descriptor를 넣어
  0.434에서 0.343으로 크게 좁혔지만 못 넘었다. MLP 0.288이 신경망 중
  가장 가깝다.

seed 5개의 표준편차 대비 2배 수준은 **일관된 방향**이라고 말할 수 있지만
엄밀한 유의성 주장은 아니다. seed 5개는 리더보드 제출 형식이라 그에
맞췄고, 더 강하게 말하려면 seed를 늘려야 한다.

표준편차가 여전히 트리 모델보다 한 자릿수 크다(0.026~0.527 대
0.002~0.020). GNN은 seed에 민감하고, PPBR의 한 seed는 epoch 8에서
멈췄다. "GNN이 이겼다"만 쓰면 이 안정성 차이를 빠뜨리는 것이 된다.

### 읽기

Caco2는 투과도, PPBR은 단백결합, Clearance는 대사 청소율이다.
1주차에서 fingerprint가 descriptor를 이긴 과제는 Clearance 뿐이었고,
대사는 특정 대사 취약부위가 결정하기 때문이라고 읽었다.

그 읽기와 지금 결과가 맞물린다. 구조만 보는 GIN은 Clearance에서 가장
세고 Caco2에서 가장 약하다. descriptor를 더하면 세 과제 모두 좋아지는데
**Caco2에서 개선 폭이 가장 크다**(0.434 → 0.343). 그 과제가 물리화학적
성질을 가장 많이 필요로 하기 때문이다. 입력이 과제의 물리에 맞을 때만
이긴다는 것이 네 모델에서 일관됐다.

## 3. test를 보고 고르지 않았다

하이퍼파라미터는 데이터 크기만 보고 미리 정했다(batch 64, AdamW
lr 1e-3, weight decay 1e-5, Huber loss delta 1.0, patience 40).
epoch 선택은 전부 valid로 했고, `results/models.csv`에 seed별 valid
점수와 멈춘 epoch을 남겨 근거를 볼 수 있게 했다.

Caco2에서 신경망이 RandomForest를 못 이기는 것이 아쉽지만, 그걸 이기려고
지금 하이퍼파라미터를 만지면 test를 보고 고르는 것이 된다. 개선은
모델 구조를 바꾸고 valid로 판단하는 방향으로만 한다. descriptor 추가가
그 예다.

## 4. 아직 안 끝난 것

- **DDP 검증.** 코드는 `scripts/train_ddp.py`에 있고 틀리기 쉬운 네
  지점을 주석으로 표시했다. 이 PC는 GPU가 한 장이라 `world_size=1`
  이상을 확인할 수 없다. 실제 프로젝트 진행 시로 미룬다.
