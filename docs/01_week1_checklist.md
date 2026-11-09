# 1주차 체크리스트 (11/5 ~ 11/11)

목표: **환경을 갖추고, 세 과제의 베이스라인 숫자를 손에 넣는다.**
2주차 PyTorch 모델이 이 숫자를 넘는지로 개선을 판단하므로, 여기서 나온
숫자가 프로젝트의 기준선이 된다.

## 할 일

- [ ] WSL2 + CUDA 설치, `python scripts/check_env.py`에서 `cuda available : True`
- [ ] `conda activate pk-admet` 후 패키지 전부 확인
- [ ] TDC 데이터 최초 다운로드 (`data/`에 캐시된다)
- [ ] `notebooks/01_rdkit_intro.ipynb` 처음부터 끝까지 실행
      — RDKit 기초(SMILES 파싱, descriptor, fingerprint, scaffold)를 여기서 익힌다
- [ ] 빠른 확인: 한 과제·한 모델·한 seed로 파이프라인이 도는지
      (seed가 5개 미만이면 리더보드 제출 형식이 아니라 참고용 숫자다)
      ```bash
      python scripts/run_baseline.py --tasks Caco2_Wang --models rf --features desc --seeds 1
      ```
- [ ] 전체 베이스라인 실행 (세 과제 x 2모델 x 3특징 x 5seed)
      ```bash
      python scripts/run_baseline.py
      ```
- [ ] `results/baseline.csv` 확인, 과제별로 가장 좋은 조합 기록
- [ ] git init + 첫 커밋, GitHub 공개 저장소 생성

## 완료 기준

세 과제 모두에서 TDC 공식 평가 함수로 **평균±표준편차**가 나오고,
`results/baseline.csv`에 저장돼 있다.

## 주의

- **test set으로 모델을 고르지 않는다.** TDC가 준 train/valid만 쓴다.
  test 점수를 보고 하이퍼파라미터를 바꾸면 리더보드와 비교할 수 없는 숫자가 된다.
- 남이 읽고 한 줄씩 설명할 수 있는 코드로 쓴다. 이해되지 않는 줄은 그냥
  두지 말고 물어본다.
- 연구실 미공개 데이터·코드는 넣지 않는다. 공개 데이터만 쓴다.

## 예상 소요

| 작업 | 시간 |
| --- | --- |
| WSL2 + CUDA + 환경 | 1~2시간 (처음이면 더) |
| RDKit 노트북 | 2~3시간 |
| 베이스라인 전체 실행 | 30분~2시간 (CPU 코어 수에 따라) |
