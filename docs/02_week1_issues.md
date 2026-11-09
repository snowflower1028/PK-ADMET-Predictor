# 1주차에 실제로 막혔던 것들 (2026-11-06)

1주차 코드를 처음으로 끝까지 돌리면서 막힌 지점과 해결을 남긴다.
같은 환경을 다시 만들 때 다시 하루를 쓰지 않기 위한 기록이고,
"환경 구성에서 뭐가 어려웠나"에 답할 거리이기도 하다.

실행한 PC: Windows 10 (19045), RTX 3070 8GB, 드라이버 621.01.
**Windows 네이티브와 WSL2 Ubuntu-24.04 두 환경을 다 만들었고, 막히는
지점이 서로 다르다.** 어느 쪽 이야기인지 표에 적어 둔다.

## 요약

| # | 문제 | Windows | WSL2 | 상태 |
| --- | --- | --- | --- | --- |
| 1 | `src/baseline.py`의 `evaluate_many` 오사용 | 해당 | 해당 | 해결 |
| 2 | BIOS 가상화가 꺼져 WSL2가 안 올라감 | 해당 | — | 해결 |
| 3 | `tiledbsoma` 빌드 실패로 PyTDC 설치 중단 | 해당 | 없음 | 우회 |
| 4 | `setuptools<81` (`pkg_resources` 제거) | 해당 | 해당 | 해결 |
| 5 | MKL BLAS에서 `np.corrcoef`가 프로세스를 죽임 | 해당 | 없음 | 해결 |
| 6 | Miniforge를 공백 있는 경로에 설치하면 실패 | 해당 | — | 회피 |
| 7 | LightGBM 두 조합이 Windows에서만 다르게 나옴 | 해당 | — | **미해결** |

---

## 1. `group.evaluate_many()`를 `group.evaluate()`처럼 읽고 있었다

두 함수의 반환 구조가 다르다. 더미 예측을 넣고 직접 찍어 확인했다.

```python
evaluate()      -> {'caco2_wang': {'mae': 0.561}}
evaluate_many() -> {'caco2_wang': [0.561, 0.0]}
```

`evaluate()`는 지표 이름으로 한 겹 더 들어가는데 `evaluate_many()`는
바로 `[평균, 표준편차]` 리스트다. 기존 코드의 `scores[key][metric]`은
리스트를 문자열로 인덱싱하므로 `TypeError`가 난다.

seed 5개 이상일 때만 `evaluate_many()`로 가는 분기라서, 빠른 점검
(`--seeds 1`)에서는 멀쩡하고 **본 실행에서만 터진다.** 고친 코드는
`scores[key]`로 바로 받는다.

## 2. WSL2가 올라가지 않았다 — BIOS 가상화

`wsl --version`은 3.0.3.0으로 멀쩡한데 배포가 하나도 없었고,
설치하면 VM 생성에서 죽었다.

```
wsl --install -d Ubuntu-24.04 --no-launch
→ HCS_E_HYPERV_NOT_INSTALLED
```

원인은 `systeminfo`에 그대로 나온다.

```
VM Monitor Mode Extensions      : Yes   (CPU는 지원)
Virtualization Enabled In Firmware : No (BIOS에서 꺼짐)
```

재부팅 → BIOS → **SVM Mode**(AMD) 또는 **Intel VT-x**를 Enabled.
확인은 PowerShell 한 줄이 빠르다.

```powershell
(Get-CimInstance Win32_ComputerSystem).HypervisorPresent   # True면 켜짐
```

## 3. `tiledbsoma` 빌드 실패 (Windows 전용)

`conda env create`의 pip 단계에서 PyTDC 자체는 빌드되는데 의존성에서 멈춘다.

```
Building wheel for tiledbsoma (pyproject.toml): error
Failed to build tiledbsoma
```

`tiledbsoma`는 Windows 휠이 없어 C++ 컴파일이 필요하고, TDC의
**단일세포 모듈** 전용이라 ADMET에는 쓰이지 않는다. Windows에서는
의존성을 건너뛰고 실제로 필요한 것만 넣었다.

```bash
pip install --no-deps PyTDC==1.1.15
pip install fuzzywuzzy python-Levenshtein "huggingface_hub<1.0"
```

**리눅스에서는 이 문제가 없다.** 휠이 있어서 `conda env create`가
그대로 통과하고, 덤으로 PyTDC가 CUDA 되는 torch를 끌고 온다
(`torch 2.15.0+cu132`, `cuda available: True`). 즉 WSL에서는
`docs/00_setup_wsl2_cuda.md` 4번의 `pip install torch --index-url .../cu124`
단계가 필요 없다.

## 4. `setuptools<81` — 양쪽 다 걸린다

```
File "tdc/metadata.py", line 5, in <module>
    import pkg_resources
ModuleNotFoundError: No module named 'pkg_resources'
```

setuptools 81부터 `pkg_resources`가 제거됐는데 PyTDC 1.1.15는 아직
쓴다. **Windows와 WSL2 양쪽에서 똑같이 났다.** 세 문제 중 유일하게
플랫폼을 안 가리므로 `environment.yml`에 넣었다.

## 5. MKL BLAS에서 `np.corrcoef`가 프로세스를 죽였다 (Windows 전용)

가장 오래 걸린 문제다. 증상이 엉뚱한 곳을 가리킨다.

**증상**: 베이스라인 18조합 중 Caco2 6개, PPBR 6개는 정상인데
Clearance에서 **traceback 없이 종료**(exit 127), CSV도 안 만들어짐.

**왜 한 과제만?** Clearance만 지표가 **Spearman**이다. Spearman은
`scipy.stats.spearmanr` → `np.corrcoef`를 타고, MAE인 두 과제는
이 경로를 안 지난다.

**추적**: `faulthandler.enable()`을 붙이니 네이티브 예외가 잡혔다.

```
Windows fatal exception: code 0xc06d007f   (DELAYLOAD_DLL_NOT_FOUND)
  numpy/lib/function_base.py:2747 in cov
  numpy corrcoef  ←  scipy spearmanr  ←  tdc/evaluator.py:488
```

세 줄로 재현된다.

```python
a @ b              # 120.0, 정상
np.corrcoef(a, b)  # 프로세스 즉사
```

**원인**: 환경의 BLAS가 **MKL 2026.1.0**인데 numpy 1.26.4는 훨씬 옛날
MKL에 맞춰 빌드돼 있어, delay-load하는 DLL이 지금 MKL에 없다. numpy가
1.26.4에 묶인 것은 PyTDC가 요구하는 rdkit 2023.09 상한이 numpy 1.x만
지원하기 때문이다. conda 메타데이터는 numpy·scipy 모두
`libblas >=3.9.0,<4.0a0`만 요구해서 **의존성 제약은 만족하는데 런타임에
죽는** 경우다.

**해결**: BLAS를 OpenBLAS로 바꾼다. 버전은 아무것도 안 건드린다.

```bash
conda install -n pk-admet "libblas=*=*openblas"
```

**리눅스에서는 안 난다.** conda-forge가 리눅스에서는 OpenBLAS를 기본으로
주기 때문이다. win-64 전용 문제라 `environment.yml`에 넣지 않고 여기와
`docs/00_setup_wsl2_cuda.md`에 적어 둔다.

교훈 하나: traceback 없이 죽으면 파이썬 예외가 아니다. `faulthandler`를
먼저 붙인다.

## 6. Miniforge를 공백 있는 경로에 설치하면 조용히 실패한다

`C:\Users\Minsoo Lee\miniforge3`로 설치하면 설치 프로그램이 종료 코드도
안 남기고 아무것도 만들지 않는다. 경로의 공백 때문이다.
`D:\miniforge3`처럼 공백 없는 경로를 쓴다.

---

## 7. 아직 안 끝난 것 — LightGBM 재현성

세 환경(클라우드 Linux, 로컬 WSL2, 로컬 Windows)에서 18조합을 다 돌려
비교했다. **두 리눅스 환경은 18개가 전부 일치**한다. Windows만 2개가 다르다.

| 조합 | Linux (클라우드·WSL2) | Windows |
| --- | --- | --- |
| Clearance lgbm desc | 0.274 ± 0.020 | 0.279 ± 0.016 |
| Clearance lgbm desc+fp | 0.486 ± 0.008 | 0.489 ± 0.011 |

RandomForest 9조합은 세 환경 전부 완전 일치한다(sklearn 1.9.1 동일).
차이는 **lightgbm 4.6.0(Windows) vs 4.7.0(Linux)** 때문으로 보인다.
두 경우 다 표준편차 1배 안쪽이고 과제별 최고 조합(rf desc+fp)도 그대로라
결론은 흔들리지 않는다.

그래도 "재현 가능하다"고 말하려면 `environment.yml`에 lightgbm을
고정하는 게 맞다. 지금은 RandomForest 행만 플랫폼 간 비트 단위로
재현된다. **고정할지는 아직 정하지 않았다.**

부수적으로 pandas도 2.1.4(Windows) / 2.3.3(WSL2)로 갈렸는데 상한
`<3` 안쪽이고 결과에 영향이 없었다.

## 그밖에 열려 있는 것

- **`conda env create -f environment.yml`은 Windows에서 그대로는 실패한다**
  (3번). 권장 경로가 WSL2이므로 환경 파일을 Windows에 맞춰 고치지 않고
  문서로 남긴다.
- **WSL 사용자 계정이 아직 없다.** `--no-launch`로 설치해 계정 생성
  프롬프트를 건너뛰었고 root만 있다. conda를 `/opt/miniforge3`에 깔아
  두었으므로 어떤 계정을 만들어도 바로 쓸 수 있다.
  ```bash
  wsl -d Ubuntu-24.04 -u root -- bash -c "adduser minsoo && usermod -aG sudo minsoo"
  ```
- **DDP는 이 PC에서 검증할 수 없다.** GPU가 한 장이다. 계획대로
  Kaggle T4 x2에서 확인한다.
