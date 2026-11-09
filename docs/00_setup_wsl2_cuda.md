# WSL2 + CUDA 환경 만들기 (RTX 3070, Windows)

가장 막히기 쉬운 단계다. 순서대로만 하면 된다.

## 1. Windows 쪽 준비

1. **NVIDIA 드라이버를 Windows에 설치**한다 (GeForce Experience 또는 nvidia.com).
   WSL용 CUDA는 Windows 드라이버가 그대로 전달된다.
2. **WSL 안에는 리눅스용 NVIDIA 드라이버를 절대 설치하지 않는다.** 설치하면
   GPU 전달이 깨진다. 이게 가장 흔한 실수다.
3. PowerShell(관리자)에서 WSL2와 Ubuntu를 설치한다.

   ```powershell
   wsl --install -d Ubuntu
   wsl --update
   ```

## 2. WSL 안에서 GPU가 보이는지 확인

Ubuntu 터미널에서:

```bash
nvidia-smi
```

RTX 3070과 드라이버 버전이 나오면 성공이다. 안 나오면 Windows 드라이버를
업데이트하고 `wsl --shutdown` 후 다시 연다.

## 3. 파이썬 환경

Miniforge(conda)를 권한다.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
exec $SHELL -l
```

프로젝트 환경을 만든다.

```bash
cd ~/PK-ADMET-Predictor          # 아래 5번 참고
conda env create -f environment.yml
conda activate pk-admet
```

## 4. PyTorch는 CUDA 빌드로 따로 설치

`environment.yml`에는 들어 있지 않다. 기본 wheel을 받으면 CPU 버전이 와서
GPU를 못 쓴다.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

확인:

```bash
python scripts/check_env.py
```

`cuda available : True`와 `NVIDIA GeForce RTX 3070`이 나와야 한다.

## 5. 프로젝트 파일을 어디에 둘까

- **권장: WSL 리눅스 홈(`~/PK-ADMET-Predictor`)**. `/mnt/c/...`는 파일
  입출력이 몇 배 느려서, RDKit 특징 추출과 데이터 캐싱에서 체감된다.
- Windows의 `C:\Users\flyop\source\PK-ADMET-Predictor`는 전달용 원본이다.
  WSL에서 한 번 복사해 쓰면 된다.

  ```bash
  cp -r /mnt/c/Users/flyop/source/PK-ADMET-Predictor ~/
  cd ~/PK-ADMET-Predictor
  ```

- VS Code를 쓰면 `code .`로 Remote-WSL로 열린다.

## 6. git

이 폴더는 아직 git 저장소가 아니다. 직접 초기화한다.

```bash
cd ~/PK-ADMET-Predictor
git init
git add .
git commit -m "프로젝트 뼈대: TDC ADMET 세 과제 베이스라인"
```

GitHub에 올릴 때는 TDC가 내려받는 원본 데이터가 따라가지 않도록
`.gitignore`를 먼저 확인한다. 저장소에는 요약 csv와 그림만 올린다.

## 막혔을 때

| 증상 | 확인할 것 |
| --- | --- |
| WSL2가 안 올라감 (`HCS_E_HYPERV_NOT_INSTALLED`) | BIOS에서 가상화가 꺼져 있다. `systeminfo`의 `Virtualization Enabled In Firmware`를 보고, AMD면 SVM Mode / Intel이면 VT-x를 켠다 |
| `nvidia-smi` 없음 | Windows 드라이버 설치, `wsl --update`, `wsl --shutdown` |
| Clearance 과제만 traceback 없이 죽음 | Windows 네이티브 환경의 MKL 문제. `conda install "libblas=*=*openblas"`. WSL에서는 안 난다. `docs/02_week1_issues.md` 5번 |
| Windows에서 `conda env create`가 `tiledbsoma`에서 실패 | Windows 휠이 없다. WSL을 쓰는 게 정답이고, 꼭 Windows여야 하면 `docs/02_week1_issues.md` 3번의 우회를 따른다 |
| `cuda available : False` | PyTorch를 cu124 index로 받았는지 (`torch.__version__`에 `+cu`가 있어야 한다) |
| RDKit import 실패 | conda-forge 채널로 설치했는지 |
| TDC 다운로드 실패 | 네트워크·방화벽. `data/` 지우고 재시도 |
| GPU 메모리 부족 | batch size를 줄인다. 8GB면 이번 과제엔 넉넉하다 |
