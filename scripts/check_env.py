"""환경 점검. WSL2에 설치한 뒤 가장 먼저 이것부터 돌린다."""
import importlib
import platform
import sys

CHECKS = ["numpy", "pandas", "sklearn", "lightgbm", "rdkit", "tdc", "torch", "scipy"]


def main() -> int:
    print(f"python   : {sys.version.split()[0]} ({platform.platform()})")
    missing = []
    for name in CHECKS:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "(버전 정보 없음)")
            print(f"{name:9}: {version}")
        except ImportError:
            print(f"{name:9}: 없음")
            missing.append(name)

    try:
        import torch

        print(f"cuda available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda device    : {torch.cuda.get_device_name(0)}")
            print(f"cuda version   : {torch.version.cuda}")
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"gpu memory     : {total:.1f} GiB")
        else:
            print("GPU를 찾지 못했다. WSL2라면 Windows에 NVIDIA 드라이버가")
            print("깔려 있는지, PyTorch를 CUDA 빌드로 받았는지 확인한다.")
    except ImportError:
        print("cuda available : torch 미설치로 확인 불가")

    # RDKit이 실제로 분자를 파싱하는지까지 확인한다
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles("CC(C)NCC(O)COc1ccccc1CC=C")  # 프로프라놀롤 유사체
        print(f"rdkit 파싱 확인 : MolWt={Descriptors.MolWt(mol):.2f}")
    except ImportError:
        print("rdkit 파싱 확인 : rdkit 미설치로 확인 불가")

    if missing:
        print(f"\n빠진 패키지: {', '.join(missing)}")
        return 1
    print("\n모든 패키지 확인 완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
