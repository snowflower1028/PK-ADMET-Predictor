"""재현성 유틸과 결과 저장."""
import json
import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """파이썬·numpy·torch의 난수를 한 번에 고정한다.

    seed만 고정해서는 부족하다는 것을 2주차에 확인했다. 같은 seed로
    두 번 돌려도 GIN 결과가 달라졌다. 원인은 난수가 아니라 **덧셈 순서**다.

    GNN의 메시지 집계는 `index_add_`로 흩어진 값을 더하는데, GPU에서는
    여러 스레드가 atomic으로 같은 자리에 더하므로 **더해지는 순서가
    실행마다 바뀐다.** 부동소수점 덧셈은 교환법칙은 성립하지만
    결합법칙은 성립하지 않아서, 순서가 바뀌면 마지막 비트가 달라진다.
    그 미세한 차이가 수십 epoch 학습을 거치며 증폭된다.

    CPU에서도 BLAS가 여러 스레드로 리덕션을 쪼개면 같은 일이 생긴다.

    deterministic=True면 PyTorch에 결정적 커널만 쓰라고 지시한다.
    느려지지만, 재현되지 않는 숫자는 리더보드에 쓸 수 없다.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cuBLAS의 리덕션 작업공간을 고정해야 GEMM도 결정적이 된다.
            # torch가 CUDA를 초기화하기 전에 설정돼 있어야 한다.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # 1주차에는 torch 없이도 돌아가야 한다


def seeded_generator(seed: int):
    """DataLoader 셔플용 생성기.

    DataLoader에 generator를 넘기지 않으면 전역 RNG를 쓴다. 그러면 앞에서
    난수를 몇 번 뽑았는지에 따라 배치 순서가 바뀐다. 생성기를 따로 주면
    호출 순서와 무관하게 같은 순서가 나온다.
    """
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def save_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
