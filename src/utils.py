"""재현성 유틸과 결과 저장."""
import json
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """파이썬·numpy·torch의 난수를 한 번에 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # 1주차에는 torch 없이도 돌아가야 한다


def save_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
