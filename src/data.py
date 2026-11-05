"""TDC ADMET Benchmark Group 데이터 로드.

평가는 반드시 TDC 공식 split과 group.evaluate()로 한다. test set을 보고
하이퍼파라미터를 고치면 리더보드와 비교할 수 없는 숫자가 된다.
"""
from __future__ import annotations

import pandas as pd

from .config import DATA_DIR


def get_group(path=None):
    """ADMET group 객체를 돌려준다. 첫 호출 때 데이터를 내려받는다."""
    from tdc.benchmark_group import admet_group

    path = str(path or DATA_DIR)
    return admet_group(path=path)


def load_task(group, task_name: str, seed: int) -> dict:
    """한 과제를 train/valid/test로 나눠 돌려준다.

    TDC가 train_val과 test를 고정해 주고, train/valid 분할만 seed로 바뀐다.
    """
    benchmark = group.get(task_name)
    train, valid = group.get_train_valid_split(
        benchmark=task_name, split_type="default", seed=seed
    )
    return {
        "name": benchmark["name"],
        "train": train,
        "valid": valid,
        "test": benchmark["test"],
    }


def xy(df: pd.DataFrame) -> tuple[list[str], pd.Series]:
    """TDC DataFrame에서 SMILES 목록과 타깃을 뽑는다."""
    return df["Drug"].tolist(), df["Y"]
