"""DistributedDataParallel 학습.

왜 별도 파일인가. DDP는 프로세스를 GPU 개수만큼 띄우고 각 프로세스가
데이터의 서로 다른 조각을 보는 방식이다. `scripts/run_models.py`의
단일 프로세스 루프와 구조가 달라서, 억지로 한 파일에 합치면 양쪽 다
읽기 어려워진다.

실행
  torchrun --nproc_per_node=2 scripts/train_ddp.py --task Caco2_Wang --model gin

**이 PC(RTX 3070 한 장)에서는 검증이 안 된다.** world_size=1로는 DDP가
감싸져도 통신이 일어나지 않아 "돌아간다"는 것 말고는 확인되는 게 없다.
GPU가 두 장 이상인 환경이 생기면 그때 확인한다.

DDP에서 틀리기 쉬운 지점 네 개를 주석으로 표시해 두었다. (1) sampler의
set_epoch, (2) 프로세스마다 다른 seed가 되지 않게 하는 것, (3) valid
점수를 프로세스 간에 모으는 것, (4) 저장은 rank 0만 하는 것.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

# torch import 전에 세워야 cuBLAS가 결정적으로 동작한다 (scripts/run_models.py 참고)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TASKS  # noqa: E402
from src.data import get_group, load_task, xy  # noqa: E402
from src.models import make_model  # noqa: E402
from src.torch_data import (  # noqa: E402
    FeatureScaler,
    GraphDataset,
    TargetScaler,
    VectorDataset,
    collate_graphs,
)
from src.train import TrainConfig, _forward, _valid_score  # noqa: E402
from src.utils import set_seed  # noqa: E402


def setup() -> tuple[int, int, int]:
    """torchrun이 넣어 준 환경변수로 프로세스 그룹을 연다."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        # 리눅스·NVIDIA면 nccl이 맞다. Windows에서는 gloo를 써야 한다.
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup(world_size: int) -> None:
    if world_size > 1:
        dist.destroy_process_group()


def all_mean(value: float, world_size: int, device) -> float:
    """프로세스마다 다른 valid 점수를 평균으로 모은다.

    (3) 이걸 빼먹으면 프로세스마다 best epoch이 달라지고, 그때부터
    가중치가 갈라진다. 모든 rank가 같은 판단을 해야 한다.
    """
    if world_size == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / world_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Caco2_Wang", choices=list(TASKS))
    parser.add_argument("--model", default="gin",
                        choices=["mlp", "gin", "gin+desc"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    args = parser.parse_args()

    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    # (2) 모든 프로세스가 같은 seed로 시작해야 초기 가중치가 같다.
    # rank를 섞어 넣으면 프로세스마다 다른 모델에서 출발한다.
    set_seed(args.seed)

    metric = TASKS[args.task]["metric"]
    group = get_group()
    split = load_task(group, args.task, args.seed)
    tr_smiles, tr_y = xy(split["train"])
    va_smiles, va_y = xy(split["valid"])
    tr_y, va_y = tr_y.to_numpy(), va_y.to_numpy()

    scaler = TargetScaler(tr_y)  # train만 보고 만든다

    if args.model == "mlp":
        train_ds = VectorDataset(tr_smiles, scaler.transform(tr_y))
        valid_ds = VectorDataset(va_smiles, scaler.transform(va_y))
        collate = None
        model = make_model("mlp", n_features=train_ds.n_features)
    else:
        train_ds = GraphDataset(tr_smiles, scaler.transform(tr_y))
        valid_ds = GraphDataset(va_smiles, scaler.transform(va_y))
        # descriptor 통계도 train에서만 구한다
        desc_scaler = (
            FeatureScaler(train_ds.desc_raw) if args.model == "gin+desc" else None
        )
        train_ds.set_desc_scaler(desc_scaler)
        valid_ds.set_desc_scaler(desc_scaler)
        collate = collate_graphs
        model = make_model(args.model, n_features=train_ds.n_desc)

    # 각 프로세스는 데이터의 서로 다른 조각만 본다.
    sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler, collate_fn=collate,
    )
    # valid는 나누지 않는다. 데이터가 작아서 쪼개는 이득이 없고,
    # 모든 rank가 같은 valid 전체를 보면 판단이 어긋날 일이 없다.
    valid_loader = DataLoader(
        valid_ds, batch_size=256, shuffle=False, collate_fn=collate
    )

    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=TrainConfig.weight_decay
    )
    loss_fn = nn.HuberLoss(delta=1.0)

    best_score, best_epoch, bad = -np.inf, 0, 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        # (1) set_epoch을 부르지 않으면 모든 epoch에서 같은 순서로 섞인다.
        # 조용히 학습이 나빠지는 전형적인 DDP 실수다.
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in train_loader:
            pred, y = _forward(model, batch, args.model, device)
            loss = loss_fn(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for batch in valid_loader:
                p, _ = _forward(model, batch, args.model, device)
                preds.append(p.detach().cpu().numpy())
        va_pred = scaler.inverse(np.concatenate(preds))
        score = all_mean(_valid_score(va_y, va_pred, metric), world_size, device)

        if score > best_score + TrainConfig.min_delta:
            best_score, best_epoch, bad = score, epoch, 0
            inner = model.module if world_size > 1 else model
            best_state = copy.deepcopy(inner.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                break

        if is_main and epoch % 20 == 0:
            print(f"epoch {epoch:3d}  valid {metric}={score:+.4f} "
                  f"(best {best_score:+.4f} @ {best_epoch})", flush=True)

    # (4) 저장은 rank 0만. 모든 프로세스가 같은 경로에 쓰면 파일이 깨진다.
    if is_main:
        print(f"\n{args.task} {args.model} seed {args.seed}: "
              f"best valid {metric}={best_score:+.4f} @ epoch {best_epoch} "
              f"(world_size={world_size})", flush=True)
        if best_state is not None:
            out = Path(__file__).resolve().parents[1] / "results" / "ddp_checkpoint.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "task": args.task,
                        "model": args.model, "seed": args.seed,
                        "valid_score": best_score}, out)
            print(f"체크포인트 저장: {out}", flush=True)
        if world_size == 1:
            print("\n주의: world_size=1이다. DDP 경로는 감싸졌지만 프로세스 간 "
                  "통신이 일어나지 않아 검증된 것이 아니다. GPU 2장에서 "
                  "torchrun --nproc_per_node=2로 확인해야 한다.", flush=True)

    cleanup(world_size)


if __name__ == "__main__":
    main()
